import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Ensure MiniHack envs are registered with Gymnasium when this module is imported
try:  # pragma: no cover
    import minihack  # noqa: F401
except Exception:  # pragma: no cover
    minihack = None


class MiniHackWrap(gym.Env):
    """
    Minimal MiniHack wrapper that:
    - Uses built-in agent-centric crop observations (e.g., 'chars_crop' or 'glyphs_crop').
    - Appends goal distance (dx, dy) to the observation (relative to crop center).
    - Shapes reward: step_reward for each step, goal_reward when terminated.

    Assumptions:
    - '*_crop' keys are present in observations (e.g., 'chars_crop', 'glyphs_crop').
    - Goal location is denoted by the '>' character in the 'chars' crop.

    Disclaimer (MiniHack-Corridor-specific tuning):
    - The default compact one-hot vocabulary targets symbols commonly visible in
      MiniHack-Corridor-R2-v0: [' ', '-', '|', '#', '.', '<', '>', '@'] plus an
      optional "other" class. Other MiniHack tasks can expose additional tiles,
      colors, specials, messages, or different goal markers.
    - If you switch environments, consider:
        * Extending `char_vocab` to include the symbols present in that task;
        * Disabling compact mapping by setting `include_other_class=False` and/or
          using a broader vocabulary;
        * Turning off character one-hot (`one_hot=False`) or using `glyphs_crop`
          by setting `use_chars=False`;
        * Adjusting goal detection if the goal character is not '>' via
          `goal_chars=(...)`.
    - This wrapper assumes the env provides full `chars` alongside `*_crop` so
      global goal deltas can be computed; if not, dx/dy fall back to zeros.
    """

    def __init__(
        self,
        env: gym.Env,
        seed: int | None = None,
        view_size: int = 9,
        step_reward: float = -1.0,
        goal_reward: float = 1.0,
        goal_chars: tuple[str, ...] = (">",),
        use_chars: bool = True,
        one_hot: bool = True,
        n_char_classes: int = 256,
        include_dxdy: bool = True,
        char_vocab: tuple[str, ...] | None = None,
        include_other_class: bool = True,
    ):
        super().__init__()
        self.env = env
        self.seed_ = seed
        self.view_size = int(view_size)
        self.step_reward = float(step_reward)
        self.goal_reward = float(goal_reward)
        self.goal_chars = tuple(goal_chars)
        self.use_chars = bool(use_chars)
        self.one_hot = bool(one_hot)
        self.n_char_classes = int(n_char_classes)
        self.include_dxdy = bool(include_dxdy)
        self.include_other_class = bool(include_other_class)

        # Precompute eye for one-hot to avoid reallocs
        self._eye_chars = None
        self._lut_chars = None  # maps ASCII code -> compact index
        if self.one_hot and self.use_chars:
            # Default compact vocabulary tailored for MiniHack-Corridor
            # Visible chars: space, '-', '|', '#', '.', '<', '>', '@'
            if char_vocab is None:
                char_vocab = (" ", "-", "|", "#", ".", "<", ">", "@")
            # Build ASCII code list and LUT to compact indices
            vocab_codes = np.array([ord(c) for c in char_vocab], dtype=np.int64)
            vocab_size = len(vocab_codes) + (1 if self.include_other_class else 0)
            self._eye_chars = np.eye(vocab_size, dtype=np.float32)
            # LUT spans 256 codes; default to 'other' index if enabled else 0
            other_idx = vocab_size - 1 if self.include_other_class else 0
            lut = np.full(256, other_idx, dtype=np.int64)
            for i, code in enumerate(vocab_codes):
                lut[code] = i
            self._lut_chars = lut

        # Delegate action space to underlying env
        self.action_space = env.action_space

        # Infer observation space from env.observation_space without resetting.
        # This avoids triggering MiniHack/NLE reset during VectorEnv construction,
        # which can cause low-level crashes (SIGFPE) in some environments.
        self.last_obs = None

        # Determine crop subspace shape
        crop_subspace = None
        try:
            obs_space = getattr(self.env, "observation_space", None)
            if isinstance(obs_space, spaces.Dict):
                if self.use_chars and "chars_crop" in obs_space.spaces:
                    crop_subspace = obs_space.spaces["chars_crop"]
                elif (not self.use_chars) and "glyphs_crop" in obs_space.spaces:
                    crop_subspace = obs_space.spaces["glyphs_crop"]
                elif self.use_chars and "chars" in obs_space.spaces:
                    crop_subspace = obs_space.spaces["chars"]
                elif "glyphs" in obs_space.spaces:
                    crop_subspace = obs_space.spaces["glyphs"]
        except Exception:
            crop_subspace = None

        if crop_subspace is not None and hasattr(crop_subspace, "shape"):
            crop_shape = tuple(int(x) for x in crop_subspace.shape)
        else:
            # Fallback: assume square crop of view_size if unknown
            crop_shape = (int(self.view_size), int(self.view_size))

        # Compute flattened feature size
        if self.one_hot and self.use_chars:
            vocab_size = self._eye_chars.shape[0] if self._eye_chars is not None else int(self.n_char_classes)
            flat_size = int(np.prod(crop_shape)) * int(vocab_size)
        else:
            flat_size = int(np.prod(crop_shape))
        if self.include_dxdy:
            flat_size += 2

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_size,), dtype=np.float32
        )

        self.spec = getattr(self.env, "spec", None)

    # ----- Observation helpers -----
    def _grid(self, obs) -> np.ndarray:
        # Prefer built-in cropped keys
        if self.use_chars and "chars_crop" in obs:
            return obs["chars_crop"]
        if (not self.use_chars) and "glyphs_crop" in obs:
            return obs["glyphs_crop"]
        # Fallbacks if ever needed
        if self.use_chars and "chars" in obs:
            return obs["chars"]
        return obs.get("glyphs", obs.get("chars"))

    def _agent_xy(self, obs) -> tuple[int, int]:
        bl = obs.get("blstats")
        if bl is None:
            return 0, 0
        return int(bl[0]), int(bl[1])

    def _global_goal_delta(self, obs) -> tuple[float, float]:
        """Compute dx,dy using full observation (not the crop).
        dx = agent_x - goal_x, dy = agent_y - goal_y
        """
        chars = obs.get("chars")
        if chars is None:
            return 0.0, 0.0
        ax, ay = self._agent_xy(obs)
        targets = [ord(c) for c in self.goal_chars]
        ys, xs = np.where(np.isin(chars, targets))
        if xs.size == 0:
            return 0.0, 0.0
        dists = np.abs(xs - ax) + np.abs(ys - ay)
        i = int(np.argmin(dists))
        gx, gy = int(xs[i]), int(ys[i])
        return float(ax - gx), float(ay - gy)

    def _encode_crop(self, crop: np.ndarray) -> np.ndarray:
        # One-hot encode characters if requested
        if self.one_hot and self.use_chars:
            flat_codes = crop.astype(np.int64).reshape(-1)
            if self._lut_chars is not None:
                mapped = self._lut_chars[np.clip(flat_codes, 0, 255)]
                oh = self._eye_chars[mapped]
            else:
                # Fallback to 256-way one-hot
                flat_codes = np.clip(flat_codes, 0, self.n_char_classes - 1)
                oh = self._eye_chars[flat_codes]
            return oh.reshape(-1)
        # Fallback to raw values
        return crop.astype(np.float32).flatten()

    def _build_observation(self, obs) -> np.ndarray:
        grid = self._grid(obs)
        crop = grid  # always crop view
        # For dx,dy compute using full observation (global goal location)
        dx = dy = 0.0
        if self.include_dxdy:
            dx, dy = self._global_goal_delta(obs)
        enc = self._encode_crop(crop)
        if self.include_dxdy:
            enc = np.concatenate([enc, np.array([dx, dy], dtype=np.float32)])
        return enc.astype(np.float32)

    # ----- Gym API -----
    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.seed_ = seed
        obs, info = self.env.reset(seed=self.seed_, options=options)
        self.last_obs = obs
        return self._build_observation(obs), info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        self.last_obs = obs
        reward = self.goal_reward if terminated else self.step_reward
        return self._build_observation(obs), reward, bool(terminated), bool(truncated), info

    def render(self):
        return self.env.render()

    def seed(self, seed: int):
        self.seed_ = seed
        self.env.reset(seed=seed)

    def close(self):
        return self.env.close()


def make_env_minihack(*, env_id: str = "MiniHack-Corridor-R2-v0", seed: int = 0, view_size: int = 9):
    """Vector-friendly builder returning a thunk that constructs the wrapped env."""
    def thunk():
        if minihack is None:
            raise ImportError("minihack is not installed or failed to import; please `pip install minihack`.")
        base = gym.make(
            env_id,
            observation_keys=(
                "chars",
                "glyphs",
                "blstats",
                "chars_crop",
                "glyphs_crop",
            ),
        )
        env = MiniHackWrap(base, seed=seed, view_size=view_size, step_reward=-1.0, goal_reward=1.0)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def get_minihack_env(*, env_id: str = "MiniHack-Corridor-R2-v0", seed: int = 0, view_size: int = 9):
    if minihack is None:
        raise ImportError("minihack is not installed or failed to import; please `pip install minihack`.")
    base = gym.make(
        env_id,
        observation_keys=(
            "chars",
            "glyphs",
            "blstats",
            "chars_crop",
            "glyphs_crop",
        ),
    )
    env = MiniHackWrap(base, seed=seed, view_size=view_size, step_reward=-1.0, goal_reward=1.0)
    return env

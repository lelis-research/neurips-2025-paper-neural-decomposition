#!/usr/bin/env python3
"""
Quick test runner for MiniHack in a SyncVectorEnv.

Creates N parallel MiniHack environments using the project's MiniHack wrapper,
steps with random actions for a few iterations, and prints basic diagnostics.

Usage:
  python Scripts/test_minihack_vectorenv.py \
      --env_id MiniHack-Corridor-R2-v0 \
      --num_envs 4 \
      --steps 20 \
      --seed 0 \
      --view_size 9

Notes:
  - Requires `minihack` to be installed and importable.
  - Uses `gym.vector.SyncVectorEnv` with thunks from `environments.environments_minihack.make_env_minihack`.
  - Each env is wrapped with `RecordEpisodeStatistics` inside the thunk, so
    episode returns/lengths appear in `infos` when episodes end.
"""

from __future__ import annotations

import argparse
import sys
from typing import List

import gymnasium as gym
import numpy as np

try:
    import minihack  # noqa: F401
except Exception:
    minihack = None

from environments.environments_minihack import make_env_minihack


def build_vector_env(env_id: str, num_envs: int, base_seed: int, view_size: int) -> gym.vector.SyncVectorEnv:
    env_fns: List = [
        make_env_minihack(env_id=env_id, seed=base_seed + i, view_size=view_size)
        for i in range(num_envs)
    ]
    # Try with autoreset=True (Gymnasium >= 0.27), else fallback
    try:
        envs = gym.vector.SyncVectorEnv(env_fns, autoreset=True)  # type: ignore[arg-type]
    except TypeError:
        envs = gym.vector.SyncVectorEnv(env_fns)
    return envs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="MiniHack-Corridor-R2-v0")
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--view_size", type=int, default=9)
    args = parser.parse_args()

    if minihack is None:
        print("ERROR: minihack is not installed or failed to import. Try `pip install minihack`.", file=sys.stderr)
        return 1

    print(f"Creating {args.num_envs} x {args.env_id} (view_size={args.view_size}) …")
    envs = build_vector_env(args.env_id, args.num_envs, args.seed, args.view_size)

    obs, infos = envs.reset(seed=args.seed)
    print(f"Vector obs shape: {obs.shape}; dtype={obs.dtype}")
    print(f"Action space: {envs.single_action_space}")

    ep_returns = np.zeros(args.num_envs, dtype=np.float32)
    ep_lengths = np.zeros(args.num_envs, dtype=np.int32)

    for t in range(1, args.steps + 1):
        actions = np.array([envs.single_action_space.sample() for _ in range(args.num_envs)])
        obs, rewards, terminateds, truncateds, infos = envs.step(actions)
        dones = np.logical_or(terminateds, truncateds)

        ep_returns += rewards.astype(np.float32)
        ep_lengths += 1

        print(
            f"t={t:03d} rewards={np.round(rewards, 2)} dones={dones} any_done={bool(dones.any())}"
        )

        # If autoreset=True, final stats are provided in infos["final_info"]
        if isinstance(infos, dict) and "final_info" in infos and infos["final_info"] is not None:
            for i, finfo in enumerate(infos["final_info"]):
                if finfo is None:
                    continue
                # Print RecordEpisodeStatistics if present
                ep_r = finfo.get("episode", {}).get("r")
                ep_l = finfo.get("episode", {}).get("l")
                if ep_r is not None or ep_l is not None:
                    print(f"  env[{i}] episode: return={ep_r} length={ep_l}")
                # Reset local trackers for that env
                ep_returns[i] = 0.0
                ep_lengths[i] = 0

        # For older gymnasium without autoreset/reset_done: reset finished envs
        if dones.any() and ("final_info" not in infos):
            reset_idxs = np.nonzero(dones)[0]
            for i in reset_idxs:
                # If wrapped with RecordEpisodeStatistics, stats may appear in infos per env index
                ep = infos.get("episode")
                if isinstance(ep, np.ndarray) and ep.size > i and isinstance(ep[i], dict):
                    print(
                        f"  env[{i}] episode: return={ep[i].get('r')} length={ep[i].get('l')}"
                    )
                ep_returns[i] = 0.0
                ep_lengths[i] = 0

            reset_done_handled = False
            # Try indices-based reset API if available
            try:
                obs2, reset_infos = envs.reset(indices=reset_idxs)
                if obs2 is not None:
                    obs = obs2
                reset_done_handled = True
            except TypeError:
                pass
            except Exception:
                pass

            # Try per-index reset_at
            if not reset_done_handled and hasattr(envs, "reset_at"):
                try:
                    for i in reset_idxs:
                        # Some APIs return (obs_i, info_i), others just obs_i
                        res = envs.reset_at(int(i))
                        if isinstance(res, tuple) and len(res) == 2:
                            oi, _ = res
                        else:
                            oi = res
                        # Place back into the batched obs (assumes flat vector obs)
                        try:
                            obs[int(i)] = oi
                        except Exception:
                            pass
                    reset_done_handled = True
                except Exception:
                    pass

            # Fallback: reset all envs
            if not reset_done_handled:
                obs, _ = envs.reset()

    envs.close()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

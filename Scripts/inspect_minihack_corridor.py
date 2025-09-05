#!/usr/bin/env python3
"""
Inspect MiniHack Corridor environment: prints action space and raw observation space.

Usage:
    python Scripts/inspect_minihack_corridor.py [--env_id MiniHack-Corridor-R2-v0] [--seed 0]
"""
import argparse
import pprint
from typing import Any

import gymnasium as gym
import numpy as np
import re

# Importing minihack registers its environments with Gym/Gymnasium
try:
    import minihack  # noqa: F401
except Exception:
    minihack = None


def describe_value(name: str, v: Any) -> str:
    if isinstance(v, np.ndarray):
        return f"ndarray shape={v.shape} dtype={v.dtype} min={v.min() if v.size else 'NA'} max={v.max() if v.size else 'NA'}"
    if isinstance(v, (bytes, bytearray)):
        return f"bytes len={len(v)}"
    if isinstance(v, str):
        return f"str len={len(v)} value={v!r}"
    if isinstance(v, (list, tuple)):
        return f"{type(v).__name__} len={len(v)} sample0_type={type(v[0]).__name__ if v else 'NA'}"
    if isinstance(v, dict):
        return f"dict keys={list(v.keys())}"
    return f"{type(v).__name__}: {v}"


def try_action_meanings(env) -> list[str] | None:
    for attr in ("get_action_meanings", "get_action_names", "get_action_descriptions"):
        if hasattr(env, attr):
            try:
                names = getattr(env, attr)()
                if isinstance(names, (list, tuple)) and names:
                    return [str(x) for x in names]
            except Exception:
                pass
    # Try unwrapped
    un = getattr(env, "unwrapped", None)
    for attr in ("get_action_meanings", "get_action_names", "get_action_descriptions"):
        if un is not None and hasattr(un, attr):
            try:
                names = getattr(un, attr)()
                if isinstance(names, (list, tuple)) and names:
                    return [str(x) for x in names]
            except Exception:
                pass
    return None


def _create_env_with_fallback(env_id: str):
    try:
        return gym.make(env_id)
    except Exception as e:
        # If version is missing (e.g., "-v0"), try appending "-v0"
        print(f"Error creating env {env_id}: {e}")
        if not re.search(r"-v\d+$", env_id):
            try:
                return gym.make(env_id + "-v0")
            except Exception:
                raise e
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="MiniHack-Corridor-R2-v0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(args.env_id, "seed", args.seed)

    env = _create_env_with_fallback(args.env_id)
    env.obs_crop_h = 21
    env.obs_crop_w = 21
    print(f"Env: {args.env_id}")
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)

    meanings = try_action_meanings(env)
    if meanings is not None:
        print(f"Action meanings ({len(meanings)}):", meanings)

    obs, info = env.reset(seed=args.seed)
    print("\nRaw observation keys:", list(obs.keys()))

    print("\nPer-key observation summary:")
    for k in sorted(obs.keys()):
        try:
            desc = describe_value(k, obs[k])
        except Exception as e:
            desc = f"<error describing: {e}>"
        print(f"- {k}: {desc}")

    # Pretty-print one sample of initial state for common representations
    if "chars" in obs and isinstance(obs["chars"], np.ndarray):
        chars = obs["chars"]
        h, w = chars.shape
        print(f"\nInitial 'chars' view (h={h}, w={w}):")
        # Convert ASCII codes to printable characters
        show_w = w  # print full width
        for y in range(h):
            row = ''.join(chr(int(c)) if 32 <= int(c) <= 126 else ' ' for c in chars[y, :show_w])
            print(row)

        # Also show cropped agent-centric view similar to our wrapper (manual)
        def center_crop(grid: np.ndarray, ax: int, ay: int, view: int) -> np.ndarray:
            half = view // 2
            pad = np.pad(grid, ((half, half), (half, half)), constant_values=0)
            sy, sx = ay + half, ax + half
            return pad[sy - half : sy + half + 1, sx - half : sx + half + 1]

        # Agent position from blstats (x, y)
        if "blstats" in obs:
            ax, ay = int(obs["blstats"][0]), int(obs["blstats"][1])
            view = 9 if (args is None) else 9
            if view % 2 == 0:
                view += 1
            crop = center_crop(chars, ax, ay, view)
            print(f"\nCropped 'chars' (agent-centric, view_size={view}):")
            for y in range(crop.shape[0]):
                row = ''.join(chr(int(c)) if 32 <= int(c) <= 126 else ' ' for c in crop[y])
                print(row)
            # Also show dx, dy to nearest '>' if present
            gxgy = None
            ys, xs = np.where(crop == ord('>'))
            if xs.size > 0:
                # Translate crop-relative to global
                half = view // 2
                gx, gy = int(ax - half + xs[0]), int(ay - half + ys[0])
                dx, dy = ax - gx, ay - gy
                print(f"dx={dx}, dy={dy} (to nearest '>')")

    if "glyphs" in obs and isinstance(obs["glyphs"], np.ndarray):
        glyphs = obs["glyphs"]
        gh, gw = glyphs.shape
        print(f"\nInitial 'glyphs' sample (top-left crop): shape={glyphs.shape} dtype={glyphs.dtype}")
        crop_h, crop_w = min(gh, 21), min(gw, 79)
        # Print a compact numeric crop
        for y in range(crop_h):
            row_vals = ' '.join(f"{int(v):3d}" for v in glyphs[y, :crop_w])
            print(row_vals)

        # Also show cropped glyphs around the agent (manual)
        if "blstats" in obs:
            ax, ay = int(obs["blstats"][0]), int(obs["blstats"][1])
            view = 9
            half = view // 2
            pad = np.pad(glyphs, ((half, half), (half, half)), constant_values=0)
            sy, sx = ay + half, ax + half
            gcrop = pad[sy - half : sy + half + 1, sx - half : sx + half + 1]
            print(f"\nCropped 'glyphs' (agent-centric, view_size={view}):")
            for y in range(gcrop.shape[0]):
                row_vals = ' '.join(f"{int(v):3d}" for v in gcrop[y])
                print(row_vals)

    # Auto-detect and print any built-in '*_crop' observation keys
    crop_keys = sorted([k for k in obs.keys() if k.endswith("_crop")])
    if crop_keys:
        print("\nDetected built-in cropped observation keys:", crop_keys)
    for k in crop_keys:
        v = obs[k]
        print(f"\nInitial '{k}':")
        if isinstance(v, np.ndarray):
            print(f"shape={v.shape} dtype={v.dtype}")
            # Render common types nicely
            if k == "chars_crop" or k.endswith("tty_chars_crop") or k.endswith("chars_crop"):
                for y in range(v.shape[0]):
                    row = ''.join(chr(int(c)) if 32 <= int(c) <= 126 else ' ' for c in v[y])
                    print(row)
            elif k == "glyphs_crop" or k.endswith("glyphs_crop"):
                for y in range(v.shape[0]):
                    row_vals = ' '.join(f"{int(x):3d}" for x in v[y])
                    print(row_vals)
            else:
                # Fallback to summary view for other crop arrays (e.g., colors_crop, specials_crop)
                flat = v.flatten()
                n = min(flat.size, 100)
                sample = ' '.join(str(int(x)) for x in flat[:n])
                print(f"sample (first {n} values): {sample}")

    if "blstats" in obs and isinstance(obs["blstats"], np.ndarray):
        bl = obs["blstats"].astype(int)
        print("\nInitial 'blstats':", bl.tolist())

    if "message" in obs:
        msg = obs["message"]
        if isinstance(msg, (bytes, bytearray, np.ndarray)):
            try:
                msg_str = bytes(msg).decode("utf-8", errors="ignore")
            except Exception:
                msg_str = str(msg)
        else:
            msg_str = str(msg)
        print("\nInitial 'message':", repr(msg_str))

    env.close()


if __name__ == "__main__":
    main()

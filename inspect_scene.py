import argparse
import json
import os
from pathlib import Path
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--auto_reset_seconds", type=float, default=0.0)
parser.add_argument("--reset_preview_frames", type=int, default=3)
parser.add_argument("--reset_preview_seconds", type=float, default=0.5)
parser.add_argument("--save_reset_depth", action="store_true")
parser.add_argument("--depth_output_dir", type=str, default="debug_outputs/reset_depth")
parser.add_argument("--depth_viz_near", type=float, default=0.15)
parser.add_argument("--depth_viz_far", type=float, default=0.50)
parser.add_argument("--lock_viewport_to_task_camera", action="store_true")
parser.add_argument("--show_camera_marker", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.headless = False
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.cfg import LocalInsertEnvCfg
from env.core import LocalInsertEnv
from env.observations import compute_intermediate_values, get_observations
import __init__ as _env_register  # noqa: F401

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

try:
    from omni.kit.viewport.utility import get_viewport_from_window_name
except ImportError:
    get_viewport_from_window_name = None


def print_scene_state(env):
    compute_intermediate_values(env.unwrapped)
    e = env.unwrapped
    print("peg_tip:", [round(float(x), 4) for x in e.peg_tip_pos[0].cpu()])
    print("hole_top:", [round(float(x), 4) for x in e.hole_top_pos[0].cpu()])
    print("xy_dist:", round(float(torch.norm(e.peg_tip_pos[0, :2] - e.hole_top_pos[0, :2]).cpu()), 4))
    print("z_gap:", round(float((e.peg_tip_pos[0, 2] - e.hole_top_pos[0, 2]).cpu()), 4))


def resolve_depth_output_dir() -> Path:
    base_dir = Path(args.depth_output_dir)
    if not base_dir.is_absolute():
        base_dir = Path(__file__).resolve().parent / base_dir
    session_dir = base_dir / time.strftime("%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def lock_viewport_to_task_camera():
    if args.headless or not args.lock_viewport_to_task_camera or get_viewport_from_window_name is None:
        return
    viewport = get_viewport_from_window_name("Viewport")
    if viewport is None:
        print("[inspect_scene] Could not find active viewport window.")
        return
    camera_path = "/World/envs/env_0/Camera"
    viewport.set_active_camera(camera_path)
    print(f"[inspect_scene] Viewport locked to task camera: {camera_path}")


def save_reset_depth(env, reset_index: int, output_dir: Path | None):
    if not args.save_reset_depth:
        return
    if output_dir is None:
        return

    obs = get_observations(env.unwrapped)
    if "depth" not in obs:
        print("[inspect_scene] No depth observation found; skipped saving reset depth.")
        return

    depth = obs["depth"][0, 0].detach().cpu()
    depth_m = depth * 3.0
    stem = output_dir / f"reset_{reset_index:03d}"

    torch.save(depth, stem.with_suffix(".pt"))

    metadata = {
        "reset_index": reset_index,
        "shape": list(depth.shape),
        "normalized_min": float(depth.min()),
        "normalized_max": float(depth.max()),
        "normalized_mean": float(depth.mean()),
        "meters_min": float(depth_m.min()),
        "meters_max": float(depth_m.max()),
        "meters_mean": float(depth_m.mean()),
        "viz_near_m": float(args.depth_viz_near),
        "viz_far_m": float(args.depth_viz_far),
    }
    stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    viz_near = min(args.depth_viz_near, args.depth_viz_far)
    viz_far = max(args.depth_viz_near, args.depth_viz_far)
    depth_viz = depth_m.clamp(viz_near, viz_far)
    depth_viz = 1.0 - (depth_viz - viz_near) / max(viz_far - viz_near, 1e-6)
    image = (depth_viz.clamp(0.0, 1.0) * 255.0).to(torch.uint8).numpy()
    png_path = stem.with_suffix(".png")
    if Image is not None:
        Image.fromarray(image).save(png_path)
        print(f"[inspect_scene] Saved reset depth files to: {output_dir}")
    elif imageio is not None:
        imageio.imwrite(png_path, image)
        print(f"[inspect_scene] Saved reset depth files to: {output_dir}")
    else:
        print(f"[inspect_scene] No PNG writer available; saved raw depth files to: {output_dir}")


def show_reset_preview(env):
    for _ in range(max(args.reset_preview_frames, 0)):
        env.unwrapped.sim.render()
        simulation_app.update()
    if args.reset_preview_seconds > 0:
        time.sleep(args.reset_preview_seconds)


def main():
    env_cfg = LocalInsertEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.show_camera_proxy = args.show_camera_marker

    env = gym.make("LocalInsert-UR10e-Direct-v0", cfg=env_cfg, render_mode="human")
    reset_index = 0
    depth_output_dir = resolve_depth_output_dir() if args.save_reset_depth else None
    env.reset()
    show_reset_preview(env)
    lock_viewport_to_task_camera()
    save_reset_depth(env, reset_index, depth_output_dir)
    print_scene_state(env)

    last_reset = time.time()
    print("Scene inspector running. Close the Isaac Sim window or press Ctrl+C to exit.")

    try:
        while simulation_app.is_running():
            simulation_app.update()
            if args.auto_reset_seconds > 0 and time.time() - last_reset >= args.auto_reset_seconds:
                reset_index += 1
                env.reset()
                show_reset_preview(env)
                lock_viewport_to_task_camera()
                save_reset_depth(env, reset_index, depth_output_dir)
                print("\nReset scene")
                print_scene_state(env)
                last_reset = time.time()
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

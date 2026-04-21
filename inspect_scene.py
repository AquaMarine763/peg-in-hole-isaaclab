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
parser.add_argument("--save_reset_rgb", action="store_true")
parser.add_argument("--rgb_output_dir", type=str, default="debug_outputs/reset_rgb")
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


def resolve_rgb_output_dir() -> Path:
    base_dir = Path(args.rgb_output_dir)
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


def save_reset_rgb(env, reset_index: int, output_dir: Path | None):
    if not args.save_reset_rgb:
        return
    if output_dir is None:
        return

    obs = get_observations(env.unwrapped)
    if "rgb" not in obs:
        print("[inspect_scene] No rgb observation found; skipped saving reset rgb.")
        return

    gray = obs["rgb"][0, 0].detach().cpu()
    raw_rgb = env.unwrapped._camera.data.output["rgb"][0, ..., :3].detach().cpu().float()
    if torch.max(raw_rgb) > 1.0:
        raw_rgb = raw_rgb / 255.0
    stem = output_dir / f"reset_{reset_index:03d}"

    torch.save(raw_rgb.permute(2, 0, 1).contiguous(), stem.with_name(stem.name + "_rgb").with_suffix(".pt"))
    torch.save(gray, stem.with_name(stem.name + "_gray").with_suffix(".pt"))

    metadata = {
        "reset_index": reset_index,
        "rgb_shape_hwc": list(raw_rgb.shape),
        "rgb_min": float(raw_rgb.min()),
        "rgb_max": float(raw_rgb.max()),
        "rgb_mean": float(raw_rgb.mean()),
        "gray_shape": list(gray.shape),
        "gray_min": float(gray.min()),
        "gray_max": float(gray.max()),
        "gray_mean": float(gray.mean()),
    }
    stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    rgb_image = (raw_rgb.clamp(0.0, 1.0) * 255.0).to(torch.uint8).numpy()
    gray_image = (gray.clamp(0.0, 1.0) * 255.0).to(torch.uint8).numpy()
    rgb_png_path = stem.with_name(stem.name + "_rgb").with_suffix(".png")
    gray_png_path = stem.with_name(stem.name + "_gray").with_suffix(".png")
    if Image is not None:
        Image.fromarray(rgb_image).save(rgb_png_path)
        Image.fromarray(gray_image).save(gray_png_path)
        print(f"[inspect_scene] Saved reset raw+gray rgb files to: {output_dir}")
    elif imageio is not None:
        imageio.imwrite(rgb_png_path, rgb_image)
        imageio.imwrite(gray_png_path, gray_image)
        print(f"[inspect_scene] Saved reset raw+gray rgb files to: {output_dir}")
    else:
        print(f"[inspect_scene] No PNG writer available; saved raw+gray rgb tensors to: {output_dir}")


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
    rgb_output_dir = resolve_rgb_output_dir() if args.save_reset_rgb else None
    env.reset()
    show_reset_preview(env)
    lock_viewport_to_task_camera()
    save_reset_rgb(env, reset_index, rgb_output_dir)
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
                save_reset_rgb(env, reset_index, rgb_output_dir)
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

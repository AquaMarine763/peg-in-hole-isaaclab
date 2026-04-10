import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--auto_reset_seconds", type=float, default=0.0)
parser.add_argument("--reset_preview_frames", type=int, default=3)
parser.add_argument("--reset_preview_seconds", type=float, default=0.5)
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
from env.observations import compute_intermediate_values
import __init__ as _env_register  # noqa: F401


def print_scene_state(env):
    compute_intermediate_values(env.unwrapped)
    e = env.unwrapped
    print("peg_tip:", [round(float(x), 4) for x in e.peg_tip_pos[0].cpu()])
    print("hole_top:", [round(float(x), 4) for x in e.hole_top_pos[0].cpu()])
    print("xy_dist:", round(float(torch.norm(e.peg_tip_pos[0, :2] - e.hole_top_pos[0, :2]).cpu()), 4))
    print("z_gap:", round(float((e.peg_tip_pos[0, 2] - e.hole_top_pos[0, 2]).cpu()), 4))


def show_reset_preview(env):
    for _ in range(max(args.reset_preview_frames, 0)):
        env.unwrapped.sim.render()
        simulation_app.update()
    if args.reset_preview_seconds > 0:
        time.sleep(args.reset_preview_seconds)


def main():
    env_cfg = LocalInsertEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    env = gym.make("LocalInsert-UR10e-Direct-v0", cfg=env_cfg, render_mode="human")
    env.reset()
    show_reset_preview(env)
    print_scene_state(env)

    last_reset = time.time()
    print("Scene inspector running. Close the Isaac Sim window or press Ctrl+C to exit.")

    try:
        while simulation_app.is_running():
            simulation_app.update()
            if args.auto_reset_seconds > 0 and time.time() - last_reset >= args.auto_reset_seconds:
                env.reset()
                show_reset_preview(env)
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

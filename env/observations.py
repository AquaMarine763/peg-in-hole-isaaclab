from __future__ import annotations

import torch

from isaacsim.core.utils.torch.transformations import tf_combine


def update_force_sensor(env):
    raw_force = env._robot.root_physx_view.get_link_incoming_joint_force()[:, env.ee_body_idx] - env.force_bias
    alpha = env.cfg.force_sensor.smoothing_factor
    env.force_smooth = alpha * raw_force + (1 - alpha) * env.force_smooth
    noise = torch.randn((env.num_envs, 6), device=env.device) * env.cfg.force_sensor.noise_level
    noisy_force = env.force_smooth + noise
    env.force_history = torch.roll(env.force_history, shifts=-1, dims=1)
    env.force_history[:, -1, :] = noisy_force


def compute_intermediate_values(env, update_force: bool = True):
    env.hole_pos = env._hole.data.root_pos_w - env.scene.env_origins
    env.hole_quat = env._hole.data.root_quat_w
    env.hole_top_pos = env.hole_pos

    env.peg_pos = env._peg.data.root_pos_w - env.scene.env_origins
    env.peg_quat = env._peg.data.root_quat_w
    peg_tip_local = torch.zeros((env.num_envs, 3), device=env.device)
    peg_tip_local[:, 2] = -env.cfg_task.peg.height
    _, env.peg_tip_pos = tf_combine(env.peg_quat, env.peg_pos, env._identity_quat, peg_tip_local)

    env.ee_pos = env._robot.data.body_pos_w[:, env.ee_body_idx] - env.scene.env_origins
    env.ee_quat = env._robot.data.body_quat_w[:, env.ee_body_idx]
    env.ee_linvel = env._robot.data.body_lin_vel_w[:, env.ee_body_idx]
    env.ee_angvel = env._robot.data.body_ang_vel_w[:, env.ee_body_idx]

    jacobians = env._robot.root_physx_view.get_jacobians()
    env.ee_jacobian = jacobians[:, env._jacobian_body_idx, :6, : env._num_arm_dofs]
    env.joint_pos = env._robot.data.joint_pos[:, : env._num_arm_dofs].clone()
    env.joint_vel = env._robot.data.joint_vel[:, : env._num_arm_dofs].clone()

    if update_force:
        update_force_sensor(env)


def get_observations(env):
    compute_intermediate_values(env, update_force=False)
    obs = {
        "ee_z": env.ee_pos[:, 2:3],
        "policy": torch.cat([env.ee_pos, env.ee_quat], dim=-1),
        "hole_state": env.hole_top_pos[:, :2],
    }

    if hasattr(env, "_camera"):
        env._action_step_count += 1
        should_update = env._cached_rgb is None or env._action_step_count % env.cfg.camera_step_interval == 0
        if should_update:
            rgb_raw = env._camera.data.output["rgb"]
            rgb = rgb_raw[..., :3].permute(0, 3, 1, 2).contiguous().float()
            if torch.max(rgb) > 1.0:
                rgb = rgb / 255.0
            gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
            env._cached_rgb = gray.clamp(0.0, 1.0)
        obs["rgb"] = env._cached_rgb

    return obs

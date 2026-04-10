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
    proprio = torch.cat([env.joint_pos, env.joint_vel, env.ee_pos, env.ee_quat], dim=-1)
    phase = env.phase_flag.unsqueeze(-1)
    force_history_masked = env.force_history * phase.unsqueeze(-1)
    force_flat = force_history_masked.reshape(env.num_envs, -1)
    obs = {"policy": torch.cat([proprio, phase, force_flat], dim=-1)}

    if hasattr(env, "_camera"):
        env._action_step_count += 1
        should_update = env._cached_depth is None or env._action_step_count % env.cfg.camera_step_interval == 0
        if should_update:
            depth_raw = env._camera.data.output["depth"]
            depth = depth_raw.permute(0, 3, 1, 2).contiguous()
            depth = torch.clamp(depth, 0.0, 3.0) / 3.0
            env._cached_depth = depth
        obs["depth"] = env._cached_depth

    return obs

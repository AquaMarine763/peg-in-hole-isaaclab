from __future__ import annotations

import torch

import isaacsim.core.utils.torch as torch_utils
from isaacsim.core.utils.torch.transformations import tf_combine


def reset_idx(env, env_ids: torch.Tensor | None):
    super(type(env), env)._reset_idx(env_ids)
    if env_ids is None:
        return

    env.ep_succeeded[env_ids] = 0
    n = len(env_ids)

    joint_pos = env._robot.data.default_joint_pos[env_ids].clone()
    joint_pos[:, : env._num_arm_dofs] = torch.tensor(env.cfg_task.robot_init_joint_pos, device=env.device).unsqueeze(0).repeat(n, 1)
    joint_vel = torch.zeros_like(joint_pos)
    env._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    env._robot.reset(env_ids)
    env._robot.set_joint_position_target(joint_pos, env_ids=env_ids)

    ee_pos_w = env._robot.data.body_pos_w[env_ids, env.ee_body_idx]
    ee_quat_w = env._robot.data.body_quat_w[env_ids, env.ee_body_idx]
    env.nominal_ee_quat[env_ids] = ee_quat_w

    mount_offset = torch.tensor(env.cfg_task.peg_mount_offset, device=env.device).unsqueeze(0).repeat(n, 1)
    mount_angle = torch.full((n,), torch.pi, device=env.device)
    mount_axis = torch.tensor([[1.0, 0.0, 0.0]], device=env.device).repeat(n, 1)
    mount_quat = torch_utils.quat_from_angle_axis(mount_angle, mount_axis)
    peg_root_quat = torch_utils.quat_mul(ee_quat_w, mount_quat)
    peg_root_pos = ee_pos_w + torch_utils.quat_rotate(ee_quat_w, mount_offset)

    peg_tip_local = torch.zeros((n, 3), device=env.device)
    peg_tip_local[:, 2] = -env.cfg_task.peg.height
    _, peg_tip_w = tf_combine(peg_root_quat, peg_root_pos, env._identity_quat[:n], peg_tip_local)

    xy_offset = (2 * torch.rand((n, 2), device=env.device) - 1) * env.cfg_task.reset_xy_offset_range
    z_lo, z_hi = env.cfg_task.reset_z_above_hole
    z_above = z_lo + torch.rand(n, device=env.device) * (z_hi - z_lo)
    hole_state = env._hole.data.default_root_state[env_ids].clone()
    hole_state[:, 0] = peg_tip_w[:, 0] - xy_offset[:, 0]
    hole_state[:, 1] = peg_tip_w[:, 1] - xy_offset[:, 1]
    hole_state[:, 2] = peg_tip_w[:, 2] - z_above
    hole_state[:, 3:7] = env._identity_quat[:n]
    hole_state[:, 7:] = 0.0
    env._hole.write_root_pose_to_sim(hole_state[:, 0:7], env_ids=env_ids)
    env._hole.write_root_velocity_to_sim(hole_state[:, 7:], env_ids=env_ids)
    env._hole.reset(env_ids)

    raw_force = env._robot.root_physx_view.get_link_incoming_joint_force()[env_ids, env.ee_body_idx]
    env.force_bias[env_ids] = raw_force.clone()

    peg_tip_local = peg_tip_w - env.scene.env_origins[env_ids]
    hole_top_pos = hole_state[:, 0:3] - env.scene.env_origins[env_ids]
    env.prev_tip_to_hole_dist[env_ids] = torch.norm(peg_tip_local - hole_top_pos, p=2, dim=-1)
    env.prev_xy_dist[env_ids] = torch.norm(peg_tip_local[:, 0:2] - hole_top_pos[:, 0:2], p=2, dim=-1)
    env.prev_insertion_depth[env_ids] = torch.clamp(hole_top_pos[:, 2] - peg_tip_local[:, 2], min=0.0, max=env.cfg_task.hole.height)
    env.prev_z_gap[env_ids] = peg_tip_local[:, 2] - hole_top_pos[:, 2]
    env.phase_flag[env_ids] = 0.0
    env.contact_hold_count[env_ids] = 0
    env.contact_force_norm[env_ids] = 0.0

    env.prev_ee_pos[env_ids] = 0.0
    env.prev_ee_quat[env_ids] = env._identity_quat[:n]
    env.force_history[env_ids] = 0.0
    env.force_smooth[env_ids] = 0.0
    env.actions[env_ids] = 0.0
    env.prev_actions[env_ids] = 0.0
    env._cached_depth = None

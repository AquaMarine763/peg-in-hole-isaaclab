from __future__ import annotations

import torch

import isaacsim.core.utils.torch as torch_utils
import omni.usd

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import axis_angle_from_quat
from omni.physx.scripts import physicsUtils
from pxr import Gf

from .cfg import LocalInsertEnvCfg
from .observations import compute_intermediate_values, get_observations
from .reset import reset_idx
from .rewards import get_rewards
from .success import get_curr_successes


class LocalInsertEnv(DirectRLEnv):
    cfg: LocalInsertEnvCfg

    def __init__(self, cfg: LocalInsertEnvCfg, render_mode: str | None = None, **kwargs):
        self.cfg_task = cfg.task
        super().__init__(cfg, render_mode, **kwargs)
        self._init_tensors()

    def _init_tensors(self):
        self.ema_factor = self.cfg.ctrl.ema_factor
        self.ee_body_idx = self._robot.body_names.index("wrist_3_link")
        self._jacobian_body_idx = self.ee_body_idx - 1
        self._num_arm_dofs = 6

        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.ee_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.ee_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.ee_quat[:, 0] = 1.0
        self.ee_linvel = torch.zeros((self.num_envs, 3), device=self.device)
        self.ee_angvel = torch.zeros((self.num_envs, 3), device=self.device)
        self.ee_jacobian = torch.zeros((self.num_envs, 6, 6), device=self.device)
        self.prev_ee_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_ee_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.prev_ee_quat[:, 0] = 1.0
        self.nominal_ee_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.nominal_ee_quat[:, 0] = 1.0
        self.hole_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.hole_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.hole_quat[:, 0] = 1.0
        self.hole_top_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.peg_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.peg_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.peg_quat[:, 0] = 1.0
        self.peg_tip_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_tip_to_hole_dist = torch.zeros((self.num_envs,), device=self.device)
        self.prev_xy_dist = torch.zeros((self.num_envs,), device=self.device)
        self.prev_insertion_depth = torch.zeros((self.num_envs,), device=self.device)
        self.prev_z_gap = torch.zeros((self.num_envs,), device=self.device)
        self.phase_flag = torch.zeros((self.num_envs,), device=self.device)
        self.contact_hold_count = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.contact_force_norm = torch.zeros((self.num_envs,), device=self.device)
        self.last_success = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.last_too_far = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.last_time_out = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.joint_pos = torch.zeros((self.num_envs, self._num_arm_dofs), device=self.device)
        self.joint_vel = torch.zeros((self.num_envs, self._num_arm_dofs), device=self.device)
        self.force_history = torch.zeros((self.num_envs, self.cfg.force_sensor.history_len, 6), device=self.device)
        self.force_smooth = torch.zeros((self.num_envs, 6), device=self.device)
        self.force_bias = torch.zeros((self.num_envs, 6), device=self.device)
        self.joint_pos_target = torch.zeros((self.num_envs, self._num_arm_dofs), device=self.device)
        self.ep_succeeded = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self._action_step_count = 0
        self._cached_depth = None

    def _setup_scene(self):
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, 0.0))

        self._robot = Articulation(self.cfg.robot)
        self._hole = RigidObject(self.cfg.hole)
        self._peg = RigidObject(self.cfg.peg)

        if hasattr(self.cfg, "camera"):
            self._camera = TiledCamera(self.cfg.camera)
            self.scene.sensors["camera"] = self._camera

        self._create_peg_fixed_joint()
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["hole"] = self._hole
        self.scene.rigid_objects["peg"] = self._peg

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _create_peg_fixed_joint(self):
        stage = omni.usd.get_context().get_stage()
        source_env = self.scene.env_prim_paths[0]
        mount_rot = Gf.Quatf(0.0, Gf.Vec3f(1.0, 0.0, 0.0))
        mount_offset = Gf.Vec3f(*self.cfg_task.peg_mount_offset)
        physicsUtils.add_joint_fixed(
            stage=stage,
            jointPath=f"{source_env}/Robot/wrist_3_link/peg_fixed_joint",
            actor0=f"{source_env}/Robot/wrist_3_link",
            actor1=f"{source_env}/Peg",
            localPos0=mount_offset,
            localRot0=mount_rot,
            localPos1=Gf.Vec3f(0.0, 0.0, 0.0),
            localRot1=Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)),
            breakForce=1.0e12,
            breakTorque=1.0e12,
        )

    def _pre_physics_step(self, action: torch.Tensor):
        self.actions = self.ema_factor * action.clone().to(self.device) + (1 - self.ema_factor) * self.actions

    def _apply_action(self):
        compute_intermediate_values(self)

        # --- Position ---
        pre_scale = torch.tensor(self.cfg.ctrl.precontact_pos_action_threshold, device=self.device).unsqueeze(0)
        post_scale = torch.tensor(self.cfg.ctrl.postcontact_pos_action_threshold, device=self.device).unsqueeze(0)
        phase = self.phase_flag.unsqueeze(-1)
        pos_scale = pre_scale * (1.0 - phase) + post_scale * phase
        pos_actions = self.actions[:, 0:3] * pos_scale
        ctrl_target_ee_pos = self.ee_pos + pos_actions

        # --- Orientation ---
        # Pre-contact: rotation actions masked to 0, maintain nominal orientation
        # Post-contact: apply policy rotation delta on top of nominal
        rot_scale = torch.tensor(self.cfg.ctrl.postcontact_rot_action_threshold, device=self.device).unsqueeze(0)
        rot_actions = self.actions[:, 3:6] * rot_scale * phase
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / (angle.unsqueeze(-1) + 1e-8)
        rot_delta_quat = torch_utils.quat_from_angle_axis(angle, axis)
        rot_delta_quat = torch.where(
            angle.unsqueeze(-1) > 1e-6,
            rot_delta_quat,
            self._identity_quat,
        )
        ctrl_target_ee_quat = torch_utils.quat_mul(rot_delta_quat, self.nominal_ee_quat)

        # --- 6-DOF Damped Least-Squares IK ---
        pos_error = ctrl_target_ee_pos - self.ee_pos
        quat_error = torch_utils.quat_mul(ctrl_target_ee_quat, torch_utils.quat_conjugate(self.ee_quat))
        quat_error = quat_error * torch.sign(quat_error[:, 0]).unsqueeze(-1)
        rot_error = axis_angle_from_quat(quat_error)

        pose_error = torch.cat([pos_error, rot_error], dim=-1)
        jacobian_full = self.ee_jacobian
        eye6 = torch.eye(6, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        damping_sq = self.cfg.ctrl.ik_damping ** 2
        jj_t = torch.bmm(jacobian_full, jacobian_full.transpose(1, 2))
        inv_term = torch.linalg.inv(jj_t + damping_sq * eye6)
        jacobian_dls = torch.bmm(jacobian_full.transpose(1, 2), inv_term)
        delta_joint_pos = torch.bmm(jacobian_dls, pose_error.unsqueeze(-1)).squeeze(-1)

        self.joint_pos_target = self.joint_pos + delta_joint_pos
        self._robot.set_joint_position_target(self.joint_pos_target, joint_ids=list(range(self._num_arm_dofs)))

    def _get_observations(self) -> dict:
        return get_observations(self)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        compute_intermediate_values(self, update_force=True)
        self._update_phase_state()
        success = get_curr_successes(self)
        xy_dist = torch.norm(self.peg_tip_pos[:, 0:2] - self.hole_top_pos[:, 0:2], p=2, dim=-1)
        too_far = xy_dist > self.cfg_task.too_far_xy_threshold
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        self.last_success = success.clone()
        self.last_too_far = too_far.clone()
        self.last_time_out = time_out.clone()
        return success | too_far, time_out

    def _update_phase_state(self):
        self.contact_force_norm = torch.norm(self.force_smooth[:, 0:3], p=2, dim=-1)
        contact_now = self.contact_force_norm > self.cfg_task.contact_force_threshold
        self.contact_hold_count = torch.where(
            contact_now,
            self.contact_hold_count + 1,
            torch.zeros_like(self.contact_hold_count),
        )
        entered_post = self.contact_hold_count >= self.cfg_task.contact_persistence_steps
        self.phase_flag = torch.where(
            (self.phase_flag > 0.5) | entered_post,
            torch.ones_like(self.phase_flag),
            torch.zeros_like(self.phase_flag),
        )

    def _get_rewards(self) -> torch.Tensor:
        return get_rewards(self)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        reset_idx(self, env_ids)

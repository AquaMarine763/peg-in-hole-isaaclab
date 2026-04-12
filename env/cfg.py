from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")

PEG_RADIUS = 0.015
PEG_HEIGHT = 0.08
HOLE_CLEARANCE = 0.0025
HOLE_DIAMETER = PEG_RADIUS * 2 + HOLE_CLEARANCE * 2
HOLE_DEPTH = 0.05
HOLE_BLOCK_HEIGHT = 0.14


@configclass
class PegCfg:
    usd_path: str = os.path.join(ASSETS_DIR, "peg_cylinder.usda")
    diameter: float = PEG_RADIUS * 2
    height: float = PEG_HEIGHT
    mass: float = 0.1


@configclass
class HoleCfg:
    usd_path: str = os.path.join(ASSETS_DIR, "hole_block.usda")
    diameter: float = HOLE_DIAMETER
    height: float = HOLE_DEPTH
    top_z: float = HOLE_BLOCK_HEIGHT


@configclass
class TaskCfg:
    peg: PegCfg = PegCfg()
    hole: HoleCfg = HoleCfg()
    robot_init_joint_pos: list = [0.0, -1.55, 1.95, -1.97, -1.5708, 0.0]
    hole_workspace_xy_center: list = [0.66, 0.17]
    hole_workspace_xy_half_range: list = [0.04, 0.04]
    max_initial_xy_dist: float = 0.06
    success_xy_tolerance: float = 0.002
    success_depth_fraction: float = 0.8
    too_far_xy_threshold: float = 0.08
    soft_gate_sigma: float = 0.005
    contact_force_threshold: float = 8.0
    contact_persistence_steps: int = 5
    phase_switch_xy_threshold: float = 0.02
    precontact_distance_progress_scale: float = 0.25
    precontact_xy_progress_scale: float = 160.0
    precontact_z_progress_scale: float = 25.0
    precontact_z_gate_sigma: float = 0.015
    precontact_xy_penalty_scale: float = 15.0
    precontact_misaligned_downward_xy_threshold: float = 0.015
    precontact_misaligned_downward_penalty_scale: float = 10.0
    precontact_misaligned_downward_progress_penalty_scale: float = 80.0
    postcontact_xy_progress_scale: float = 80.0
    postcontact_distance_progress_scale: float = 20.0
    postcontact_insertion_progress_scale: float = 320.0
    postcontact_force_penalty_scale: float = 0.05
    action_penalty_scale: float = 0.0015
    success_bonus: float = 100.0
    peg_mount_offset: list = [0.0, 0.0, -0.01]


@configclass
class CtrlCfg:
    ema_factor: float = 0.05
    precontact_pos_action_threshold: list = [0.0005, 0.0005, 0.0008]
    postcontact_pos_action_threshold: list = [0.0008, 0.0008, 0.0015]
    ik_damping: float = 0.05
    postcontact_rot_action_threshold: list = [0.03, 0.03, 0.03]


@configclass
class ForceSensorCfg:
    smoothing_factor: float = 0.25
    noise_level: float = 1.0
    history_len: int = 4


@configclass
class LocalInsertEnvCfg(DirectRLEnvCfg):
    decimation: int = 8
    action_space: int = 6
    observation_space: int = 44
    state_space: int = 0
    episode_length_s: float = 8.0

    task: TaskCfg = TaskCfg()
    ctrl: CtrlCfg = CtrlCfg()
    force_sensor: ForceSensorCfg = ForceSensorCfg()
    camera_step_interval: int = 1
    show_camera_proxy: bool = False

    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1 / 120,
        render_interval=8,
        gravity=(0.0, 0.0, -9.81),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8,
        env_spacing=3.0,
        replicate_physics=True,
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur10e/ur10e.usd",
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.1,
                "elbow_joint": 1.5,
                "wrist_1_joint": -1.97,
                "wrist_2_joint": -1.5708,
                "wrist_3_joint": 0.0,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=800.0,
                damping=40.0,
                effort_limit_sim=150.0,
            ),
        },
    )

    hole: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Hole",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(ASSETS_DIR, "hole_block.usda"),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.50, 0.17, HOLE_BLOCK_HEIGHT), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    peg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Peg",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(ASSETS_DIR, "peg_cylinder.usda"),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.50, -0.07, 0.66),
            rot=(0.926571, 0.239775, -0.072598, -0.280543),
            convention="world",
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        width=84,
        height=84,
        data_types=["depth"],
        update_period=0,
    )

# Peg-in-Hole 强化学习项目进度报告

## 1. 项目概述

本项目使用深度强化学习（Deep RL）训练 UR10e 机械臂完成精密 peg-in-hole 插入任务。系统采用多模态感知（depth camera + force/torque + proprioception），通过 PPO 算法学习从视觉粗对准到力引导插入的完整局部插入策略。

- **仿真平台**：MuJoCo → NVIDIA Isaac Lab（Isaac Sim）
- **训练算法**：PPO（via rsl_rl / OnPolicyRunner）
- **机械臂**：Universal Robots UR10e
- **参考文献**：[Sim2Real for Peg-Hole Insertion with Eye-in-Hand Camera](https://arxiv.org/pdf/2005.14401.pdf)

---

## 2. 项目演进历程

### 阶段一：MuJoCo 环境搭建

- 仿真环境中加入 UR5e 机械臂（替代原先悬浮的 peg），相机改为外部相机
- 训练效果不理想，发现相机存在遮挡问题，调整到不会被遮挡的位置
- 拆分庞大的 `env.py` 为六个子模块（`cfg.py`, `core.py`, `observations.py`, `reset.py`, `rewards.py`, `success.py`），并优化了一版训练模型

### 阶段二：性能瓶颈发现

- 训练速度极慢，定位根因：
  - MuJoCo 物理仿真为纯 CPU 计算
  - 相机渲染 CPU/GPU 同步开销大
  - 视觉数据传入慢，模型一直在等待 visual observation
- 将 MuJoCo 版本上传 GitHub

### 阶段三：Isaac Lab 平台迁移

- 安装并配置 Isaac Lab 环境
- 发现 Isaac Lab 的仿真环境无法在 WSL2 中运行，将项目克隆到 Windows 环境
- 修改项目文件，将全部 MuJoCo 依赖替换为 Isaac Lab 依赖
- 机械臂升级为 UR10e

### 阶段四：GPU 显存优化

- Isaac Sim 对 GPU 显存要求高（本机仅 8GB），采取以下措施：
  - 改为异步渲染（每十步渲染一帧）→ 后恢复为每步一帧
  - 将 RGB 图像改为更适合机器人操作的 depth map（记录每个像素点到相机的距离）
  - 减少并行环境数（从 16 降到 8）以降低 GPU 渲染压力

### 阶段五：Clean 子项目重建

- 训练效果不好且动作幅度过大，重新参考教程，简化控制链
- 新建 `clean_isaaclab_local_insert` 子项目，目标：
  - 只保留局部插入任务（假设视觉粗定位已完成）
  - 控制链简化为：末端局部位移 → 差分 IK → 关节位置目标
  - 策略只控制 3D 平移，姿态固定保持

### 阶段六：奖励工程与训练调优

| 问题 | 解决方案 |
|------|---------|
| Demo 时 peg block 离基座太近 | 修正 peg block 初始位置，hole 工作区前移至 X=0.50 |
| 大部分 episode 因 too_far 终止 | too_far 判定从 40mm 放宽到 80mm |
| 训练效果不好 | 拆分为 pre-contact / post-contact 两阶段，分别给予不同 reward |
| Pre-contact 被几何真值过度塑形 | 移除强下压/XY 真值项，改为弱 distance progress reward，保持视觉主导 |
| Pre-contact 阶段力传感器噪声影响训练 | Pre-contact 屏蔽 force observation（`force_history * phase_flag`） |
| 水平移动过大导致 too_far | 缩小 pre-contact XY action：1mm → 0.5mm，加强 XY penalty（scale=15） |
| Peg 已插入但阶段不切换（需同时满足力检测+向下命令+向下运动） | 放松切换条件：只要力传感器连续 3 步检测到接触就切换 |

---

## 3. 当前系统架构

### 3.1 场景配置

```
UR10e 机械臂
  └── wrist_3_link (末端)
        └── [FixedJoint] Peg (圆柱, 朝下)

地面
  └── Hole Block (固定, 顶面带孔)

外部固定 Depth Camera (侧上方俯视工作区)
```

**核心几何参数：**

| 参数 | 值 |
|------|-----|
| Peg 半径 | 15 mm |
| Peg 高度 | 80 mm |
| Hole clearance | 2.5 mm（单边） |
| Hole 深度 | 50 mm |
| Hole block 高度 | 140 mm |
| Reset XY 偏差范围 | ±8 mm |
| Reset Z 高度（peg tip 在孔口上方） | 10–15 mm |

### 3.2 控制链

```
策略输出 action ∈ [-1, 1]³
    ↓ EMA 平滑 (α = 0.05)
    ↓ Phase-aware 缩放
    ↓   Pre-contact:  [0.5mm, 0.5mm, 2.0mm]
    ↓   Post-contact: [0.8mm, 0.8mm, 1.5mm]
    ↓ 末端位移 → Damped Least-Squares IK (λ = 0.05)
    ↓ 关节位置目标
    ↓ Implicit Actuator (stiffness=800, damping=40)
```

### 3.3 Observation Space

**Policy 向量（44D）：**

| 分量 | 维度 | 说明 |
|------|------|------|
| `joint_pos` | 6 | 关节角度 |
| `joint_vel` | 6 | 关节角速度 |
| `ee_pos` | 3 | 末端位置 |
| `ee_quat` | 4 | 末端姿态四元数 |
| `phase_flag` | 1 | 阶段标志（0=pre-contact, 1=post-contact） |
| `force_history` | 24 | 4 步 × 6D 力/力矩历史（pre-contact 阶段屏蔽为 0） |

**Depth 图像：** 48×48 单通道，clamp 到 [0, 3m] 后归一化到 [0, 1]

### 3.4 两阶段 Reward 设计

**Phase 切换条件：** 力传感器 `force_norm > 5.0 N` 且连续保持 ≥ 3 步

**Reward 公式：**

```
reward = dist_reward + xy_reward + insertion_reward
       - force_penalty - action_penalty - xy_penalty
       + success_bonus

其中：
  pre_mask  = 1 - phase_flag
  post_mask = phase_flag

  dist_reward      = Δdist_3d × (pre_mask × 10.0 + post_mask × 20.0)
  xy_reward        = Δxy_dist × 80.0 × pre_mask
  insertion_reward  = Δinsertion_depth × 320.0 × alignment_gate × post_mask
  force_penalty    = force_norm × 0.01 × post_mask
  action_penalty   = ‖action‖₂ × 0.0015
  xy_penalty       = xy_dist × 15.0 × pre_mask
  success_bonus    = 100.0 （当 xy < 2mm 且插入深度 > 80% hole depth）
```

- `alignment_gate`：仅当 XY 偏差 < 3mm 时开启 insertion reward
- Pre-contact 阶段完全屏蔽 force penalty，避免噪声干扰

### 3.5 终止条件

| 条件 | 判定规则 |
|------|---------|
| **Success** | XY 偏差 < 2mm **且** 插入深度 > 40mm (80% × 50mm) |
| **Too Far** | XY 偏差 > 80mm |
| **Timeout** | episode 时长 > 8.0s |

### 3.6 PPO 训练配置

| 参数 | 值 |
|------|-----|
| `num_steps_per_env` | 128 |
| `max_iterations` | 1000 |
| `learning_rate` | 1e-4 (adaptive schedule) |
| `gamma` | 0.995 |
| `lam` (GAE λ) | 0.95 |
| `clip_param` | 0.2 |
| `num_learning_epochs` | 8 |
| `num_mini_batches` | 4 |
| `desired_kl` | 0.008 |
| `num_envs` | 8 |
| `sim dt` | 1/120 s |
| `decimation` | 8 |

**Actor**：CNN（[32, 64, 64] channels, kernel [8, 4, 3], stride [4, 2, 1]）→ MLP [256, 128]，输入 policy + depth

**Critic**：MLP [512, 256, 128]，仅输入 policy 向量

---

## 4. 关键技术决策

| 决策 | 理由 |
|------|------|
| MuJoCo → Isaac Lab | MuJoCo 纯 CPU 仿真 + CPU/GPU 同步瓶颈导致训练极慢；Isaac Lab 支持 GPU 并行物理+渲染 |
| RGB → Depth map | 深度图信息密度更高、维度更低，更适合机器人操作任务；降低 GPU 渲染压力 |
| 两阶段 reward（pre/post-contact） | 接触前视觉主导学对准，接触后力主导学插入，避免信号冲突 |
| Pre-contact 屏蔽 force | 未接触时力传感器读数为噪声，会干扰训练 |
| Action 幅度 1mm → 0.5mm | 防止 pre-contact 阶段水平移动过大导致 too_far |
| Phase 切换条件简化 | 原先要求力检测+向下命令+向下运动三条件同时满足，导致 peg 已插入但不切换；改为纯力阈值持续检测 |
| 控制链简化（去掉 torque-level OSC） | 更接近教程风格，减少调试难度，让 IK 控制更稳定 |

---

## 5. 当前进展

- ✅ Isaac Lab 环境搭建完成，clean 子项目结构清晰
- ✅ 两阶段 reward 框架和 phase switching 逻辑实现
- ✅ 烟雾测试全部通过（1 env × 1 iteration）
- ✅ 短训练（16 envs × 20 iterations）已跑通
- ✅ Demo 脚本（`play.py`）和场景检查工具（`inspect_scene.py`）可用
- ⏳ 策略已学会局部靠近（XY 稳定在 10-13mm），但插入深度接近 0
- ⏳ Too_far 终止比例已显著降低，但仍是主要失败模式

---

## 6. 后续计划

1. 继续调优 reward 参数，提升插入深度学习效果
2. 考虑升级 GPU 显存（当前 8GB 限制了并行环境数）后进行大规模训练
3. 进一步优化 reset 几何和 pre-contact 动作策略
4. 训练收敛后进行 domain randomization 以支持 sim-to-real transfer

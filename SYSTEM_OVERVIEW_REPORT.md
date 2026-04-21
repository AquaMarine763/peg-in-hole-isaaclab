# Peg-in-Hole Isaac Lab — 系统架构与训练流程汇报

## 1. 项目定位

本项目使用深度强化学习，在 NVIDIA Isaac Lab 仿真环境中训练 UR10e 机械臂完成 peg-in-hole 精密插入任务。

当前任务定义是：

- 机器人从固定初始姿态出发
- peg 初始保持竖直
- hole 在一个受限工作区内随机出现
- 策略需要先通过外部固定相机的视觉输入完成局部找孔与接近
- 然后再完成接触后的姿态修正与插入

整体目标是验证：在不假设粗定位精确的前提下，视觉引导的强化学习能否完成从"搜索"到"插入"的完整局部任务。

---

## 2. 系统整体架构

### 2.1 硬件与平台

| 项目 | 当前配置 |
|------|----------|
| 仿真平台 | NVIDIA Isaac Lab / Isaac Sim |
| 训练算法 | PPO（RSL-RL） |
| 机器人 | Universal Robots UR10e |
| GPU | RTX 5070 Laptop（8GB 显存） |

### 2.2 场景组成

场景包含五个核心元素：

1. **UR10e 机械臂**：6 自由度关节驱动
2. **Peg**：通过固定关节挂在末端，∅30mm × 80mm
3. **Hole Block**：底面贴地的大尺寸块体，顶面有 ∅35mm 圆孔
4. **外部固定相机**：从工作区侧上方拍摄，提供 128×128 RGB 图像
5. **力/力矩传感器**：读取腕部关节力，用于接触检测

### 2.3 控制链

策略不直接控制电机力矩，而是通过一条逐步转换的控制链：

```
策略输出 6D 动作（归一化）
  ↓
按阶段缩放为末端位移/旋转目标
  ↓
6-DOF Damped Least-Squares 逆运动学
  ↓
关节位置目标
  ↓
Isaac Sim 关节驱动器执行
```

其中：
- pre-contact 阶段：只放开 3D 平移，旋转被 mask 为零
- post-contact 阶段：同时放开 3D 旋转，允许策略做插入修正

---

## 3. 观测与网络结构

### 3.1 观测空间

当前采用 **asymmetric actor-critic** 设计：

| 角色 | 输入 | 说明 |
|------|------|------|
| **Actor** | policy 向量 + 灰度 RGB 图 | 部署时可获取的真实输入 |
| **Critic** | policy 向量 + hole XY 真值 | 仅训练时使用的特权信息 |

#### policy 向量（44D）

| 分量 | 维度 | 说明 |
|------|------|------|
| 关节角度 | 6 | |
| 关节速度 | 6 | |
| 末端位置 | 3 | |
| 末端姿态 | 4 | 四元数 |
| 阶段标志 | 1 | 0 = pre-contact，1 = post-contact |
| 力历史 | 24 | 4 步 × 6D，pre-contact 阶段屏蔽 |

#### 视觉输入

- 来源：外部固定 RGB 相机
- 处理：转为灰度单通道，归一化到 [0, 1]
- 分辨率：128 × 128
- 仅 actor 使用

#### Critic 特权输入

- `hole_xy`：hole 在平面上的真实坐标
- 目的：帮助 critic 在 pre-contact 阶段更准确地评估状态价值

### 3.2 网络结构

| 网络 | 结构 | 输入 | 输出 |
|------|------|------|------|
| **Actor** | CNN → MLP [256, 128] | policy + 灰度 RGB | 6D 动作分布 |
| **Critic** | MLP [512, 256, 128] | policy + hole_xy | 标量 value |

---

## 4. 两阶段奖励设计

当前 reward 围绕"先找孔，再下压，再插入"设计。

### 4.1 阶段切换机制

从 pre-contact 切到 post-contact 必须同时满足：

- 接触力 > 8N
- XY 偏差 < 20mm
- 持续 5 步

这样设计的目的是：阻止策略通过"偏着撞 block"直接切换到 post-contact 阶段。

### 4.2 Pre-contact 阶段（视觉搜索/接近）

核心原则：先奖励横向找孔，XY 对准后才逐步放开向下接近的奖励。

| 奖励/惩罚项 | 说明 |
|-------------|------|
| XY 进步奖励 | 高权重主信号，奖励横向接近 hole |
| Z 进步奖励 | 乘以 alignment gate，只有 XY 变好才值钱 |
| 3D 距离奖励 | 极弱，也乘 alignment gate |
| XY 软惩罚 | 对较大 XY 误差持续施加 |
| 未对准下压惩罚 | XY > 15mm 时惩罚向下动作 |
| 未对准下压进步惩罚 | XY > 15mm 时惩罚实际向下位移 |

其中：

```
alignment gate = exp(-xy² / (2σ²))，σ = 15mm
```

### 4.3 Post-contact 阶段（接触修正/插入）

| 奖励/惩罚项 | 说明 |
|-------------|------|
| XY 进步奖励 | 接触后继续修正横向偏差 |
| 3D 距离奖励 | 辅助整体接近 |
| 插入进步奖励 | 乘以 soft gate，非完美对齐也有梯度 |
| 接触力惩罚 | 抑制暴力接触 |

### 4.4 共享项

| 项 | 说明 |
|----|------|
| 动作惩罚 | 约束动作幅度 |
| 成功奖励 | XY < 2mm 且插入深度 > 80% 时触发 |

---

## 5. 工作区与初始化

### 5.1 几何尺寸

| 参数 | 值 |
|------|------|
| Peg 直径 | 30 mm |
| Peg 高度 | 80 mm |
| Hole 直径 | 35 mm |
| 单边间隙 | 2.5 mm |
| Hole 深度 | 50 mm |
| Block 尺寸 | 180 × 180 × 140 mm |

### 5.2 初始化策略

- 机器人：固定 ready pose
- Peg：竖直朝下
- Hole：在受限 workspace 内随机：
  - `x ∈ [0.62, 0.70]`
  - `y ∈ [0.13, 0.21]`
  - `z = 0.14`（固定）
- 约束：初始 XY 偏差 ≤ 60mm
- 初始高度差：约 244mm

### 5.3 终止条件

| 条件 | 规则 |
|------|------|
| 成功 | XY < 2mm 且插入深度 > 40mm |
| 偏离 | XY > 80mm |
| 超时 | episode > 8s |

---

## 6. 训练流程

### 6.1 单次训练迭代时序

```
┌─────────────────────────────────────────────────────┐
│ 训练入口 (train.py)                                 │
│                                                     │
│  创建环境 → 创建 PPO Runner → 获取初始观测          │
│                                                     │
│  ┌─── 每个 iteration ──────────────────────────┐    │
│  │                                              │    │
│  │  ┌── 每个 rollout step ──────────────────┐   │    │
│  │  │                                        │   │    │
│  │  │  actor(policy + rgb) → 动作            │   │    │
│  │  │           ↓                            │   │    │
│  │  │  env.step(action)                      │   │    │
│  │  │    ├─ EMA 平滑动作                     │   │    │
│  │  │    ├─ phase 感知缩放                   │   │    │
│  │  │    ├─ 6-DOF IK → 关节目标              │   │    │
│  │  │    ├─ Isaac Sim 物理步进               │   │    │
│  │  │    ├─ 更新几何状态                     │   │    │
│  │  │    ├─ 生成新 observation               │   │    │
│  │  │    ├─ 判断 done / phase 切换           │   │    │
│  │  │    ├─ 计算 reward                      │   │    │
│  │  │    └─ 如果 done → reset 随机新场景     │   │    │
│  │  │                                        │   │    │
│  │  │  PPO 收集 transition                   │   │    │
│  │  └────────────────────────────────────────┘   │    │
│  │                                              │    │
│  │  计算 returns (GAE)                          │    │
│  │  PPO update (8 epochs × 4 mini-batches)      │    │
│  │  保存 checkpoint（按 save_interval）         │    │
│  │  打印训练指标                                │    │
│  └──────────────────────────────────────────────┘    │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 6.2 PPO 超参数

| 参数 | 值 |
|------|------|
| rollout steps / env | 128 |
| learning rate | 1e-4（adaptive） |
| γ | 0.995 |
| λ (GAE) | 0.95 |
| clip | 0.2 |
| epochs per update | 8 |
| mini-batches | 4 |
| desired KL | 0.008 |
| 默认并行环境数 | 8 |
| 物理 dt | 1/120 s |
| decimation | 8 |

---

## 7. 当前进展与核心挑战

### 7.1 已完成

- 完整的 Isaac Lab 环境搭建
- 两阶段 reward 框架与 phase 切换保护
- asymmetric actor-critic 训练架构
- 外部固定相机 + 灰度 RGB 输入
- 场景/视觉调试工具

### 7.2 当前核心挑战

| 挑战 | 说明 |
|------|------|
| 策略仍倾向快速下压 | 虽然已多次加强 pre-contact 搜索信号，但 50 iteration 后仍未稳定出现横向找孔行为 |
| 视觉是否被 actor 利用尚未验证 | 需要通过 no-vision ablation 做对照实验 |
| 相机 framing 仍在调试 | 需要继续确认 peg/hole 在 RGB 图中的可见性 |
| 灰度 RGB 信息量待验证 | 当前灰度图和原始 RGB 几乎一致，说明场景外观对比不足 |

### 7.3 推荐的下一步方向

1. 继续调整相机位置和视角，确保 peg/hole/block 三者同时可见
2. 做 no-vision baseline 对照实验
3. 考虑增加场景材质/亮度对比
4. 必要时切换到 full RGB 或 RGB+depth

---

## 8. 项目文件结构

```text
clean_isaaclab_local_insert/
├── __init__.py              # Gym 环境注册
├── train.py                 # 训练入口
├── play.py                  # Demo / 评估
├── inspect_scene.py         # 场景与相机调试
├── generate_assets.py       # 资产生成
├── SYSTEM_OVERVIEW.md       # 系统结构总览（内部版）
├── SYSTEM_OVERVIEW_REPORT.md # 系统结构总览（汇报版，本文件）
├── PROGRESS_REPORT.md       # 项目进展总结
├── AGENTS.md                # 项目知识库
├── COMMANDS.md              # 常用指令
├── LOGBOOK.md               # 工作日志
├── README.md                # 仓库首页
├── agents/
│   └── rsl_rl_ppo_cfg.py    # PPO 训练配置
├── assets/                  # USD 资产
└── env/
    ├── cfg.py               # 环境配置中心
    ├── core.py              # 环境主类
    ├── observations.py      # 观测组装
    ├── reset.py             # 场景重置与随机化
    ├── rewards.py           # 奖励函数
    └── success.py           # 成功判定
```

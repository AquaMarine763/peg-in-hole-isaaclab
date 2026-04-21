# Clean Isaac Lab Local Insert — 系统结构与训练流程总览

## 1. 这份文档的目的

这份文档用于系统总结当前 `clean_isaaclab_local_insert` 项目的：

- 训练框架
- 模块划分
- 模块之间如何协同工作
- 各模块输入输出
- 单次训练迭代的时序流程

它和 `AGENTS.md` 的区别是：

- `AGENTS.md` 偏长期知识库与参数事实记录
- 本文档偏“系统工作机制”总结

---

## 2. 当前项目在做什么

当前任务不是大范围全局找孔，而是：

- 机器人从固定 ready pose 出发
- peg 初始保持竖直
- hole 在受限 workspace 内随机
- actor 依赖外部固定相机的视觉输入进行局部找孔 / 接近
- 接触后再做修正与插入

也就是：

> **固定 ready pose + 受限工作区视觉搜索 + 接触后修正插入**

---

## 3. 项目文件与职责

### 3.1 顶层入口文件

#### `__init__.py`
**作用**：向 Gym 注册环境。

**输入**：无运行时输入。  
**输出**：注册环境 ID：

```python
LocalInsert-UR10e-Direct-v0
```

并绑定：

- 环境类：`env.core:LocalInsertEnv`
- 环境配置：`env.cfg:LocalInsertEnvCfg`
- PPO 配置：`agents.rsl_rl_ppo_cfg:LocalInsertPPORunnerCfg`

---

#### `train.py`
**作用**：训练入口。

**输入**：
- `--num_envs`
- `--max_iterations`
- `--resume`

**输出**：
- checkpoint：`logs/local_insert/model_xxx.pt`
- 控制台训练日志

**主要职责**：
1. 创建 env 配置与 PPO 配置
2. 创建 Gym 环境
3. 包装成 `RslRlVecEnvWrapper`
4. 创建 `OnPolicyRunner`
5. 反复执行 rollout + PPO update

---

#### `play.py`
**作用**：加载 checkpoint，运行 demo / policy rollout。

**输入**：
- `--model`
- `--num_envs`
- `--num_episodes`
- `--show_camera_marker`

**输出**：
- rollout GUI
- 每个 episode 的 summary 日志

**注意**：
- `play.py` 默认显示的是 GUI viewport，不一定等于任务相机视角
- 但策略真正吃到的视觉输入仍来自任务相机 `env._camera`

---

#### `inspect_scene.py`
**作用**：不依赖训练效果，专门检查 reset 几何、任务相机图像和场景布局。

**输入**：
- `--save_reset_rgb`
- `--lock_viewport_to_task_camera`
- `--show_camera_marker`
- `--auto_reset_seconds`

**输出**：
- GUI 场景检查
- `debug_outputs/reset_rgb/<timestamp>/` 下的原始 RGB / 灰度 RGB 调试图

**当前保存内容**：
- `reset_000_rgb.png/.pt`
- `reset_000_gray.png/.pt`
- `reset_000.json`

---

#### `generate_assets.py`
**作用**：生成 peg / hole block 的 USD 资产。

**输入**：几何尺寸、材质/显示参数。  
**输出**：
- `assets/peg_cylinder.usda`
- `assets/hole_block.usda`

这是离线资产生成，不参与每步训练循环。

---

### 3.2 环境模块（`env/`）

#### `env/cfg.py`
**作用**：集中定义环境、任务、控制、相机、奖励等全部静态配置。

**主要配置对象**：

- `PegCfg`
- `HoleCfg`
- `TaskCfg`
- `CtrlCfg`
- `ForceSensorCfg`
- `LocalInsertEnvCfg`

**输入**：无运行时输入，属于静态配置。  
**输出**：完整环境配置对象，供 `LocalInsertEnv` 使用。

**这里定义的关键内容**：
- peg / hole 尺寸
- robot 初始 joint pose
- hole workspace
- success / too_far / phase 阈值
- reward scale
- 控制尺度
- camera pose / resolution / modality

---

#### `env/core.py`
**作用**：环境主类，负责把所有模块串起来。

类名：

```python
LocalInsertEnv(DirectRLEnv)
```

##### 关键方法

##### `_setup_scene()`
**输入**：`cfg` 里的 robot / hole / peg / camera 配置。  
**输出**：
- ground
- robot articulation
- hole rigid object
- peg rigid object
- tiled camera
- 可选 camera proxy
- peg fixed joint

##### `_init_tensors()`
**输入**：env 数量、device、配置。  
**输出**：运行时缓存 tensor，如：
- `actions`
- `ee_pos / ee_quat`
- `peg_tip_pos`
- `hole_top_pos`
- `phase_flag`
- `force_history`
- `_cached_rgb`

##### `_pre_physics_step(action)`
**输入**：actor 输出动作。  
**输出**：EMA 平滑后的 `self.actions`。

##### `_apply_action()`
**输入**：
- `self.actions`
- 当前 `phase_flag`
- 当前机器人状态与 Jacobian

**输出**：
- 关节位置目标 `joint_pos_target`

**内部链路**：

```text
动作
-> phase-aware 缩放
-> 末端位移/姿态目标
-> 6-DOF Damped Least-Squares IK
-> joint position target
-> articulation actuator 执行
```

##### `_get_observations()`
**输入**：当前 env 状态。  
**输出**：调用 `observations.py` 返回 observation dict。

##### `_get_dones()`
**输入**：当前 env 状态。  
**输出**：
- `done`
- `time_out`

内部会调用：
- `compute_intermediate_values()`
- `_update_phase_state()`
- `get_curr_successes()`

##### `_update_phase_state()`
**输入**：
- `force_smooth`
- `peg_tip_pos`
- `hole_top_pos`

**输出**：更新 `phase_flag`

当前规则：
- `force_norm > 8N`
- `xy_dist < 20mm`
- 持续 `5` 步

##### `_get_rewards()`
**输出**：调用 `rewards.py::get_rewards()` 返回 reward。

##### `_reset_idx(env_ids)`
**输出**：调用 `reset.py::reset_idx()` 对指定 env 重置。

---

#### `env/reset.py`
**作用**：定义 reset 时的场景初始化与随机化。

**输入**：
- `env_ids`
- 当前 robot articulation 状态
- `TaskCfg`

**输出**：
- 机器人回到固定 ready pose
- hole 在 workspace 内随机化
- 历史量与缓存清零

**当前 reset 流程**：
1. robot joint 重置到固定 ready pose
2. 读取 EE pose
3. 通过固定挂载关系计算 peg root / peg tip
4. 在 workspace 中采样 hole XY
5. 若 `initial_xy_dist > 0.06m` 则重采样
6. hole Z 固定为 `0.14`
7. 清空：
   - `force_history`
   - `prev_xy_dist`
   - `prev_z_gap`
   - `phase_flag`
   - `actions`
   - `_cached_rgb`

---

#### `env/observations.py`
**作用**：把当前环境状态组装成 actor / critic 所需 observation。

##### `compute_intermediate_values(env)`
**输入**：当前仿真状态。  
**输出**：更新一批中间缓存：
- `hole_top_pos`
- `peg_tip_pos`
- `ee_pos / ee_quat`
- `ee_jacobian`
- `joint_pos / joint_vel`

##### `update_force_sensor(env)`
**输入**：原始 joint incoming force。  
**输出**：
- `force_smooth`
- `force_history`

##### `get_observations(env)`
**输出**：

```python
{
    "policy": ...,
    "hole_state": ...,
    "rgb": ...,
}
```

###### `policy`
组成：
- joint_pos
- joint_vel
- ee_pos
- ee_quat
- phase_flag
- masked force_history

###### `hole_state`
- `hole_top_pos[:, :2]`
- 只给 critic 用

###### `rgb`
- 从 `env._camera.data.output["rgb"]` 读取 RGB
- 归一化到 `[0,1]`
- 转灰度单通道
- 缓存到 `_cached_rgb`

**当前 actor / critic 分工**：
- actor：`policy + rgb`
- critic：`policy + hole_state`

---

#### `env/rewards.py`
**作用**：定义 reward 函数与训练日志指标。

**输入**：
- 当前 peg / hole 几何关系
- 历史 `prev_*` 状态
- `phase_flag`
- `force`
- `actions`

**输出**：
- reward tensor
- `env.extras["log"]`

##### 当前 reward 结构

###### Pre-contact
- `precontact_xy_progress_reward`
- `precontact_distance_progress_reward`（很弱，且乘 gate）
- `precontact_z_progress_reward`（乘 gate）
- `precontact_xy_penalty`
- `misaligned_downward_penalty`
- `misaligned_downward_progress_penalty`

###### Post-contact
- `postcontact_xy_progress_reward`
- `postcontact_distance_progress_reward`
- `insertion_reward * soft_gate`
- `force_penalty`

###### Shared
- `action_penalty`
- `success_bonus`

**当前核心意图**：

> 先找孔，再下压；没对准时继续下压，应当是净负收益。

---

#### `env/success.py`
**作用**：定义成功条件。

**输入**：
- `peg_tip_pos`
- `hole_top_pos`
- success 阈值

**输出**：布尔 success。

当前 success 条件：
- `xy_dist < success_xy_tolerance`
- `insertion_depth > hole.height * success_depth_fraction`

---

## 4. 训练配置模块

### `agents/rsl_rl_ppo_cfg.py`
**作用**：定义 PPO、actor、critic 的结构和超参数。

**输入**：无运行时输入，属于静态训练配置。  
**输出**：`LocalInsertPPORunnerCfg`

### 当前关键设计

```python
obs_groups = {
    "actor": ["policy", "rgb"],
    "critic": ["policy", "hole_state"],
}
```

#### actor
- CNN + MLP
- 输入：`policy + rgb`
- 输出：动作分布

#### critic
- MLP
- 输入：`policy + hole_state`
- 输出：value

这是一种 **asymmetric actor-critic + privileged critic** 结构。

---

## 5. 模块之间怎么协同工作

## 5.1 Reset 阶段

```text
train.py / play.py / inspect_scene.py
-> env.reset()
-> core.py::_reset_idx()
-> reset.py::reset_idx()
-> robot 固定 ready pose
-> hole 在 workspace 内随机
-> peg/hole/历史缓存初始化
```

---

## 5.2 Observation 阶段

```text
core.py::_get_observations()
-> observations.py::get_observations()
-> compute_intermediate_values()
-> policy / hole_state / rgb
```

---

## 5.3 Action 执行阶段

```text
actor(policy + rgb)
-> action
-> core.py::_pre_physics_step(action)
-> core.py::_apply_action()
-> IK -> joint target
-> Isaac Sim articulation 执行
```

---

## 5.4 Reward / Done 阶段

```text
core.py::_get_dones()
-> compute_intermediate_values()
-> _update_phase_state()
-> success / too_far / timeout

core.py::_get_rewards()
-> rewards.py::get_rewards()
-> reward + log metrics
```

---

## 5.5 PPO 更新阶段

```text
train.py
-> runner.alg.act(obs)
-> env.step(action)
-> process_env_step(...)
-> compute_returns(...)
-> update()
```

---

## 6. 单次训练迭代时序图

下面这个图描述的是 `train.py` 中 **一次迭代（iteration）** 的完整流程。注意一次 iteration 内部还包含很多个 rollout step。

```text
train.py
│
├─ 1. 创建 env_cfg / agent_cfg
├─ 2. gym.make("LocalInsert-UR10e-Direct-v0")
│    └─ core.py::_setup_scene()
│         ├─ 创建 robot / hole / peg / camera / proxy
│         └─ clone environments
│
├─ 3. wrapper + runner 初始化
├─ 4. obs = env.get_observations()
│
└─ 5. for each iteration:
     │
     ├─ for each rollout step:
     │    │
     │    ├─ actor(policy + rgb) -> action
     │    ├─ env.step(action)
     │    │    │
     │    │    ├─ _pre_physics_step(action)
     │    │    ├─ _apply_action()
     │    │    │    └─ IK -> joint target
     │    │    ├─ Isaac Sim physics step
     │    │    ├─ _get_observations()
     │    │    │    └─ policy / hole_state / rgb
     │    │    ├─ _get_dones()
     │    │    │    ├─ _update_phase_state()
     │    │    │    └─ success / too_far / timeout
     │    │    ├─ _get_rewards()
     │    │    │    └─ rewards.py
     │    │    └─ 若 done -> _reset_idx() -> reset.py
     │    │
     │    ├─ runner.alg.process_env_step(...)
     │    └─ logger 累积 metrics
     │
     ├─ runner.alg.compute_returns(obs)
     ├─ runner.alg.update()
     ├─ 保存 checkpoint（按 save_interval）
     └─ 打印本 iteration 的训练指标
```

---

## 7. 详细训练流程

### 7.0 启动阶段

当你执行 `python train.py --num_envs 8 --max_iterations 50 --headless` 后，系统会依次完成以下准备工作：

#### (a) Isaac Sim 启动

`train.py` 最先启动 Isaac Sim 引擎，加载 GPU / 物理 / 渲染。这一步通常耗时最长，会输出大量初始化日志。

#### (b) 创建配置

从 `env/cfg.py` 和 `agents/rsl_rl_ppo_cfg.py` 读取所有静态参数：场景几何、控制尺度、reward 权重、相机参数、PPO 超参数、actor/critic 网络结构。

#### (c) 创建环境

`gym.make("LocalInsert-UR10e-Direct-v0")` 触发 `LocalInsertEnv.__init__()` 和 `_setup_scene()`。这一步会在 Isaac Sim 里生成 8 份（`num_envs=8`）完全独立的平行环境，每个里面包含一套完整的 robot + peg + hole block + camera。

#### (d) 包装环境

用 `RslRlVecEnvWrapper` 把 Gym 环境包装成 RSL-RL 兼容接口。

#### (e) 创建 PPO Runner

根据 `obs_groups` 配置创建 actor（CNN + MLP）和 critic（MLP），以及 PPO 算法对象和 rollout buffer。

#### (f) 获取初始观测

触发一次环境 reset（包括 hole 随机化），然后返回第一组 obs。到这里训练循环还没开始，但一切准备就绪。

---

### 7.1 训练主循环

假设 `max_iterations=50`，主循环会跑 50 次。每个 iteration 包含两大阶段：rollout 收集 + PPO 更新。

---

### 7.2 Rollout 收集阶段

每个 iteration 做 `num_steps_per_env=128` 步 rollout。8 个环境各走 128 步，总共收集 `8 × 128 = 1024` 个 transition。

#### 每一步 rollout 的内部流程

##### Step 1：actor 输出动作

actor 的 CNN 先处理灰度 RGB 图提取视觉特征，然后把视觉特征和 policy 向量拼在一起过 MLP，输出 6D 动作的均值和方差。训练时从这个分布里随机采样一个动作（带探索噪声）。

##### Step 2：环境执行动作

`env.step(actions)` 内部依次触发：

**(a) `_pre_physics_step(action)`**

把 actor 输出的原始动作做 EMA 平滑：`smoothed = 0.05 × new + 0.95 × old`，让动作不会一步跳太远。

**(b) `_apply_action()`**

按当前 `phase_flag` 选择动作缩放：

- pre-contact：XY 最多 0.5mm/step，Z 最多 0.8mm/step，旋转 mask 为 0
- post-contact：XY 最多 0.8mm/step，Z 最多 1.5mm/step，旋转最多 0.03 rad/step

然后：
1. 缩放后的动作加到当前末端位置，得到末端位移目标
2. 旋转动作转成末端姿态目标
3. 6-DOF DLS IK 把末端目标转成关节角增量
4. 关节角增量加到当前关节角，得到新的关节位置目标
5. 写入 Isaac Sim

**(c) Isaac Sim 物理步进**

Isaac Sim 内部用关节驱动器（stiffness=800, damping=40）把关节往目标角度拉，做碰撞检测，更新所有物体位置/速度。`decimation=8` 意味着一个 action step 包含 8 个物理 sub-step。

**(d) `_get_observations()`**

物理步进完成后重新读取所有状态：
- 调 `compute_intermediate_values()` 更新 peg_tip_pos, hole_top_pos, ee_pos, jacobian, force sensor 等
- 组装新的 obs dict：policy + hole_state + rgb

**(e) `_get_dones()`**

先更新 phase（检查接触条件），然后检查三种终止：
- success：XY < 2mm 且插入深度 > 40mm
- too_far：XY > 80mm
- timeout：episode 超过 8 秒

**(f) `_get_rewards()`**

调 `rewards.py::get_rewards()`，对 8 个环境分别算 reward。reward 计算完全依赖当前步和上一步之间的变化量。

**(g) 如果某个 env done 了**

自动触发 `_reset_idx(env_ids)`：robot 回到固定 ready pose，hole 重新随机，所有历史缓存清零，这个 env 立刻开始新 episode。

##### Step 3：PPO 收集 transition

把 `(obs, action, reward, done, value, log_prob)` 存进 rollout buffer。

##### Step 4：重复 128 次

8 个环境各走 128 步后，rollout 收集阶段结束，buffer 里有 1024 个 transition。

---

### 7.3 PPO 更新阶段

rollout 收集完后，用这些数据更新 actor 和 critic。

#### (a) 计算 returns

用 GAE（Generalized Advantage Estimation）从 buffer 里的 reward 序列倒推出：
- **returns**：每个 transition 从现在到 episode 结束的总回报估计
- **advantages**：每个 transition 比 critic 预期好了多少

```text
advantage = 实际回报 - critic 预估
```

GAE 参数：`gamma=0.995`, `lambda=0.95`。

#### (b) PPO 梯度更新

把 1024 个 transition 分成 4 个 mini-batch，每个 mini-batch 过 8 个 epoch，总共 32 次梯度更新。

每次梯度更新做两件事：

**更新 actor**：
1. 用当前 actor 重新算每个 transition 的 action log probability
2. 算 ratio = 新 log prob / 旧 log prob
3. 用 PPO clip 目标函数（clip_param=0.2）
4. 反向传播更新 actor 权重

**更新 critic**：
1. 用当前 critic 重新算每个 transition 的 value
2. 算 value loss = (predicted_value - target_return)²
3. 反向传播更新 critic 权重

**自适应学习率**：如果实际 KL 散度偏离 `desired_kl=0.008` 太多，自动调整 learning rate。

---

### 7.4 保存与日志

每隔 `save_interval=100` 个 iteration（或训练结束时）保存 checkpoint 到 `logs/local_insert/model_xxx.pt`。

每个 iteration 结束后打印训练指标：

```text
iter 1/50 | rew -0.995 | v_loss 479.10 | xy 0.0546 | zgap 0.0100 | depth 0.0000 | gate 0.00 | phase1 0.00 | force 68.62 | ep_done 1 (succ 0 far 0 to 1)
```

各指标含义：

| 指标 | 含义 |
|------|------|
| rew | 该 iteration 最后一步的平均 reward |
| v_loss | critic 的 value loss |
| xy | 平均 peg-hole XY 距离 |
| zgap | 平均 peg tip 到 hole top 的高度差 |
| depth | 平均插入深度 |
| gate | soft alignment gate 均值 |
| phase1 | 处于 post-contact 的比例 |
| force | 平均接触力 |
| ep_done | 该轮结束的 episode 数 |
| succ/far/to | 其中多少个是成功/偏离/超时 |

---

### 7.5 训练结束

所有 iteration 跑完后，保存最终模型，关闭环境和 Isaac Sim。

---

## 8. 每个模块的输入输出总结表

| 模块 | 输入 | 输出 |
|---|---|---|
| `__init__.py` | 无 | Gym 环境注册 |
| `train.py` | CLI 参数 + env cfg + PPO cfg | 训练循环、checkpoint、日志 |
| `play.py` | 模型路径 + env cfg | demo rollout / episode summary |
| `inspect_scene.py` | env cfg + 调试参数 | GUI 检查 + 保存 RGB 图 |
| `env/cfg.py` | 静态配置 | 场景/任务/控制/相机参数 |
| `env/core.py` | cfg + action + 当前仿真状态 | 场景创建、动作执行、phase、done、reward 接线 |
| `env/reset.py` | env_ids + cfg + robot state | 随机 hole、重置缓存 |
| `env/observations.py` | 当前 env 状态 + camera raw | `policy / hole_state / rgb` |
| `env/rewards.py` | 几何关系 + 历史状态 + force + actions | reward + metrics |
| `env/success.py` | peg/hole 几何 | success bool |
| `agents/rsl_rl_ppo_cfg.py` | 静态训练设计 | actor/critic/PPO 配置 |

---

## 9. 当前系统最重要的设计重点

当前项目里最关键的设计重点不是“能不能跑起来”，而是：

1. **任务分布**：hole 在受限 workspace 内随机
2. **actor 是否真的用视觉找孔**
3. **reward 是否真的逼策略先找孔再下压**
4. **phase 切换是否堵住了“乱撞 block”捷径**
5. **critic 的 privileged hole_xy 是否帮助了 value estimation**

所以当前各模块最关键的耦合关系是：

- `reset.py` 决定任务难度分布
- `observations.py` 决定 actor / critic 各自知道什么
- `core.py` 决定动作如何落实成实际运动
- `rewards.py` 决定策略最容易学成什么行为

---

## 10. 当前阶段最重要的问题

虽然系统架构已经比较稳定，但当前训练还没真正解决的问题仍然是：

> **策略仍然容易先下压，而不是稳定地先做横向找孔。**

这意味着接下来最重要的工作仍然集中在：

- 相机 framing 是否足够好
- RGB 图像是否包含足够 peg-hole 结构
- pre-contact reward 是否真的把错误下压变成净负收益
- 视觉信息是否真的被 actor 利用

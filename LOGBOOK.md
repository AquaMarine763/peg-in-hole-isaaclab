# 工作日志

### 2026-04-06

- 新建 clean Isaac Lab 子项目，目标是按教程式任务定义重建局部插入环境。
- 先搭好最小项目结构、资产生成脚本、env 模块拆分和训练 / demo 入口。
- 烟雾测试通过：`train.py --num_envs 1 --max_iterations 1` 可启动并完成 1 iteration。
- 修复 FixedJoint 冲突：reset 中不再直接写 peg root pose/velocity，消除了 non-root articulation link 相关警告。
- 定位到 peg 可视化问题根因：peg 之前沿世界 -Y 水平伸出而不是朝下。改为通过固定关节旋转/偏移让 peg 朝下从末端露出。
- 按教程思路将策略简化为 3D 小位移动作，只学局部平移，姿态固定保持。
- 进一步对齐教程式场景：hole block 改为底面贴地（top_z=0.15），增大 block 尺寸并固定在地面；robot reset 姿态改为 peg 朝下，reset 后 hole 与 peg 在 XY 上保持局部接近。
- 再次修正 peg 挂载：现在 peg 与 `wrist_3_link` 最后一节真正同轴（实测 `LINK_DIR == PEG_DIR`），并且 1 env / 1 iteration 训练继续可启动。
- 调整 robot 默认姿态与 peg 安装偏移，避免 peg 看起来悬空；同时把 hole 默认工作区前移，避免 block 初始位置贴近机械臂基座。
- 将 clean 项目的局部 handoff 范围收紧到更符合"粗定位已完成"的设定：XY 偏差改为 ±8mm，Z 高度改为孔口上方 6-10mm。
- 加入 tutorial-style 的 too-far 终止：peg tip 与 hole 的 XY 距离超过 40mm 直接 reset，用来抑制策略早期跑飞；1 env / 1 iteration 训练验证仍通过。
- reward 改为更接近教程的"进步奖励"版本：距离进步 + 对齐条件下的插入进步 + 成功奖励，并保留小幅力/动作惩罚；1 env / 1 iteration 训练验证仍通过。
- 相机视角改为 hole 工作区侧上方定点观察（更接近局部插入任务需要的 tight local view），优先减少机械臂遮挡并让 hole 更靠近画面中心。
- 新增 `inspect_scene.py`，用于静态检查 peg / hole / 相机 / reset 几何，不必先依赖训练后的策略效果。
- 升级为显式两阶段单策略：加入 phase flag、contact gate、阶段化 reward 和更保守的动作尺度，并把训练/demo 输出扩展到 phase、force、done reason 等指标。
- 烟雾测试通过，但新输出显示 phase1 比例很高，说明当前 reset 几何仍让 peg 很快进入接触/近接触区；后续优先继续调 reset 高度与接触判定细节。
- 短训练（16 envs × 20 iterations）已跑通。现象：XY 基本稳定在 10-13mm，too-far 很少，但插入深度仍接近 0，说明策略已学会局部靠近但还没学会真正插入。
- 将 reset 高度改为 peg tip 高于孔口 10-15mm，更贴近"视觉粗对准后，RL 主要负责最后插入"的任务定义。
- 两阶段版本继续细化：加入腕力零偏标定、接触窗口门控和 deterministic demo 推理，避免把基线载荷或采样噪声误判为接触/策略能力。
- 修正训练日志：`v_loss` 现在读取 rsl_rl 返回的正确字段 `value`，并把 `succ/far/to` 改成 `ep_done (...)`，明确表示它们是单个 iteration 内累计的 episode 结束次数。
- 按 Oracle 建议继续简化控制链：从 torque-level OSC 改为"末端局部位移 -> 差分IK -> 关节位置目标"，让 demo 运动更接近教程风格；1 env / 1 iteration 烟雾测试通过。
- 修正 reset 几何与 too-far：将 hole block 高度调整到 155mm，使 peg tip 初始高于孔口约 18mm；同时把 too-far 阈值放宽到 8cm，并通过脚本验证新的几何关系与训练可启动。
- 继续把 reset 高度提高到更接近粗定位后的 handoff：hole block 高度调整到 140mm，实测 peg tip 初始高于孔口约 33.4mm；保持 too-far 阈值 8cm。
- 进一步把视觉更新改为每个 action step 一次，并把 clean 项目默认并行环境数降到 8；复测中发现 texture cache 清理后仍会重新出现 `.ovtex` 读取错误，但训练可继续运行，当前主要瓶颈仍是策略很快偏离 hole 导致 too-far 结束。
- 继续优化 pre-contact：动作缩小为 pre-contact XY 1.0mm / Z 2.5mm、post-contact XY 0.8mm / Z 1.5mm；reward 增加"对准后向下压"项与 pre-contact XY 软惩罚，并在训练/demo 输出中新增 `zgap` 与 `dgate` 指标。
- 为避免 pre-contact 被几何真值过度塑形，移除了强下压/XY 真值项，改成更弱的 pre-contact 距离进步奖励，并恢复一个弱的 XY 进步奖励，保持视觉主导但仍给最小训练扶手；1 env / 1 iteration 烟雾测试通过。
- 继续按"接触前视觉主导、接触后力主导"收敛：pre-contact force 继续屏蔽，阶段切换改为稳定接触事件（力阈值+连续步数+向下运动），pre-contact 只保留弱的 XY/距离进步奖励；1 env / 1 iteration 烟雾测试通过。

### 2026-04-07

- hole 工作区中心前移：robot 初始关节角改为 `[0.0, -1.1, 1.5, -1.97, -1.5708, 0.0]`（更伸展，peg 仍朝下，joint2+3+4 ≈ -π/2）；hole 默认 X 从 0.30 改到 0.50；相机 X 从 0.48 同步前移到 0.68。1 env / 1 iteration 烟雾测试通过：xy=18.6mm, zgap=14.2mm, 无 too-far。

### 2026-04-08

- 针对 demo 中 4/5 episode too_far 的问题做三处调整：(1) precontact XY 动作从 1.0mm 缩小到 0.5mm；(2) 阶段切换条件去掉 downward_cmd 和 downward_motion 要求，只保留力阈值+持续步数；(3) 新增 precontact XY 距离软惩罚 (scale=15)。烟雾测试通过。

### 2026-04-09

- 大规模改进：(1) 动作空间从 3D 扩展到 6D，pre-contact 旋转 mask 为 0 并由 6-DOF IK 维持 nominal 朝向，post-contact 放开旋转自由度 (±0.03 rad/step) 用于插入修正；(2) 相机从 (0.68,-0.12,0.38) 拉近到 (0.61,0.0,0.28)，hole 从 ~5px 提升到 ~8-9px；(3) reset 高度改为动态定位 (30-40mm)；(4) hard alignment gate (xy<3mm) 替换为 soft gate `exp(-xy²/(2σ²))` σ=5mm；(5) post-contact 新增 XY 进步奖励 (scale=80)；(6) force penalty 从 0.01 加大到 0.05；(7) 修复 multi-env 下 reset.py 的坐标系混用 bug (peg_tip_w 为世界坐标但与 local hole_top_pos 做差)。
- 修正 reset 几何回归：在 `write_joint_state_to_sim()` 后显式调用 `robot.reset(env_ids)` 再读取 EE 姿态，避免沿用上个 episode 的 stale/titled EE pose；验证当前初始几何恢复为局部 handoff（xy≈9.6mm, zgap≈35.4mm），且 peg 初始轴线约为 `[0, 0, -1]`，与 hole 同轴竖直。
- 改进可视化脚本的 reset 预览：`inspect_scene.py` 在每次 reset 后先执行若干次 render/update warm-up 再打印与展示；`play.py` 在首次 reset 和每次 episode 自动 reset 后先展示 reset preview，再进入动作 rollout，避免窗口第一眼看到的是 step 后或 stale 的帧。
- 为拆分独立仓库做发布准备：将 README 改为面向独立 Isaac Lab 仓库的标题与首页介绍，准备创建公开仓库 `peg-in-hole-isaaclab`。
- 任务定义升级为“受限工作区视觉搜索 + 插入”：robot ready pose 固定、peg 初始保持竖直，hole 改为在固定 workspace `(0.92, 0.17) ± (0.04, 0.04)` 内独立随机，并增加 `initial_xy_dist <= 0.06m` 约束，打破 peg/hole 强绑定但仍保持局部任务。
- reward 重构为更偏视觉搜索的 pre-contact 设计：提高 XY progress 权重，新增 gated Z progress（σ=15mm），并加入 misaligned downward penalty，鼓励“先找孔、再下压”而不是盲目撞击 block；同时把 AGENTS.md 扩写为完整记录工作区、参数、reward 和工作流程的知识库。
- 将当前关键几何尺寸系统写入 `AGENTS.md`：包括 peg、hole / block、安装偏移、fixed ready pose 下的 peg tip 位置、hole 顶面高度、相机与 workspace 参数，后续这些尺寸有变动时应同步更新。
- 将 fixed ready pose 抬高到接近 25 cm 初始高度：robot 初始关节角改为 `[0.0, -1.55, 1.95, -1.97, -1.5708, 0.0]`，实测 peg tip 约为 `(0.659, 0.174, 0.384)`，初始 `z_gap ≈ 0.244 m`；同时把 hole workspace 中心同步改到 `(0.66, 0.17)` 以匹配新的 ready pose。
- 为相机调试补充 inspect 工具：`inspect_scene.py` 新增 `--save_reset_depth`（默认保存到 `debug_outputs/reset_depth/<时间戳>/`，包含 `.png/.pt/.json`）和 `--lock_viewport_to_task_camera`（把 GUI 视角切到 `/World/envs/env_0/Camera`），方便直接检查任务相机看到的内容以及 peg 对 hole 的遮挡情况，同时避免主目录堆满调试文件。
- 新增任务相机可见标记：通过 `--show_camera_marker` 在 `inspect_scene.py` 和 `play.py` 中显示任务相机的位置与朝向标记，便于在普通 viewport 下确认相机实体位置和观察方向。
- 放大任务相机标记并增加前向箭头，方便在 demo 中更清楚地辨认相机的实际位置与观察方向；当前更推荐用 `play.py --show_camera_marker` 来确认 demo 真正使用的任务相机 pose。
- 将任务相机标记改为 IsaacLab 官方 `VisualizationMarkers / FRAME_MARKER_CFG`（外加球形原点 marker），避免之前手写 USD marker 在 demo 中不稳定/不可见的问题；同时将 reset depth PNG 改为默认按 `0.15m ~ 0.50m` 的工作区深度范围做高对比可视化，保留 `.pt` 原始 tensor 供策略/数值分析使用。
- 放弃抽象 marker，改成更直观的实体相机代理：用一个可见机身盒子加镜头圆柱表示任务相机的位置和朝向；同时把相机从 `(0.61, 0.0, 0.28)` 拉远到 `(0.58, 0.05, 0.40)`，并将深度图分辨率从 `48×48` 提高到 `64×64`，以减少近距离遮挡风险并补偿像素占比。
- 将深度图分辨率进一步从 `64×64` 提高到 `84×84`，在相机拉远后的前提下进一步提升孔口与 peg 的可见像素占比，方便视觉搜索阶段学习更细的局部几何线索。
- 撤掉 `play.py / inspect_scene.py` 里不稳定的临时相机代理注入逻辑，改为在 `env/core.py` 的 scene setup 中生成稳定的纯视觉相机代理（机身盒子 + 镜头圆柱）。这样 demo / inspect 共享同一个环境级相机代理，不参与物理和碰撞，也避免脚本级临时 prim 注入带来的 GUI 回归。
- 相机继续往更远、更高的候选位置调整：改到 `(0.05, -0.75, 0.85)`，并按 Isaac 相机 `+X` 为 forward 的约定 look-at `(0.66, 0.17, 0.35)`，得到新的四元数 `(0.861234, -0.099822, 0.185958, 0.462311)`；目的是让 `z_gap≈0.50m` 条件下更高的 peg 起点和整个 hole workspace 都更容易进入画面。
- 修正相机实体代理的局部镜头朝向：原来代理镜头沿本地 `-Z` 摆放，导致在 inspect/demo 中看起来和真实任务相机朝向不一致；现在改成沿本地 `+X` forward 对齐 Isaac 相机约定，使代理朝向与真实任务相机输出一致。
- 修复 `train.py` 的 TensorBoard 写入链路：之前只初始化 writer 并缓存 env step 指标，但没有调用 `runner.logger.log(...)`，导致 event 文件只有 88 字节空壳、TensorBoard 无 dashboard。现在已补上 logger.log、writer.flush() 和 writer.close()，训练后应该能看到 loss/reward/xy 等标量曲线。
- 新增 `plot_training_stats.py`：支持手动指定 TensorBoard event 文件，导出 8 张单图（value/surrogate/entropy loss、reward、xy、zgap、force、done counts），默认输出到 `debug_outputs/training_plots/<时间戳>/`，方便做实验对比和汇报。
- 继续把 reward 往“先找孔再下压”方向推进：pre-contact 的 3D distance progress 改为也受 `pre_align_gate` 调制，`precontact_distance_progress_scale` 从 `5.0` 降到 `1.0`，`precontact_xy_progress_scale` 从 `120` 提高到 `160`，`misaligned_downward_penalty_scale` 从 `2.0` 提高到 `6.0`，并把 pre-contact Z 动作阈值从 `2.0mm` 收紧到 `1.2mm`，减少策略早期直接向下冲的诱因。
- 训练架构继续改进为 asymmetric actor-critic + privileged critic：保持 actor 输入不变（`policy + depth`），只给 critic 额外加入 `hole_xy` 特权真值，用来改善 pre-contact 视觉搜索阶段的 value estimation；相比让 critic 也看 depth，这个改法更轻量，也更符合当前 GPU/显存约束。
- 继续强化“先找孔再下压”：将 pre-contact Z 动作阈值从 `1.2mm` 进一步收紧到 `0.8mm`，同时把 phase 切换改为更保守的 `force_norm > 8N` 且持续 `5` 步，减少“先碰到 block 就切 phase”的捷径；pre-contact 的 `distance_progress_scale` 再降到 `0.25`，`z_progress_scale` 再降到 `25`，`misaligned_downward_penalty_scale` 再升到 `10`，进一步压制早期下冲局部最优。
- 继续把“先找孔再下压”做成更硬约束：把 misaligned downward 判定阈值从 `20mm` 收紧到 `15mm`，并新增 `misaligned_downward_progress_penalty`，当 `xy_dist > 15mm` 且 peg 真实发生 downward progress 时直接处罚实际向下位移，让“没对准就继续往下压”在 reward 上更明确地变成净负收益。
- 当前 phase 切换再加一层 XY 保护：只有 `force_norm > 8N` 且 `xy_dist < 20mm` 并持续 `5` 步，才允许进入 post-contact，避免策略靠“偏着撞到 block”直接切到接触后阶段。
- 在 `rgb` 方向上做最小迁移：把相机输出从 depth 切到 RGB，并在 observation 中直接把 RGB 转成灰度单通道 `rgb` 输入给 actor；critic 继续保持 `policy + hole_xy`。同时把 `inspect_scene.py` 的调试保存链路从 `reset_depth` 改成 `reset_rgb`，用于保存灰度 RGB 观测而不是 depth 图。
- 进一步增强 RGB 调试链路：`inspect_scene.py --save_reset_rgb` 现在会同时保存原始 RGB 和灰度 RGB 两套输出，便于区分“灰度化把信息洗掉了”还是“相机本身就没拍到有用结构”。
- 新增 `SYSTEM_OVERVIEW.md`，把当前项目的训练框架、模块职责、输入输出关系以及“单次训练迭代”的完整时序整理成一份独立总结文档，便于后续快速理解整个系统是怎么工作的。
- 重写 `PROGRESS_REPORT.md`，把当前训练框架、控制链、视觉系统、两阶段 reward、工作区设置、actor/critic 设计、当前问题与下一步建议系统整理成单文件总结，作为当前项目阶段的完整进展展示。
- 新增 `SYSTEM_OVERVIEW_REPORT.md`：基于 `SYSTEM_OVERVIEW.md` 整理的对外展示/汇报版本，用表格和结构化布局替代了代码味更重的内部版，可用于项目汇报或对外交流。
- 按方案 B 大幅改变 pre-contact reward 结构：`precontact_distance_progress_scale` 从 `0.25` 降为 `0`（彻底去掉 3D 距离进步奖励），`precontact_z_progress_scale` 从 `25` 降为 `5`（极弱且完全受 pre_align_gate 控制），目的是彻底消除"不靠视觉也能靠下压拿 reward"的局部最优，逼策略先学视觉 XY 搜索。
- 纠正先前未落实到位的环境设置：这次真正把相机分辨率改为 `160×160`，并通过直接抬高 robot articulation root `z=0.256m` 的方式，把 peg tip 到 hole 顶面的初始高度差从 `0.244m` 提升到约 `0.50m`。1-iteration 烟雾测试已通过，当前更高起始高度下 rollout 的平均 `zgap` 明显变大，但 too_far 也随之增多，需要后续继续观察视觉搜索行为是否改善。
- 将 `too_far_xy_threshold` 从 `0.08m` 放宽到 `0.16m`，让 actor 在还没学会视觉搜索时有更长的空中探索轨迹，避免 episode 因为过早出界而很快结束。
- 正式落地“最小找孔版本”：actor observation 简化为 `ee_pos + ee_quat + grayscale rgb`，critic observation 简化为 `ee_pos + ee_quat + privileged hole_xy`；动作空间从 6D 改为 3D 末端平移 `[dx, dy, dz]`，姿态由控制链维持 nominal 朝向，不再让 actor 直接输出旋转。
- 在 `wrist-camera` 分支开始腕部相机版本：相机从固定外部视角切到挂载在 `wrist_3_link` 附近的 eye-in-hand 视角，偏移设为 `(0.0, -0.06, -0.02)`、四元数 `(0.707107, 0.0, -0.707107, 0.0)`，其余 observation / action / reward 结构暂时保持当前最小找孔版本不变。
- 同步更新 `AGENTS.md` 到当前 wrist-camera Stage 1 的真实代码状态：workspace 半宽从旧的 `0.04` 收缩到 `0.015`，范围更新为 `x ∈ [0.645, 0.675]`、`y ∈ [0.155, 0.185]`，初始 `max_initial_xy_dist` 也更新为 `0.03m`，避免文档继续停留在外部相机时代的旧参数。
- 继续调整 wrist-camera 安装位姿，按方案 B 将相机相对 `wrist_3_link` 的偏移从 `(0.0, -0.06, -0.02)` 改到 `(0.03, -0.10, 0.02)`，保持四元数不变，目的是让相机更靠外、更高，不低于 wrist 主体底部，并减少对 peg/hole 的局部遮挡。
- 进一步微调 wrist-camera 的高度，使相机最下端尽量与 wrist 主体平齐：将相机相对 `wrist_3_link` 的 z 偏移从 `0.020` 调到 `0.022`，这样镜头前圈的最低点更接近 wrist 平面，减少真实硬件安装时干涉插入路径的风险。
- 按你的要求直接把 wrist-camera 再抬高到明显可见的量级：将相机相对 `wrist_3_link` 的 z 偏移从 `0.022` 提高到 `0.06`，不再做毫米级微调，确保在 inspect 中能明显看出它高于 peg 末端，而不是几乎与 peg 末端平齐。
- 经过再次验证局部坐标与世界坐标关系后，发现前一次把 `z` 从 `0.022` 提到 `0.06` 实际是在当前 wrist 局部坐标系下把相机**往下**压低了，而不是抬高；因此修正为 `(0.03, -0.10, -0.02)`，让相机在世界坐标下真正升高。
- 再往更高、更外的方向进一步试探：将 wrist-camera 偏移从 `(0.03, -0.10, -0.02)` 调到 `(0.08, -0.10, -0.06)`，在当前局部轴约定下同时实现“更外”和“更高”，以进一步减少镜头对 peg/hole 区域的遮挡。
- 基于新一轮 reset RGB 检查（孔可见但仍偏左、视角略低且前景干扰偏多），继续做一版小幅 recenter：将 wrist-camera 偏移从 `(0.08, -0.10, -0.06)` 调到 `(0.08, -0.08, -0.08)`，保持“更外”的同时再略微抬高，并沿横向回收一点，尝试把 hole workspace 往画面中心拉回。
- 基于最新 reset RGB 观察到“只看到 hole、看不到 peg tip”，按候选 2 将 wrist-camera 偏移从 `(0.08, -0.08, -0.08)` 调到 `(0.05, -0.04, -0.06)`，把视角重新收回到更典型的 eye-in-hand 近场交互区，优先让 peg tip 与 hole 同时进入画面。
- 继续按“比当前更高一点、更远一点，并带轻微倾斜”的方向做折中：将 wrist-camera 从 `(0.05, -0.04, -0.06)` / `(0.707107, 0.0, -0.707107, 0.0)` 调到 `(0.06, -0.05, -0.075)` / `(0.7044, 0.0617, -0.7044, 0.0617)`，尝试在不过度贴近末端的前提下，把 peg tip 和 hole 更稳定地同时纳入视野，并让镜头稍微朝 workspace center 内倾。
- 继续沿“更高、更外、并更明显下倾”的方向再推进一档：将 wrist-camera 从 `(0.06, -0.05, -0.075)` / `(0.7044, 0.0617, -0.7044, 0.0617)` 调到 `(0.08, -0.07, -0.09)` / `(0.6964, 0.1228, -0.6964, 0.1228)`，尝试进一步摆脱贴近末端的局部特写视角，并更明确地从上外侧斜看向 workspace center。
- 进一步加大相机抬高、外移与下倾幅度：将 wrist-camera 从 `(0.08, -0.07, -0.09)` / `(0.6964, 0.1228, -0.6964, 0.1228)` 调到 `(0.11, -0.09, -0.12)` / `(0.6830, 0.1830, -0.6830, 0.1830)`，目标是更彻底摆脱贴脸视角，并尝试从更高、更外、且更明显的俯倾角度同时看到 peg tip 与 hole。
- 修正 wrist-camera 旋转思路：前几版四元数主要改变的是绕镜头前向轴的滚转，不能真正把镜头朝 peg 方向偏；这次保持相机位置 `(0.11, -0.09, -0.12)`，将四元数改为 `(0.5092, -0.1544, -0.8415, 0.0934)`，让相机前向实际指向 peg tip 附近。同时将 peg 高度从 `80mm` 加长到 `100mm` 并重新生成 peg 资产，减少末端遮挡对 peg tip 可见性的影响。
- 根据 reset RGB 观察，上一版朝 peg 偏得略过头，hole 在画面中偏右；这次保持相机位置 `(0.11, -0.09, -0.12)` 和 100mm peg 不变，只把四元数从 `(0.5092, -0.1544, -0.8415, 0.0934)` 小幅回调到 `(0.5518, -0.0882, -0.8215, 0.1129)`，目标是在保留 peg tip 可见性的同时把 hole 往画面中心拉回。
- 为解决“下压过快、只对准 XY 就 success”的训练捷径，重新打开 gated pre-contact Z progress、post-contact XY / insertion reward 与 force penalty；success 现在要求 `xy < 5mm` 且插入深度达到孔深 80%，同时把 pre/post contact Z 动作阈值收紧到 0.6mm / 0.5mm，让策略必须学习对准后慢慢下压。
- 按“高初始高度、下降中继续对准”的方向把 robot root Z 从 `0.006m` 提高到 `0.286m`，使 reset 初始 z-gap 从实测约 `0.230m` 提升到预期约 `0.510m`（两倍以上）；同时将 episode 时长从 `8s` 扩展到 `24s`，给策略留出更长的视觉搜索与慢下压轨迹。
- 新增显式快速下压惩罚：pre-contact 真实单步下压进度超过 `2mm` 时惩罚超出部分，post-contact 单步插入进度超过 `1mm` 时更强惩罚超出部分；这样保留下降/插入梯度，但让“最大速度下冲”不再是最优策略。
- 将 actor observation 改成更视觉主导的 Version B：actor 只接收 `ee_z + grayscale rgb`，不再接收 `ee_x / ee_y / ee_quat`；critic 仍保留 `ee_pos + ee_quat + privileged hole_xy`，用于稳定 value estimation。
- 强化训练中断保存链路：不再只被动依赖 `except KeyboardInterrupt`，而是对 `SIGINT/SIGTERM/SIGBREAK` 先发起 graceful stop 请求，在当前 iteration 结束后统一保存 `model_{iteration}.pt` 与 `latest.pt`，提高 PowerShell / Windows 下 Ctrl+C 中断保存的稳定性。
- 将 pre-contact 改成更保守的控制版本，缓解高起点下随机/早期策略迅速横向跑偏和下压刹不住的问题：XY 动作阈值 `0.5mm -> 0.25mm`、Z 动作阈值 `0.6mm -> 0.35mm`、快速下压阈值 `2.0mm -> 0.8mm`，并把 gated Z progress 权重 `30 -> 15`。
### 2026-04-30

- Reviewed the current clean Isaac Lab local insert code path and summarized the environment, observation, control, reward, reset, and PPO training flow; no behavior code changed.
- 扩展训练诊断日志与绘图：训练中额外记录 raw actor action 与 executed action 的 Z/XY 偏置、下压均值和 Z 饱和比例；`plot_training_stats.py` 现在导出 pre-align gate、reward/penalty 组成、xy-vs-zgap 与 action 诊断曲线，方便训练后判断是否“没对准就持续下压”。

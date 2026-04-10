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

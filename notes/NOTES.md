# Notes

## 现实里能用末端笛卡尔坐标操控，不等于仿真里应该直接写末端 pose。
更合理的对应是：
- 笛卡尔目标作为上层接口
- IK / 关节状态作为底层实现

---

## 灰度RGB比普通RGB的优点：
4. 更不容易过拟合到颜色
full RGB 的一个风险是：
- 策略学到的是某种颜色分布
- 或某种材质/光照特征
- 而不是稳定的几何结构
这对 sim-to-real 更危险。
灰度 RGB 会逼网络更关注：
- 轮廓
- 明暗结构
- 形状边缘
这些通常比绝对颜色更稳。
---
5. 如果灰度都能明显提升，就说明颜色未必必要
这是实验设计上很重要的一点：
- 如果灰度 RGB 已经比 depth 更好
- 那你就知道：  
  真正起作用的是外观轮廓信息
- 这时你未必需要更复杂的 full RGB
反过来：
- 如果灰度没提升
- 但你觉得颜色可能有帮助
- 再去试 full RGB

---

## actor和critic的职责

在你这个项目里，actor 和 critic 是 PPO 里的两个网络，它们各自负责不同的事：
Actor
职责
决定"下一步怎么动"。
它是策略本身。
具体做什么
每一步，actor 看到当前观测后，输出一个动作分布：
输入：policy 向量 + 灰度 RGB 图
输出：6 维动作的均值和方差
然后从这个分布里采样一个具体动作，发给环境去执行。
在你这个任务里
actor 要学会的是：
- 从 RGB 图里看出 hole 在哪
- 从 proprio 里知道自己在哪
- 综合判断：应该往左还是往右、往前还是往后、要不要往下
训练时
actor 被 PPO 更新的方向是：
> 让"好动作"的概率变大，让"坏动作"的概率变小。
怎么知道好坏？靠 critic 给的价值估计。
---
Critic
职责
评估"当前状态值多少钱"。
它不出动作，只出一个数字：value。
具体做什么
每一步，critic 看到当前观测后，输出一个标量：
输入：policy 向量 + hole XY 真值
输出：一个数字（预估从现在开始到 episode 结束，总共还能拿多少回报）
在你这个任务里
critic 要学会的是：
- 当前 peg 离 hole 近不近
- 当前 phase 是 pre-contact 还是 post-contact
- 当前状态未来大概能拿多少总 reward
训练时
critic 的 value 被用来计算 advantage：
advantage = 实际拿到的回报 - critic 预估的回报
- advantage > 0：说明这步比预期好，actor 应该更多地做这个动作
- advantage < 0：说明这步比预期差，actor 应该少做这个动作
所以：
> critic 不直接控制机械臂，但它间接决定了 actor 往哪个方向更新。
---
## 两者的关系
你可以这样理解：
### Actor = 运动员
- 负责上场比赛
- 直接做动作
- 自己不知道"这步到底好不好"
### Critic = 教练
- 不上场
- 但能看到更多信息（你这里是 hole 真值位置）
- 负责告诉运动员：
  - "刚才那步做得好"
  - "刚才那步做得差"
然后运动员根据教练的反馈来调整自己的策略。
---
为什么这个项目里它们看的东西不一样
Actor 看
policy + 灰度 RGB
因为部署到真机上时，actor 只能用真实传感器数据。
它必须自己从图像里推断 hole 在哪。
Critic 看
policy + hole XY 真值
因为 critic 只在训练时存在，不上真机。
给它更多信息（hole 的真实位置），能让它更准确地评估状态价值，
从而给 actor 更好的学习信号。
这就是 asymmetric actor-critic 的核心意义：
> actor 用部署时能拿到的信息学决策，
> critic 用额外的真值信息帮助训练更高效。
---
在训练循环里它们怎么配合
1. actor 看 obs，输出动作
2. 环境执行动作，返回 reward
3. critic 看 obs，输出 value
4. 用 reward + value 计算 advantage
5. 用 advantage 更新 actor（让好动作概率更大）
6. 用 reward 更新 critic（让 value 预估更准）
7. 重复
所以：
- actor 被 advantage 更新
- critic 被实际 reward 更新
- 两者互相依赖：
  - critic 越准 → actor 学得越快
  - actor 越好 → critic 更容易学准
---
一句话总结
Actor
"我该怎么动" — 直接出动作，部署时就是它在控制机械臂。
Critic
"当前状态值多少钱" — 不出动作，只帮助 actor 在训练时学得更快更准。

---

## proprio 和 obs 的定义和关系

proprio
proprio 是 proprioception 的缩写，中文叫"本体感觉"。
在这个项目里，它就是：
> 机器人自己知道的关于自己身体状态的信息。
不需要外部传感器，机器人自己就能知道的东西：
分量	维度
joint_pos	6
joint_vel	6
ee_pos	3
ee_quat	4
你可以理解成：
> 闭着眼睛也能知道的信息。
比如你闭着眼，你也知道：
- 自己的手臂弯了多少
- 手在大概什么位置
- 手腕朝哪个方向
这就是 proprio。
它不包括：
- hole 在哪（需要视觉）
- 有没有碰到东西（需要力传感器）
- 当前画面长什么样（需要相机）
---
obs
obs 是 observation 的缩写，就是"观测"。
在你这个项目里，obs 是：
> 当前这一步，actor / critic 能看到的全部信息。
它是一个字典：
obs = {
    "policy": ...,
    "hole_state": ...,
    "rgb": ...,
}
每个 key 是什么
policy
就是上面说的 proprio + 额外任务信息，具体组成是：
joint_pos (6)
+ joint_vel (6)
+ ee_pos (3)
+ ee_quat (4)
+ phase_flag (1)
+ force_history (24)
= 44 维向量
注意它比纯 proprio 多了两个东西：
- phase_flag：当前是 pre-contact 还是 post-contact
- force_history：力传感器历史（pre-contact 阶段被 mask 为 0）
hole_state
- hole_top_pos[:, :2]
- 也就是 hole 在平面上的真实 XY 坐标
- 只给 critic 看，actor 看不到
rgb
- 外部固定相机拍出来的灰度 RGB 图
- 形状：[num_envs, 1, 128, 128]
- 只给 actor 看，critic 不看
---
它们之间的关系
proprio ⊂ policy ⊂ obs
也就是：
- proprio 是"机器人自身状态"
- policy 是"proprio + phase + force"
- obs 是"policy + rgb + hole_state"
而 actor 和 critic 从 obs 里各取自己需要的部分：
角色	从 obs 里取什么
actor	policy + rgb
critic	policy + hole_state
---
一句话总结
proprio
机器人关于自己身体状态的感知：关节角、关节速度、末端位置和朝向。
obs
当前这一步所有可用信息的集合：proprio + phase + force + 视觉 + 特权真值。

---

## PPO算法

PPO 是什么
PPO = Proximal Policy Optimization，近端策略优化。
它是当前强化学习里最常用的算法之一，你这个项目用的就是它。
---
PPO 要解决的核心问题
强化学习的目标是：
> 让策略（actor）学会在每个状态下选择能拿到最多总回报的动作。
最朴素的做法是：
1. 试一下当前策略
2. 看哪些动作拿到了好回报
3. 让好动作的概率变大
但这里有一个关键困难：
> 每次更新策略时，不能更新太多。
因为如果一次改太大：
- 策略可能突然变得很差
- 之前收集的数据就不再适用了
- 训练会变得不稳定
PPO 就是专门解决这个问题的。
---
PPO 的核心思想
一句话版本
> 每次更新策略时，限制更新幅度，不要离旧策略太远。
更具体一点
PPO 的做法是：
1. 用当前策略跑一批数据（rollout）
2. 算出每个动作比预期好还是差（advantage）
3. 更新策略，但夹住更新幅度，不让它跳太远
---
PPO 的三个核心组件
1. Advantage：这步到底好不好
advantage = 实际拿到的回报 - critic 预估的回报
- advantage > 0：这步比预期好，应该更多地做
- advantage < 0：这步比预期差，应该少做
这是 PPO 更新方向的基础。
---
2. Ratio：策略变了多少
PPO 会计算一个比值：
ratio = 新策略下这个动作的概率 / 旧策略下这个动作的概率
- ratio = 1.0：策略没变
- ratio > 1.0：新策略更倾向做这个动作
- ratio < 1.0：新策略更不倾向做这个动作
---
3. Clip：限制更新幅度
PPO 最关键的设计就是这个 clip。
它的目标函数是：
loss = -min(
    ratio × advantage,
    clip(ratio, 1-ε, 1+ε) × advantage
)
在你这个项目里，ε = 0.2。
这个 clip 在干什么
当 advantage > 0（这步做得好）
PPO 想让 ratio 变大（更多地做这个动作）。
但它不让 ratio 超过 1.2。
也就是说：
> "这步确实好，我会让你更多做，但最多只让概率涨 20%。"
当 advantage < 0（这步做得差）
PPO 想让 ratio 变小（少做这个动作）。
但它不让 ratio 低于 0.8。
也就是说：
> "这步确实差，我会让你少做，但最多只让概率降 20%。"
为什么要这么做
因为如果不 clip：
- 某个动作碰巧拿了很高的 reward
- 梯度更新可能一下把这个动作的概率拉得非常高
- 策略突然变得激进
- 然后下一轮数据就完全不对了
clip 保证了：
> 每次更新都是"小步走"，不会一下跳太远。
这就是 PPO 名字里 Proximal（近端） 的含义。
---
PPO 在你这个项目里的完整流程
Step 1：收集数据
用当前 actor 跑 128 步 × 8 环境 = 1024 个 transition。
每个 transition 记录：
- 当时的 obs
- actor 选的 action
- 拿到的 reward
- 是否 done
- actor 当时给这个 action 的 log probability
- critic 当时估计的 value
---
Step 2：算 advantage
用 GAE（Generalized Advantage Estimation） 算法：
从 episode 末尾倒着往前推：
  δ = reward + γ × next_value - current_value
  advantage = δ + γ × λ × next_advantage
其中：
- γ = 0.995：折扣因子，决定多重视未来回报
- λ = 0.95：GAE 的偏差-方差 tradeoff
GAE 直观理解
如果只看一步：
- advantage = 这步 reward + 下一步 value - 当前 value
如果看很多步：
- GAE 把多步的信号加权平均起来
- 短期信号权重更高
- 长期信号也保留一点
这比只看一步更稳定。
---
Step 3：更新 actor
把 1024 个 transition 分成 4 个 mini-batch。
每个 mini-batch 过 8 个 epoch。
总共 32 次梯度更新。
每次更新：
1. 用当前最新的 actor 重新算每个 transition 的 action log prob
2. 算 ratio = 新 log prob / 旧 log prob
3. 算 clipped 目标函数
4. 反向传播
5. 更新 actor 权重
自适应学习率
PPO 还会监控实际的 KL 散度（新旧策略的差异）：
- 如果 KL 太大（更新太激进）：降低 learning rate
- 如果 KL 太小（更新太保守）：提高 learning rate
在你这里，desired_kl = 0.008。
---
Step 4：更新 critic
同时也更新 critic：
value_loss = (critic 预估的 value - 实际 return)²
让 critic 的预测越来越准，这样下一轮算 advantage 就更可靠。
---
PPO 为什么适合这个任务
1. 稳定
clip 机制让策略不会突然崩掉。对机器人控制来说这很重要。
2. 样本效率还行
一批数据可以用 8 个 epoch 反复学习，不是用完就扔。
3. 支持连续动作空间
你的动作是 6D 连续值，PPO 天然支持。
4. 和 asymmetric actor-critic 兼容
actor 和 critic 可以看不同的输入，PPO 不在乎。
---
一句话总结
PPO 的核心
> 收集数据 → 算哪些动作好/坏 → 更新策略但限制更新幅度 → 重复。
clip 的意义
> 防止策略一次更新太多导致训练崩溃。
在你项目里的角色
> PPO 是训练引擎。你定义"环境怎么工作、reward 是什么"，PPO 负责"怎么从数据里学出更好的策略"。
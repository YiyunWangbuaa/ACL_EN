# 机器视觉物体与双目（scene）检查记录

范围：金标源 `data/vision/review/scene/sample_01.md`，编号 **1–165**。  
气动交叉用语已剔除。三层额外目录（archive / basic / reconstruction）已清除，只保留四平行域。  
门禁：`python -m domains.computer_vision.scene_pipeline_check`。

## 阶段 0–7

| 阶段 | 结果 |
| --- | --- |
| 0 稿面 | scene 165 空齐全且无气动用语 |
| 1 依赖图 | 165 节点，18 边，未来边 0 |
| 2 答案形式 | 解析 165/165 |
| 3 计算器 | 声明根 153，派生边 12 |
| 4 D 矩阵 | 119 行 |
| 5 归因 | 错答 119（declaration_root 119） |
| 6 系统修复 | ID41←(37–40)；ID43←41；ID84←(80–83)；ID86←84；ID165 自修 |
| 7 测点 | 4 合成器 latent=0 alias=0 |

## 阶段 8 对照（ID41 核 37–40）

- deepseek-flash，thinking enabled，effort low，3 趟，失败 0。
- 核槽与终点均 **3/3** 错；无核槽全对趟。
- 记录：`results/fork_scene_id41/sample_01/`，报告：`doc/会话分叉_机器视觉scene_ID41.md`。

## 未决

- 分叉臂未开。

# STM Layer Theory Version 4.0 Technical Note

## 一句话
把"表面推理连贯性"与"结构正确性"在任务复杂度标度下的分离形式化为 Structural Gap G(c)=S(c)−A(c)，并明确断言：结构层（Layer 4）不可被更长的 CoT 还原。

## 核心思想
- 分层：Layer 3（表示/语言层）= 语言模拟、局部连贯解释，可产出 plausible 叙述但不保证全局正确；Layer 4（整合抽象层）= 层级抽象、递归、跨约束稳定整合。与框架"θ 表示层 vs (G,R) 结构层"的二分直接对应。
- Structural Gap G(c)=S(c)−A(c)：以"轨迹连贯性（表面）− 正确率（结构成功）"定义"思考幻觉"的可测度；A(c) 建模为渐变下降 × 阈值崩塌（σ(α−βc)·σ(γ−δ(c−c0))），S(c) 允许在 A 崩塌后仍保持高位。
- Effort E(c)（CoT 长度）非单调：中等复杂度上升、崩塌后下降——"beyond which reasoning investment is not sustained"。
- 预测：降低 θ1（复杂度对 P(L=4|c) 的负依赖）可推移转换边界 c*、抑制 Gap；即需要架构层面的结构状态/递归支持，而非更长语言轨迹。

## 支撑框架的位置
- (e) 强支撑：论文显式断言 Layer 4 "is not reducible to longer CoT; it depends on architectural support for structural state, recursion, and constraint-consistent planning"；讨论节再次强调"rather than merely longer linguistic traces"。E(c) 非单调曲线表明预算/CoT 长度在中等复杂度只推高表示层连贯 S(c)，无法阻止 A(c) 在 c0 处的结构崩塌，且崩塌后投入本身衰减——直接支撑框架"预算改善表示层但不改善结构层"。
- (c) 弱支撑：崩塌现象位于"controlled recursive environments"（显式规则的汉诺塔类任务）的复杂度标度下，A(c) 在有效容量阈值 c0 处锐崩，提示结构复杂任务中容量/深度受限；但该文 c(x) 是通用单变量复杂度参数，未直接操纵依赖顺序、图结构或分支度，只能间接支撑"结构层存在阈值式瓶颈"。

## 边界
- 非实验论文：正文 Results 为模型蕴含的 illustrative regimes/预期曲线，经验事实全部引自 Apple (2025)，除汉诺塔轨迹合法率 S(r)=(1/T)Σℓ_t 的定义外无自有实验数字。
- 复杂度 c(x) 为单变量标度参数，不区分依赖深度/分支度/规则复杂度，不能外推到"顺序敏感"或"图结构"的具体结论。
- 不涉及等价改写/表示扰动 (a)、规范化有界性 (b)、序列化 vs 图 (d)。
- 任务域限于显式规则、复杂度可参数化的 puzzle 族；不含开放域或隐式规则任务，也不能据其对任何具体模型架构下结论。

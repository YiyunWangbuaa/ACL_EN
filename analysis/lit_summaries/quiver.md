# QUIVER_ AFormal Framework for Quantifying Perturbation Propagation and Bifurcation in Compound AI Systems

## 一句话
QUIVER 把复合 AI 流水线中的执行变异分解为"输出值漂移（θ 层）与结构路径/迭代分叉（(G,R) 层）"，并实证：语义等价的改写（表示扰动）会经阈值敏感边放大、进而翻转条件分支触发结构分叉（Dshape 12.9%→37.5%）——表示层与结构层的边界本身是可跨越的。

## 核心思想
1. 敏感性矩阵 σij（放大器/吸收器/不敏感边）沿 DAG 路径乘法合成 σ(p)=∏σij，并用 occurrence-lift λ 分离"传播概率"与"传播量级"两个正交信息。
2. 三要素轨迹散度 D=(Diter, Dshape, Doutput) 把两次执行的差异分解为迭代次数/结构路径/输出值三个分量——恰对应框架的 (G,R) 层与 θ 层分离。
3. 分叉阈值 βshape 定义引发结构路径改变的最小上游扰动量；阈值敏感边呈双峰比分布（median<1、少数尾部放大 3–7×），连续量度之外存在决策边界。
4. 拓扑决定传播模式：P（并行+循环）、Q（双 planner+快慢路径）、DSPy 严格链三种结构给出三种敏感性剖面。

## 支撑框架的位置
- **(a) 强支撑**：4 个语义同义的 rewriter prompt 变体（mean d∈[0.696,0.754]）作干预，Dshape 从 12.9% 升至 37.5%——约 1/3 的"等价改写"直接改变执行路径；放大尾部集中于"单词级查询改动的检索边界翻转"（retrieval-boundary flips at single-word query changes）。即表示扰动让 LLM 出错，甚至越入结构层。
- **(c) 强支撑**：路径敏感性乘法近似验证误差 ±4%（rewriter→composer 实测 0.366 vs 乘积 0.356）；System Q 深度级联放大至 9×（Fast LLM→Reranker σ=9.018）；节点插入/删除（web_search=certain、context_sufficiency=True）改变 DAG 拓扑后 Dshape 达 75.9%/63.4%；同为 ~37% Dshape 的不同扰动来自机理不同的级联，只有逐节点分解可辨。
- **(b) 反面/警示证据**：双峰阈值敏感边 + amplify-then-flip 级联（drewriter≈0.225 → ddiscovery≈0.367 → dplanner≈0.01 的分类字段翻转）表明表示层扰动的传播非连续、非处处有界——"有界规范化"不能默认成立；论文本身不做等价性检验。
- **(d) 弱支撑**：无分支的严格顺序链（DSPy）Diter=Dshape=Dstruct=0，结构散度仅出现于有分支/循环的图结构（P 12.9%、Q 25.1%）——链之外的结构才是分叉之源；但论文建模的是多节点图流水线，不涉及单 LLM 内部"图→序列化"。
- **(e) 弱（类比）支撑**：无预算/CoT 实验；但案例 F.2/F.3 显示 prompt 级（表示层）调优只能压幅值，结构性分叉需架构级改动（hybrid composer 分支、删除 Reranker）才消除——与"表示层干预无法根治结构层问题"一致。

## 边界
- 对象是复合 AI 流水线（多节点 LLM 调用图），非单模型内部推理，不触及 token/符号级表示参数。
- 所有 σ/β 值在温度=0、seed=42 的确定性契约与 all-MiniLM-L6-v2 均匀权重实例下测得，只是框架度量的单一实例化。
- σ 局部有效：大改 prompt/模型/索引后须重新估计；观测估计在共享上游时可能有残余混杂。
- βshape≈0.25 是稀疏覆盖下的上界，真实阈值可能更低。
- 无预算、CoT 长度或训练机制实验，不能外推"预算-性能"曲线类论断。

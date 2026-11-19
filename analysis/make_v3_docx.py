# -*- coding: utf-8 -*-
"""Generate the v3 theoretical model Word document."""
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

OUT = r"C:\Users\minec\Desktop\ACL\workflow\results\表示结构分离_理论模型_v3_含溯源表.docx"

doc = Document()

# ---- base styles ----
def set_fonts(style, latin="Cambria", east="宋体", size=None, bold=None):
    style.font.name = latin
    style._element.rPr.rFonts.set(qn("w:eastAsia"), east)
    if size is not None:
        style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold

normal = doc.styles["Normal"]
normal.font.name = "Cambria"
normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
normal.font.size = Pt(11)
normal.paragraph_format.space_after = Pt(4)

bullet_style = doc.styles["List Bullet"]
bullet_style.font.name = "Cambria"
bullet_style._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
bullet_style.font.size = Pt(11)

for name, size, color in [("Heading 1", 15, (0x1F, 0x3B, 0x57)), ("Heading 2", 13, (0x1F, 0x3B, 0x57))]:
    st = doc.styles[name]
    st.font.name = "Cambria"
    st._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
    st.font.size = Pt(size)
    st.font.bold = True
    st.font.color.rgb = RGBColor(*color)

for sec in doc.sections:
    sec.left_margin = Cm(2.5)
    sec.right_margin = Cm(2.5)

def h1(text):
    doc.add_heading(text, level=1)

def h2(text):
    doc.add_heading(text, level=2)

def p(text, bold=False, italic=False):
    par = doc.add_paragraph()
    run = par.add_run(text)
    run.bold = bold
    run.italic = italic
    return par

def formula(text):
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.space_before = Pt(4)
    par.paragraph_format.space_after = Pt(6)
    run = par.add_run(text)
    run.font.name = "Cambria"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    run.italic = True
    return par

def bullet(text):
    par = doc.add_paragraph(text, style="List Bullet")
    return par

def note(text):
    par = doc.add_paragraph()
    run = par.add_run(text)
    run.italic = True
    run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    return par

def table(header, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    for i, htxt in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        r = cell.paragraphs[0].add_run(htxt)
        r.bold = True
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            cells[i].paragraphs[0].add_run(val)
    if widths:
        for i, w in enumerate(widths):
            for row in t.rows:
                row.cells[i].width = Cm(w)
    return t

# ================= 标题 =================
title = doc.add_heading("表示—结构分离：LLM 工程任务推理的条件性理论模型", level=0)
p("版本 v3（依据审阅意见由 v2 重写）。本文档是条件定理、结构性假设与可检验预测组成的框架，不宣称对任意 LLM 与任意工程任务的普适定理；所有结论均显式列出计算模型、任务族与结构假设。")

# ================= 1 =================
h1("1 定位与适用范围")
p("本模型解释两类现象的分离：")
bullet("表示扰动（符号、坐标、单位、正方向等约定改变）可由有界规范化消解，其成本与任务深度无关；")
bullet("结构扰动（依赖图、规则系统或适用域改变）的求解成本由检测、规则选择、依赖传播与验证共同决定，预算需求随图深度与分支规模增长。")
p("相对 v2 的三处实质调整：")
bullet("表示变换空间不默认为有限群作用，改为「有限对称 × 局部重写 × 语义保持变换」的分层组合；")
bullet("「表示/结构」改为任务层定义；模型对表示扰动的路径敏感性作为独立响应变量；")
bullet("所有强结论降级为条件命题或待检验假设，上界/下界均标注其成立条件。")

# ================= 2 =================
h1("2 任务对象")
p("任务实例 x = (s, q)（场景描述 s、查询 q）。其任务语义对象为")
formula("M(x) = (G_x, R_x, θ_x, D_x)")
bullet("G_x = (V_x, E_x)：依赖 DAG（变量、事实、中间状态及其依赖边）；")
bullet("R_x：适用的规则系统（推理规则、物理定律、工程约束）；")
bullet("θ_x：表示约定（坐标轴、正方向、单位、符号命名）；")
bullet("D_x：适用域与边界条件（任务成立的论域、初始与边界约束）。")
p("对查询 q 定义结构量：传播深度 d_q(G)（答案依赖的最长必要路径）、并行宽度 w_q(G)（最大并行依赖规模）、分支规模 b_q(G)（候选推导分支数）。")

# ================= 3 =================
h1("3 扰动分解（任务层定义）")

h2("3.1 两类扰动")
p("定义 3.1（表示等价）。称 x̃ 与 x 表示等价（x̃ ~_rep x），当且仅当")
formula("G_x̃ ≅ G_x，  R_x̃ ≅ R_x，  D_x̃ ≅ D_x，  且 T(x̃) = T(x)")
p("即依赖结构、规则系统与适用域同构，仅表示约定 θ 改变。")
p("定义 3.2（结构扰动）。若上述三条件至少一个不成立，即 (G, R, D) 的同构类改变，则 x̃ 是 x 的结构扰动。")
note("类别是任务层性质：只由任务语义对象决定，与被测模型无关；模型失败不能反过来改变扰动在任务层的类别。")

h2("3.2 表示变换空间（不假定有限群）")
p("完整的表示变换空间写为分层组合：")
formula("G_rep = G_finite ⋉ R_local ⋉ P_semantic")
bullet("G_finite：有限对称变换（轴翻转、轴置换、符号置换）；")
bullet("R_local：有限描述的局部重写（定义逐条改写）；")
bullet("P_semantic：语义保持的自然语言变换（释义、事实重排、扩展、收缩，见 metamorphic testing 对语义保持关系的分类 [1]）。")
p("工程记号题（符号/定义异构）对应 G_finite 与 R_local 的并集。有限群作用是其中的特例：符号集合可增长、新符号可动态生成、自然语言表述可无限变化，因此一般表示变换空间不是固定有限群。")
p("即使变换候选集有限，|G_θ| 也只是候选数量，不等于规范化成本——每个候选的检查成本仍可能随输入长度增长，轨道有限不保证常数时间定位。")

# ================= 4 =================
h1("4 模型层响应与三层安全")

h2("4.1 任务层与模型层分离")
p("给定模型 f，其响应差异为")
formula("Δ_f(x, x̃) = d_Y(f(x), f(x̃))")
p("任务层 x̃ ~_rep x 保持语义等价；模型层 Δ_f 可大可小（敏感性是模型性质）。正确表述：表示扰动在任务语义层仍保持表示等价，但可能诱发模型内部的结构性路径变化——而不是「高灵敏模型下表示扰动变成结构扰动」。")

h2("4.2 三层安全")
table(
    ["层次", "定义", "关系"],
    [
        ["语义安全", "任务对象不变：M(x̃) ≅_θ M(x)", "任务层性质"],
        ["路径安全", "模型的离散推理路径不变", "模型层性质"],
        ["答案安全", "最终答案不变：f(x̃) = f(x)", "输出层性质"],
    ],
    widths=[2.5, 8.5, 3.5],
)
formula("语义安全 ⇏ 路径安全 ⇏ 答案安全")
p("路径安全 ⇒ 答案安全，仅在输出完全由该路径决定时成立。")

h2("4.3 节点级敏感性与脆弱性定位")
p("设决策节点 v ∈ V_q 的表示为 h_v(x)，节点级敏感性取局部 Jacobian 范数（或其有限差分估计）：")
formula("κ_v(x) = ‖∂h_v/∂x‖，   κ_max(x) = max_{v∈V_q} κ_v(x)")
p("按 κ_v 的支撑结构区分两类脆弱性：")
bullet("局部化脆弱：S_loc = {v : κ_v 显著}；")
bullet("分布式脆弱：S_dist = {v : κ_v 非零且广泛分布}。")
formula("ρ_loc = |S_loc| / |V_q|    （可测量的局部化比率）")

h2("4.4 命题 3（局部表示安全的充分条件）")
p("设节点表示满足局部 Lipschitz 界 ‖h_v(x̃) − h_v(x)‖ ≤ κ_v·‖x̃ − x‖，决策边界距离 τ_v = dist(h_v(x), ∂C_v)。若")
formula("max_v κ_v·ε < min_v τ_v")
p("则 Path_f(x̃) = Path_f(x)，即扰动不改变模型在这些节点上的离散决策。")
note("这是充分条件，不是充要条件：即使 ε·κ_v ≥ τ_v，模型仍可能因决策边界方向、冗余证据或后续纠错而保持原路径。κ 必须局部化（κ_v），全局单个 κ 太粗。")

# ================= 5 =================
h1("5 规范化成本（条件性定理）")

h2("5.1 成本分解")
formula("C_norm = C_parse + C_locate + C_apply + C_check")
p("（解析变换描述、定位受影响位置、逐处一致替换、一致性验证。）")

h2("5.2 显式变换的线性界")
p("命题 4。当以下条件同时成立：")
bullet("(i) 变换 g 已显式给出；(ii) 变换规则类型已知；(iii) 受影响位置可线性时间定位；(iv) 一致替换局部可验证；(v) 无需重新证明整体语义等价；")
p("则规范化成本近似线性：")
formula("C_norm = O(|x| + |g| + m)")
p("其中 m = m(x, g) 为受影响出现数。")
p("「显式给出」与「隐式推断」应视为两种计算问题，而非同一有限群定理的两个情形。")

h2("5.3 隐式变换的假设空间界")
p("若变换需从上下文推断，则")
formula("C_norm = O(|H_g| · C_check)")
p("其中 H_g 是候选变换假设空间。工程记号任务中 H_g 有限（如 2^d·d!·s!）只是特例，不是一般结论；C_check 仍可能随输入长度增长。")

h2("5.4 可判定性条件")
p("规范化可证明有界的前提是等价性检查在受限片段内句法可判定；一般语义等价不可判定（Rice 定理 [2]）。受限正片段（C-RASP⁺）有可计算的长度界，完整规则系统的长度界经丢番图方程归约不可计算 [3]。")

h2("5.5 规范化错误（并集界）")
formula("1 − A_F ≤ δ_parse + δ_apply + ε_I(B)，   δ_apply ≤ m·δ_local")
p("其中 ε_I(B) 为规范任务上的推理错误率。δ_apply ≤ m·δ_local 仅在并集界意义下成立；错误事件不可简单相加（解析错后不再应用）。")

# ================= 6 =================
h1("6 结构成本（条件性定理）")

h2("6.1 成本分解")
formula("C_struct = C_detect + C_select + C_prop + C_verify")

h2("6.2 深度下界（在线计算模型假设下）")
p("命题 5（串行时间下界）。对逐步暴露中间状态、每个状态依赖前一状态、且不可预计算、不可并行前缀扫描、不可从训练中直接检索整个映射的在线计算模型，关键依赖链长度 d_q 给出串行时间下界：至少 d_q 个串行步。")
note("此下界不是一般 DAG 任务的普遍下界：存在长图深度但可用闭式公式、并行扫描或预计算解决的任务。")

h2("6.3 有效传播深度（定义 + 建模假设）")
p("单遍链容量界")
formula("2^(L−1) − 1 ≤ S_L^(T,Π) ≤ (3^(L−1) − 1)/2")
p("仅在 Tian 等规定的任务族 T 与信息传播规则 Π 下成立 [4]，不是任意 L 层 Transformer 的普适单遍容量。")
p("CoT 情形下，定义有效传播深度为待估计量（定义而非定理）：")
formula("S_eff(B) = 预算 B 下经实验证实的有效依赖传播深度")
p("并提出条件性放大假设：")
formula("S_eff(t) ≤ α·t·S_L + β")
p("其中 α、β 由架构、提示格式与状态保留机制决定。已有理论支持 CoT 增强 Transformer 的串行计算能力，但依赖特定架构与复杂度假设 [5][6]；递归深度可改善深度外推但过度递归有害 [7]，故「预算增加」≠「有效推理深度增加」。")

h2("6.4 传播误差的幂律模型（待检验假设）")
formula("E_prop(d, w) = α · d^ν · w^(−η) · κ(G)，   α > 0，ν ≥ 1，η ≥ 0")
p("ν = 3/2、η = 1/2 来自特定聚合传播模型（链式 vs 星/网格的闭式界），作为待检验的特殊假设，不作为一般定理；κ(G) 为图条件数。单步误差定义、宽度与并行路径的对应、误差独立性、误差范数与概率错误率的转换均需在实验中明确。")

h2("6.5 规则选择分解（含补偿项）")
formula("A_S(B) = p_D(B)·p_R(B)·p_P(B | R*)·p_V(B | R̂) + P(ŷ = y* | R̂ ≠ R*)·P(R̂ ≠ R*)")
p("p_D 检测偏离默认结构、p_R 选对规则系统、p_P 正确规则下完成传播、p_V 验证通过；补偿项避免低估「错误规则下偶然正确」的概率。")

# ================= 7 =================
h1("7 验证—搜索成本不对称假设")
p("（作为假设陈述，不作为普遍定理。）")
p("给定候选解与局部检查规则，验证成本为算法上界：")
formula("C_verify = O(|E| · c_local)")
p("搜索成本为逐层分支之和：")
formula("C_search = O(Σ_{h=0}^{d_q} b_h)")
p("仅当 b_h ≥ b > 1（∀h）时才有最坏情况下界 C_search = Ω(b^{d_q})。很多任务可经动态规划、约束传播、剪枝或闭式公式避开指数搜索。")
p("规范化属验证类（检查 x̃→x 的逐处一致性），结构求解属搜索类；两类成本的差距因此是条件性的，不是先验事实。")

# ================= 8 =================
h1("8 预算响应")

h2("8.1 表示任务（局部错误累积型）")
formula("A_F(B) = p_parse(B) · p_apply(B | parse) · p_solve(B | normalized)")

h2("8.2 结构任务（阶段瓶颈与乘积效应型）")
formula("A_S(B) = p_D(B) · p_R(B) · p_P(B) · p_V(B)")

h2("8.3 预算指标")
bullet("预算利用率：U(B) = S_eff(B) / d_q(G)；U(B) < 1 表示传播能力不足。")
bullet("预算边际收益：Δ_A(B; Δ) = A(B + Δ) − A(B)；平台定义为连续多个预算区间 Δ_A(B; Δ) < ε。")
note("删除 v2 的 G(B) = S(B) − A(B)：S(B) 是深度/容量、A(B) 是概率，量纲与语义不一致。")

h2("8.4 核心预测（条件性）")
formula("∂A_F/∂B > ∂A_S/∂B")
p("仅对「表示扰动主要受规范化限制、结构扰动主要受检测或规则选择限制」的任务族成立；不写成无条件不等式。")

# ================= 9 =================
h1("9 核心命题（推荐表述）")
p("对语义保持、局部、显式或可判定的表示扰动，额外计算主要承担规范化与一致性检查，其成本由输入长度、变换描述长度和受影响位置数控制。对改变依赖图、规则系统或适用域的结构扰动，额外计算还必须承担异常检测、规则选择、依赖传播和验证，因此其预算需求由图深度、分支规模、传播条件数和验证结构共同决定。两类扰动在任务语义层面可区分，而模型对表示扰动的路径敏感性应作为独立的响应变量分析。", bold=True)

# ================= 10 =================
h1("10 可检验预测")
table(
    ["预测", "形式", "对应实验"],
    [
        ["符号异构修复成本不随任务深度增长", "C_norm = O(|x| + |g| + m)", "逐空规范化错误率 vs 题深"],
        ["结构任务预算需求随深度增长", "S_eff(B) < d_q 时正确率受限", "端到端错误率 vs 子过程深度"],
        ["路径安全充分条件", "max_v κ_v ε < min_v τ_v", "符号改写越界处错误聚簇"],
        ["宽度阻尼", "E_prop ∝ w^(−η)", "并行子过程 vs 串行链"],
        ["验证/搜索不对称", "gold-trunc ≫ final-e2e", "前缀增益对比（+87.5pp）"],
        ["幂律标度", "ν ≈ 1.5，η ≈ 0.5", "不同深度/宽度图任务拟合"],
    ],
    widths=[4.5, 4.5, 5.5],
)

# ================= 11 =================
h1("11 假设清单（显式边界）")
table(
    ["#", "假设/边界", "来源与性质"],
    [
        ["A1", "预算 B 为 CoT 步数计算族；无 CoT ⊆ TC⁰；CoT(t) ⊆ SPACE(t + log n)", "Merrill–Sabharwal [5]，能力类"],
        ["A2", "单遍容量界 S_L^(T,Π) 只适用于 Tian 等的任务族与传播规则", "[4]，适用范围限定"],
        ["A3", "S_eff(t) ≤ αtS_L + β 为建模假设，α、β 待估", "定义 + 假设，非定理"],
        ["A4", "E_prop = α d^ν w^(−η) κ(G) 为幂律模型；ν=3/2、η=1/2 待检验", "经验模型，非定理"],
        ["A5", "深度下界只在在线计算模型下成立", "命题 5 的条件"],
        ["A6", "验证/搜索不对称是假设，C_verify 是算法上界", "假设，非定理"],
        ["A7", "错误合并用并集界与条件概率；δ ≤ 14ε^KJ 仅在原文概率假设下引用", "概率工具"],
        ["A8", "一般语义等价不可判定 ⇒ 规范化可判定性要求受限片段", "Rice 定理 [2]，C-RASP 分层 [3]"],
    ],
    widths=[1.0, 7.5, 6.0],
)

# ================= 附录 A =================
h1("附录 A  理论依据文献溯源表")
p("说明：「直接引用」为原样采用原文献结论；「条件化引用」为采用原结论并显式限定其适用范围；「建模假设」为借鉴原文献机制但改写为待检验假设；「经验支撑」为使用原文献的实测数字作为证据；「类比迁移」为把原文献在另一设定下的量迁移到本框架。")
table(
    ["v3 条目", "节", "溯源文献", "原文献性质", "使用方式"],
    [
        ["任务对象 M(x)=(G,R,θ,D)", "2", "本工作（D_x 依审阅意见新增）", "建模定义", "定义"],
        ["表示等价 x̃~_rep x 与结构扰动", "3.1", "本工作 v1/v2", "建模定义", "定义"],
        ["G_rep = G_finite ⋉ R_local ⋉ P_semantic", "3.2", "[1] 语义保持变换分类；[9] 改写范畴", "经验分类 / 范畴理论", "修正后的定义"],
        ["任务层—模型层分离 Δ_f", "4.1", "[9] success invariance 违反属模型行为", "理论", "定义（依审阅意见）"],
        ["三层安全：语义/路径/答案", "4.2", "本工作 + 审阅意见", "建模定义", "定义"],
        ["κ_v 与 S_loc/S_dist、ρ_loc", "4.3", "[10] 谱条件数；[12] localized/distributed 签名", "理论 + 机制实验", "类比迁移 + 经验支撑"],
        ["命题 3：ε·κ<τ ⟹ 路径保持", "4.4", "[11] 阈值分叉；标准 Lipschitz 论证", "理论 + 实验", "条件化（充要→充分）"],
        ["C_norm = C_parse+C_locate+C_apply+C_check", "5.1", "本工作 + 审阅意见", "建模定义", "定义"],
        ["命题 4：C_norm = O(|x|+|g|+m)", "5.2", "本工作定理 1 + 审阅五条件 (i)–(v)", "命题", "条件性命题"],
        ["隐式推断 O(|H_g|·C_check)", "5.3", "[9] 集成在采样极限恢复不变性（Prop 14）", "定理", "建模命题"],
        ["可判定性分层：受限片段可算 / 完整不可判定", "5.4", "[3] C-RASP 不可计算界；[9] B.5 Rice 不可判定", "定理", "直接引用"],
        ["并集界 1−A_F ≤ δ_parse+δ_apply+ε_I", "5.5", "概率论标准工具", "定理", "直接引用"],
        ["深度下界（在线计算模型）", "6.2", "标准 DAG 深度论证 + 审阅条件", "命题", "条件性命题"],
        ["S_L^(T,Π) ∈ [2^(L−1)−1, (3^(L−1)−1)/2]", "6.3", "[4] 定理 6.1/6.2、推论 6.4", "定理", "条件化引用（限定 T,Π）"],
        ["S_eff 定义；S_eff(t) ≤ αtS_L+β", "6.3", "[5] SPACE 界；[6] CoT 解串行任务；[7] 递归深度", "定理 / 实验", "建模假设（非定理）"],
        ["E_prop = α·d^ν·w^(−η)·κ(G)", "6.4", "[10] 闭式误差界 T^{3/2}/√k", "闭式界", "参数化假设（ν、η 待估）"],
        ["A_S 四因子分解 + 补偿项", "6.5", "本工作 v1 的 p_D·p_R·p_P + 审阅 p_V", "建模分解", "定义"],
        ["验证—搜索成本不对称", "7", "[14] 拓扑序归纳验证；[13] 核验 vs 搜索不对称", "方法 + 实验", "假设（非定理）"],
        ["A_F / A_S 概率分解", "8.1–8.2", "本工作 + 审阅意见", "建模分解", "定义"],
        ["U(B) 与 Δ_A（删除 G(B)）", "8.3", "本工作 + 审阅意见", "指标", "可检验指标"],
        ["核心预测 ∂A_F/∂B > ∂A_S/∂B", "8.4", "[19] 结构缺口 G(c)=S(c)−A(c)；[13] 前缀救表面不救内核", "理论 + 实验", "条件化假设"],
        ["无 CoT ⊆ TC⁰；CoT(t) ⊆ SPACE(t+log n)", "A1", "[5] Thm 4", "定理", "直接引用"],
        ["δ ≤ 14ε^KJ（概率等价检验）", "A7", "[13]", "概率界", "条件化引用（限原文概率假设）"],
        ["表示扰动致错（经验证据）", "4.4、10", "[9][12][13][16][17][18]", "实验", "经验支撑"],
        ["结构敏感与图验证（经验证据）", "6.3、7", "[4][10][11][14][15]", "实验", "经验支撑"],
        ["长度泛化差异（背景动机）", "A8", "[8]", "实验", "背景动机"],
    ],
    widths=[4.2, 1.2, 4.6, 2.0, 3.0],
)

# ================= 参考文献 =================
h1("参考文献")
refs = [
    "[1] J. De Curtò, I. De Zarzà. Metamorphic Testing for Semantic Invariance in Large Language Models. IEEE Access, 2025.",
    "[2] （语义等价不可判定性：Rice 定理的引用见 What are the Right Symmetries 附录 B.5；正式投稿前核对原始书目。）",
    "[3] A. Yang, P. Bergsträßer, G. Zetzsche, D. Chiang, A. Lin. Length Generalization Bounds for Transformers. 2026.",
    "[4] Q. Tian, Y. Chen, Z. Wang, Z.-Q. J. Xu. Limit Analysis for Symbolic Multi-Step Reasoning Tasks with Information Propagation Rules Based on Transformers. arXiv:2509.23178.",
    "[5] W. Merrill, A. Sabharwal. The Expressive Power of Transformers with Chain of Thought. ICLR 2024.",
    "[6] Z. Feng et al. Chain of Thought Empowers Transformers to Solve Inherently Serial Problems. ICLR 2024.（作者列表投稿前核对）",
    "[7] Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers. 2026.（书目信息待核对）",
    "[8] H. Zhou et al. What Algorithms Can Transformers Learn? A Study in Length Generalization. arXiv:2310.16028.",
    "[9] What are the Right Symmetries.（改写范畴、success invariance、语义等价不可判定；书目信息见文献摘要）",
    "[10] Predictive Maps of Multi-Agent Reasoning: A Successor-Representation Spectrum for LLM Communication Topologies.（条件数 κ 与 T^{3/2}/√k 误差界来源）",
    "[11] QUIVER: A Formal Framework for Quantifying Perturbation Propagation and Bifurcation in Compound AI Systems.（阈值分叉证据来源）",
    "[12] Fragile Reasoning: A Mechanistic Analysis of LLM Sensitivity to Meaning-Preserving Perturbations.（局部化/分布式脆弱性证据）",
    "[13] An Investigation of Robustness of LLMs in Mathematical Reasoning: Benchmarking with Mathematically-Equivalent Transformations.（PutnamGAP；概率等价检验界）",
    "[14] Graph of Verification: Structured Verification of LLM Reasoning.（拓扑序归纳验证）",
    "[15] GNNVerifier: Graph-based Verifier for LLM Task Planning.（结构错误的图感知验证）",
    "[16] Measuring Representation Robustness in Large Language Models for Geometry.（规范化修复的容量门槛）",
    "[17] Lost in Reordering: Structural Sensitivity of Multilingual LLMs under Semantics-Preserving Perturbations.",
    "[18] Semantic Invariance in Agentic AI.",
    "[19] STM Layer Theory Version 4.0 Technical Note.（表面连贯与结构正确的分离）",
]
for r in refs:
    p(r)

doc.save(OUT)
print("saved", OUT)

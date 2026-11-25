# -*- coding: utf-8 -*-
"""Generate the ACL Methods chapter (Chinese first draft) as a Word document."""
from docx import Document
from docx.shared import Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

OUT = r"C:\Users\minec\Desktop\ACL方法章节_初版.docx"

doc = Document()

# ---------- base styles ----------
def set_font(style, latin, east, size, bold=False, color=None):
    style.font.name = latin
    style.font.size = Pt(size)
    style.font.bold = bold
    if color is not None:
        style.font.color.rgb = color
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn('w:rFonts'))
    if rfonts is None:
        rfonts = rpr.makeelement(qn('w:rFonts'), {})
        rpr.append(rfonts)
    rfonts.set(qn('w:eastAsia'), east)

normal = doc.styles['Normal']
set_font(normal, 'Times New Roman', '宋体', 11)
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.25

set_font(doc.styles['Heading 1'], 'Arial', '黑体', 15, bold=True, color=RGBColor(0, 0, 0))
set_font(doc.styles['Heading 2'], 'Arial', '黑体', 12.5, bold=True, color=RGBColor(0, 0, 0))
set_font(doc.styles['Heading 3'], 'Arial', '黑体', 11.5, bold=True, color=RGBColor(0x33, 0x33, 0x33))

for sec in doc.sections:
    sec.top_margin = Cm(2.2)
    sec.bottom_margin = Cm(2.2)
    sec.left_margin = Cm(2.5)
    sec.right_margin = Cm(2.5)

# ---------- helpers ----------
def run_font(run, latin='Times New Roman', east='宋体', size=None, bold=None, italic=None, color=None):
    run.font.name = latin
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn('w:rFonts'))
    if rfonts is None:
        rfonts = rpr.makeelement(qn('w:rFonts'), {})
        rpr.append(rfonts)
    rfonts.set(qn('w:eastAsia'), east)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = color

def para(text='', style=None, align=None, space_after=None):
    par = doc.add_paragraph(style=style)
    if text:
        run = par.add_run(text)
        run_font(run)
    if align is not None:
        par.alignment = align
    if space_after is not None:
        par.paragraph_format.space_after = Pt(space_after)
    return par

def bullet(text, bold_lead=None):
    par = doc.add_paragraph(style='List Bullet')
    if bold_lead:
        r1 = par.add_run(bold_lead)
        run_font(r1, bold=True)
        r2 = par.add_run(text)
        run_font(r2)
    else:
        run = par.add_run(text)
        run_font(run)
    return par

def math(text):
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.space_before = Pt(4)
    par.paragraph_format.space_after = Pt(4)
    run = par.add_run(text)
    run_font(run, latin='Cambria Math', east='宋体', size=11.5)
    return par

def code_block(text):
    par = doc.add_paragraph()
    par.paragraph_format.space_before = Pt(4)
    par.paragraph_format.space_after = Pt(6)
    for i, line in enumerate(text.split('\n')):
        if i > 0:
            par.add_run().add_break()
        run = par.add_run(line)
        run_font(run, latin='Consolas', east='宋体', size=9.5)
    return par

def table(headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = 'Table Grid'
    hdr = t.rows[0].cells
    for j, h in enumerate(headers):
        hdr[j].text = ''
        run = hdr[j].paragraphs[0].add_run(h)
        run_font(run, bold=True, size=10)
    for row in rows:
        cells = t.add_row().cells
        for j, val in enumerate(row):
            cells[j].text = ''
            run = cells[j].paragraphs[0].add_run(val)
            run_font(run, size=10)
    if widths:
        for j, w in enumerate(widths):
            for row in t.rows:
                row.cells[j].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t

# ================= TITLE =================
tp = para('ACL 方法章节 · 核心设计（初版）', align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
for run in tp.runs:
    run_font(run, latin='Arial', east='黑体', size=18, bold=True)
para('（面向 LLM 审稿人；中文初版，正式投稿前统一译为英文）', align=WD_ALIGN_PARAGRAPH.CENTER, space_after=12)

# ================= 编者说明 =================
doc.add_heading('编者说明（提交前删除本页）', level=2)
bullet('语言：中文（按你的选择）。正式投稿前统一翻译为英文，术语对照见文末附录。')
bullet('章节定位：本文件拟作为论文第 3 节「方法（Method）」，小节号 3.1–3.9 可直接并入；若实际 Methods 是第 4 节，整体平移编号即可。')
bullet('面向 LLM 审稿人的重写策略：把「故障诊断 / 签名矩阵 / 极小碰集」等工程术语改写为 LLM 社区熟悉的过程监督、逐步验证器、归因、反事实 等概念；每个形式化定义都补一句「这回答了 LLM 评测的什么问题」。')
bullet('范围：本版只覆盖「核心设计三层」（任务形式化 → 确定性数据生成 → 错误诊断与因果归因），不含 A–E 评测协议、实验结果与相关工作。')
bullet('评测集名称需重拟：当前「FD-Eval」带故障诊断色彩，应体现「异构工程契约」主题；请提供新名，或授权我拟名。')
bullet('已同步：故障类 E1–E5 → DIR/RSN/EXE/ALG/RES（识别 = 一个能力、两个深度）；中间计算器改为契约认知重构（在模型认知契约 ĉ 下执行父节点，见 3.3/3.6）。')

para('待你确认的写作决策：', space_after=2)
bullet('「可诊断的系统模型」这一类比需专门系统性的论述（不止一两句）；本版先给出骨架，是否扩成独立小节或放相关工作，待定。')
bullet('评价指标应单独成一节、用公式形式化，并只保留核心指标（当前三层指标表偏多）；本版保留汇总表作占位。')
bullet('正文把「故障诊断」改写为「错误诊断 / 归因」；英文术语首次出现处用括号标注，对译若有偏差请直接批注。')

# ================= 3 方法 =================
doc.add_heading('3  方法（Method）', level=1)

para('本文提出一个面向异构工程契约的过程级评测框架，把「工程契约遵循」作为显式、可观测、可归因的评测对象。'
     '工程推理与一般算术或证明推理的关键区别在于：正确答案是契约敏感的（随契约可变）——它由题面声明的一套工程契约'
     '（坐标轴指向、角度正方向、变换约定、旋转顺序、安装方式、符号约定）所决定。同一物理场景、同一计算流程，'
     '只要契约字段改变，正确答案就会系统性改变。因此工程推理的核心能力不只是「算对」，而是'
     '「准确识别契约条件，并一致性地遵循契约推导出正确答案」。')

para('现有过程级评测大多只区分「步骤对/错」（process supervision）或「最终答案对/错」，无法回答三个更细的问题：'
     '(i) 一个错误是「没读懂契约」还是「读懂了却执行错」；(ii) 一个「最终正确」的答案，其过程是可靠地正确，'
     '还是带着可追溯的遮蔽（过程有错、答案碰对），还是矛盾下的随机命中；(iii) 某个上游错误对最终目标的影响'
     '是因果的还是仅仅相关。本框架把评测数据从传统的「题目 + 参考答案」提升为「可诊断的系统模型」：'
     '每个任务被建模为一个确定性系统（依赖图 + 局部规则 + 契约），LLM 在该系统中作为待诊断的黑盒组件——'
     '我们不假设其内部结构，只观察它在各测点上的输出，并依据系统模型做故障隔离、错误归因与错误传播路径分析。'
     '这一「系统建模」的类比贯穿全篇：评测数据从一份份孤立的问答，变成一台可被诊断的仪器；'
     '「模型为什么错」从后验猜测变成一个可计算的诊断问题。方法分三层：设计层（构造系统、审计可诊断性）→ 诊断层（归因错误）→ 因果层（量化影响）。')

# ---- 3.1 ----
doc.add_heading('3.1  任务形式化：契约、骨架与目标', level=2)

para('一道工程任务分解为三元组：')
math('T = (x, c, G)')

bullet('x：物理场景（飞机模型、相机布置、机械臂构型），描述「世界是什么」。', bold_lead='物理场景 ')
bullet('c = (c_1, …, c_K)：工程契约，一组有限的约定/定义字段（坐标轴指向、角度正方向、变换约定、旋转顺序、安装方式、符号约定），描述「题面要求遵守什么规则」。', bold_lead='工程契约 ')
bullet('G = (V, E)：逻辑骨架，一个有向无环图（DAG）。节点 v ∈ V 是一个待回答的步骤或答案字段，边 (u → v) 表示 v 的推导使用 u 的输出。', bold_lead='逻辑骨架 ')

para('核心设计：契约与骨架分离。对同一场景和骨架，可系统改变契约字段而不必重写推理流程：')
math('T_c = (x, c, G)。')

para('契约之所以要作为独立系统变量，是为了模拟真实工程中异构契约并存的处境：对同一物理实体，不同约定（坐标系、正方向、旋转顺序、安装方式）'
     '会给出系统性不同的正确答案。把契约单独抽出后，我们就能生成「反常识」的契约条件（例如轴向向后为正、反装、非标准旋转顺序），'
     '真实考察 LLM 对契约的理解与应用能力，防止它依赖训练分布中的先验约定对答案做直接模板匹配。契约同时进入模型提示与参考执行程序，'
     '使「是否遵循契约」成为可测量性质：模型若回落到默认约定，就会被显式观测。')

# ---- 3.2 ----
doc.add_heading('3.2  推理图与测点', level=2)

para('为什么要显式建模推理图与测点：契约遵循是 LLM 的内化能力，无法由目标问题对错反映出来；它与 LLM 的计算能力、语义理解、'
     '任务规划能力相耦合，且契约遵循本身是综合能力——先「识别契约」（读对契约 + 按契约想对路线），再「执行契约」。'
     '这些能力只呈现在隐含的中间推理步骤里，因此必须设置显性的、针对性的问题节点来暴露相应的能力缺陷。')

para('每个参考流程表示为带契约条件的有向图：')
math('G_c = (V, E, {φ_v}_{v∈V})')
para('其中 V 是答案节点集；E ⊆ V × V 是推导依赖边，(u → v) ∈ E 表示 v 的推导直接使用 u 的输出；'
     'φ_v 是节点 v 在契约 c 下的局部规则——「给定父节点值与契约，v 的答案应是什么」。')

para('按能力维度把测点分四类（与 3.4 节的故障类一一对应）：')
table(
    ['测点类', '暴露的能力', '对应故障类'],
    [
        ['契约识别测点', '读对契约（直接识别 + 推理理解）', 'DIR / RSN'],
        ['契约执行测点', '把认出的契约写进对象', 'EXE'],
        ['数理应用测点', '对已写好的量做计算', 'ALG'],
        ['目标问题', '上述能力综合', '—'],
    ],
    widths=[3.5, 6.0, 3.0],
)

para('测点 m = (v_m, π_m, g_m, φ_{v_m}) 可对应整个节点，也可对应节点答案的一个字段（矩阵元素、向量分量、外层符号、力臂、分子/分母）。'
     '字段级测点把能直接反映契约执行失败的关键错误模式（符号反转、特定位置的元素形式）分离出来，使局部选择直接可观测。')

para('四个领域的规模统计（按测点类型细化，具体数字见实验部分）：')
table(
    ['领域', '节点数', '覆盖的工程契约'],
    [
        ['风洞空气动力学', '207', '坐标约定、姿态变换、力/力矩轴向'],
        ['飞行力学', '131', '气动量与载荷换算'],
        ['机器视觉', '66 + 165', '位姿、双目几何'],
        ['机器人运动学', '156', '齐次变换、运动学求解'],
    ],
    widths=[4.0, 2.5, 8.5],
)

# ---- 3.3 ----
doc.add_heading('3.3  契约条件化的确定性数据生成', level=2)

para('数据由三个功能模块的确定性程序生成（无模型调用、无标注噪声）。三者以「契约 → 骨架 → 执行」的顺序协同：'
     '契约实例化器把契约 c 展开为题面中的工程定义与方法要求；骨架生成器按 (x, G) 把它切成逐步问题并标注依赖；'
     '契约条件化执行器沿骨架执行局部规则，产出金标轨迹。三者共享同一份契约对象 c 与骨架 G，保证题面、依赖、金标三者一致。')
bullet('契约实例化器（contract instantiator）：c → 题面中的工程定义与方法要求。', bold_lead='1. ')
bullet('骨架生成器（skeleton generator）：(x, G) → 逐步问题及其依赖。', bold_lead='2. ')
bullet('契约条件化执行器（contract-conditioned executor）：在 c 下沿骨架执行局部规则，生成金标轨迹。', bold_lead='3. ')

para('局部计算规则依赖契约与父节点值（生成金标用正确契约 c）：')
math('g_v = φ_v( g_Pa(v), x, c )。')
para('金标轨迹与目标量：')
math('τ* = ( g_{v_1}, …, g_{v_{|V|}} )，      y* = Target(τ*)。')
para('题面、金标、目标统一由 q = Gen(x, c, G) 与 τ* = Exec(x, c, G) 生成。')

para('契约组合实例化。契约组合空间 Θ ⊆ C_1 × ⋯ × C_K，每个 θ ∈ Θ 生成：', space_after=2)
math('q_θ = Gen(x, θ, G)，      τ*_θ = Exec(x, θ, G)。')
para('保持物理实体与目标不变、只改契约，生成结构可比、契约不同的评测实例。关键点：金标不是「从标准答案变换得到的标签」，'
     '而是该契约条件下完整执行骨架的结果——「改契约」与「换标签」有本质区别：前者改变的是推理过程本身的合法性条件，后者只是改变对错标签。')

para('两层计算与契约认知重构。确定性程序包含两个层级的计算：', space_after=2)
bullet('金标层：用金标父节点、正确契约 c 执行 φ_v，输出自洽标准结果。', bold_lead='')
bullet('局部计算层：用模型实际父节点、模型认知契约 ĉ 执行 φ_v，输出模型前缀下的条件结果。', bold_lead='')
para('契约认知重构是本方法的关键：局部计算层不直接按正确契约执行父节点，而是先从模型在识别测点（DIR/RSN）上的回答'
     '重构出模型实际认知到的契约 ĉ（例如由模型写下的坐标轴三元组推断其认知的手性，而非读金标的手性），再在 ĉ 下执行：')
math('õ_v = φ_v( o_Pa(v), x, ĉ )。')
para('这样才能区分「识别错」（ĉ ≠ c，模型忠实执行了错误契约）与「执行错」（ĉ = c，模型没把它认对的契约做对）。'
     '当模型父节点违反契约约束（轴绑定不成组、连乘码重复、顺序非法、三元组写不齐）时：')
math('φ_v( o_Pa(v), x, ĉ ) = ⊥。')
para('约束违反在局部节点截断，不向深层传播：浅层违约后深层推理大概率矛盾不自洽，确定性系统无法为矛盾前缀构造自洽全局解释，'
     '只能认定深层受上游传播影响。这一「局部截断」是本方法的重要设计决策。')

# ---- 3.4 ----
doc.add_heading('3.4  故障模型：可注入、可验证的错误假设', level=2)

para('为把「归因」从主观判断变成可计算命题，我们把模型的错误空间建模为一组可注入、可验证的错误假设。'
     '识别是同一个能力的两个深度，执行和数理是后面的能力（类名取英文关键词缩写，不再用 E 代号）：')
table(
    ['类', '能力', '含义'],
    [
        ['DIR', '识别 · 直接', '契约写明映射，读出来即可（Direct）'],
        ['RSN', '识别 · 推理', '契约只暗示路线，要理解才能选中（Reason）'],
        ['EXE', '执行', '认出之后照着做（Execute）'],
        ['ALG', '数理', '做的时候算错（Algebra）'],
        ['RES', '残差', '未建模，无注入（Residual）'],
    ],
    widths=[2.0, 3.0, 10.0],
)
para('契约类 = 识别（DIR + RSN）+ 执行（EXE）。DIR 测能不能读对契约，RSN 测能不能按契约想对路线，EXE 测想对之后能不能做对。'
     'ALG 不依赖契约、只读已写好的中间量，因此不受上游契约错误影响——这一区分让「契约错误」与「代数错误」在机制上可分离。')

para('故障实例库：')
table(
    ['类', '键', '一行代表'],
    [
        ['DIR', '每条写明映射的识别契约', '一条直接识别错'],
        ['RSN', '每条需推理的路线（≥2 空）', '一条推理识别错'],
        ['EXE', '每个派生父空', '一个父空写错（见证 = 被最多计算器接受的错文）'],
        ['ALG', '每个数理空', '一个矩阵/向量/表达式上的数理错（单一滑移作见证）'],
    ],
    widths=[2.0, 5.0, 8.0],
)
para('口径：故障定位在「答案写错的空」，不在「读到错的子节点」；子节点的读取是传播，交给递归并集处理，不另立行。')
para('每个故障实例 f 在依赖图上带一个直接写入集 Direct(f)（它直接改写的空/测点）与一个可达集 Reach(f)'
     '（Direct(f) 的后代闭包，即可能受其影响的测点）。二者是 3.5 节图可达审计的输入：Direct 决定「根在哪」，Reach 决定「影响能传到哪」。')

# ---- 3.5 ----
doc.add_heading('3.5  可诊断性审计：二值 D 矩阵（图可达）', level=2)

para('D 矩阵采用标准故障签名矩阵（FSM）的二值支持层语义：D_{f,m} ∈ {0,1}，D_{f,m}=1 ⟺ 故障 f 会影响到测点 m——'
     '它是「会不会变」的支持关系，不是「变成什么值」的预测。本方法直接从确定性系统的依赖图计算这一支持层（零注入、零重算）：')
math('D_{f,m} = 1[ m ∈ Reach(f) ]。')
para('该方法由模型故障诊断中的结构可诊断性分析启发迁移而来：在确定性系统 + 显式依赖图的前提下，「能否影响到」是纯图结构性质，'
     '因此只需一次图遍历即可得到全部故障-测点支持关系，避免了逐实例注入重算的冗余与大规模遍历成本。')
para('由此，D 矩阵定位在数据审阅（设计）阶段，是一个辅助设计者设计题面测点的工具：它回答「当前测点能否检测、隔离每类错误，'
     '以及应当在哪些节点补测点」。需强调：D 矩阵只承载二值支持层，不承载任何预测值；诊断中真正用到的「值」来自 φ_v 对模型'
     '真实错值的条件重算（见 3.6），两者是两回事，不能混为一谈。')

para('模糊组按二值支持列归并，分两类读：')
table(
    ['类型', '特征', '处理'],
    [
        ['类标签重复', '组员注入同一空/契约，只是 DIR/RSN/EXE 标签不同', '合并（一空一条根，不另立类）'],
        ['真模糊', '组员注入不同空，却共享同一可达集（支持列相同）', '加测点劈开'],
    ],
    widths=[3.5, 7.0, 4.5],
)
para('合并掉类标签重复之后，真正的「异空同 reach」模糊组才浮出水面——只有这类才需要补测点。可检测性与可隔离性均由二值支持列决定：'
     'f 可检测 ⟺ ∃m: D_{f,m}=1；f_i 与 f_j 可隔离 ⟺ ∃m: D_{f_i,m} ≠ D_{f_j,m}（支持列不同）。')
para('（LLM 视角：这一步是评测数据设计的「可诊断性预检」——在真实调用模型之前，先靠图结构确认这套题面测点是否足以暴露并区分'
     '我们关心的错误类型，相当于在数据发布前对测点设计做一次静态审计。）')

# ---- 3.6 ----
doc.add_heading('3.6  过程级诊断：局部重算与五类分类', level=2)

para('条件局部重算 õ_v = φ_v(o_Pa(v), x, ĉ)（在模型认知契约 ĉ 下重算，ĉ 由识别测点回答重构，见 3.3），局部一致性：')
math('LC(v) = 1[ o_v ≡_v õ_v ]。')
para('两根轴分开：忠实轴（o_v = õ_v vs o_v ≠ õ_v）用于读数（忠实 vs 局部偏差）；'
     '确定性轴（φ_v 有值 vs ⊥）用于可归因性（因果 vs 关联）。局部一致性 LC 只落在忠实轴上，'
     '因果/关联的切分不依赖 LC，而依赖确定性轴。')

para('五类 = 答案对错 × 可归因性（确定性）+ 忠实基线：')
table(
    ['答案', '可归因性（确定性）', '分类'],
    [
        ['正确', '忠实（父全对）', '一致性正确'],
        ['正确', '可归因（φ 有值，过程有错）', '可追溯答案遮蔽'],
        ['正确', '不可归因（φ=⊥，矛盾）', '随机性正确（未诊断）'],
        ['错误', '可归因（φ 有值）', '可归因错误'],
        ['错误', '不可归因（φ=⊥）', '不可归因错误'],
    ],
    widths=[2.5, 6.0, 6.5],
)
para('「可归因性」是确定性轴（φ 有值 vs ⊥），与忠实轴解耦。局部执行偏差（EXE）是因果根：即使 o_v ≠ õ_v（LC=0），'
     '只要 φ_v 有值，错误仍沿规则确定性传播，因此 DIR、RSN、EXE、局部偏差都是因果。可追溯答案遮蔽合并了原来的「确定性掩盖」'
     '（o=φ=g、父错）与「不忠实的正确」（o=g、o≠φ）：二者都是答案对、过程有错、但系统未退化随机，都可追溯到具体故障。')
para('（LLM 视角：这一步对应「过程监督」中「步骤正确性」的严格化，且更进一步——它把「最终答案对/错」与「过程能否被确定性解释」'
     '两个维度交叉，得到完备且不重叠的五类；「正确」不再是一团，而是拆成忠实正确、可追溯遮蔽、随机命中三档。）')

# ---- 3.7 ----
doc.add_heading('3.7  归因：候选集、可归因/关联切分', level=2)

para('递归保守归并。不因「中间节点正确」而截断归因。对每个错测点 m（位于节点 v），先看该字段与 φ_v 是否一致：'
     '一致则只收祖先里的故障；不一致、无规则或 φ_v=⊥ 则本空注入 ∪ 祖先故障。祖先故障须注入点落在 v 的严格祖先内，'
     '且二值可达集盖住该测点。由此得到候选集（超集），而不是冲突集：')
math('F(t) = { 局部错 at t } ∪ { w : w ∈ Anc(t), o_w ≠ g_w }。')
para('中间「正确」无法区分「因果链恢复」与「可追溯遮蔽」，因此不假设恢复，递归归并所有错误祖先（用精度换完备性）。'
     '这与「找到第一个错步就停」的归因不同——那种做法会漏掉被遮蔽的上游错误。')

para('可归因/关联切分。对候选根 w 到目标 t 的路径，判据是确定性（链上是否存在 φ=⊥），不是忠实（LC）：', space_after=2)
math('w ∈ C(t)  ⟺  链 w→t 无矛盾（所有 φ 有值，确定性传播）；')
math('w ∈ R(t)  ⟺  链 w→t 存在矛盾（某处 φ=⊥，非确定性）。')
para('DIR、RSN、EXE、局部偏差都是可归因（局部错是根，照样确定性往下传）；只有 φ=⊥ 的矛盾才判关联。候选集按此切成三块：')
table(
    ['子集', '含义'],
    [
        ['可归因 C(t)', '路径无矛盾，确定性传播（含局部偏差这个根）'],
        ['关联 R(t)', '路径上出现 φ=⊥，证据链断裂'],
        ['未解开 U(t)', '无规则或无法评分，且路径上没有 ⊥'],
    ],
    widths=[5.0, 10.0],
)
para('诊断层不做极小碰集（那是在线诊断引擎从症状反推最小异常组；这里候选集已是超集，切分靠确定性轴）。'
     '完备性由超集保证（可归因 + 关联都收进来，不漏）；精度由可归因子集保证；诚实性由关联子集保证——'
     '「这条链在确定性处断了」本身是结论，不能假装它没断。最小修复集归因果层（3.8）。')

# ---- 3.8 ----
doc.add_heading('3.8  反事实修复核与不可分解度', level=2)

para('目标 t 的反事实修复核：')
math('K*(t) = argmin_{S ⊆ F} |S|   s.t.   y_{−S} ≡ y*。')
para('即：把哪些上游错误改回金标之后，目标 t 才恢复金标，且改的个数最少。搜索上界是目标 t 的全部上游节点。'
     '不可分解度 δ(t) = |K*(t)| 度量「终点错误是单个根因造成的，还是多个根因纠缠造成的」。'
     'K*(t) 与诊断层候选集 F(t) 分开：前者问「改回哪几处金标终点才恢复」，后者问「可能根里哪些还能被确定性解释」。')
para('系统侧反事实（确定性重算：改上游 → φ 重算）与模型侧反事实（重新调用模型：把上游槽位替换成金标再问下游）分开报告。'
     '二者一致时，才能说「系统内的传播因果迁移到了模型的可见行为层」；不一致处（随机性、提示敏感、其他混杂）单独量化。'
     '多个最小核报集合与大小分布，不压缩成单点。')
para('设计审计：', space_after=2)
bullet('latent：目标直接因子中没有独立测点的个数（隐藏错误来源）。', bold_lead='')
bullet('alias：不同因子赋值在某测点产生相同值的对数（值反解不开）。', bold_lead='')
para('latent 与 alias 是「可诊断性」的前置审计，本身不保证可检测/可唯一归因。')

# ---- 3.9 ----
doc.add_heading('3.9  三层评测流程', level=2)

code_block('设计层 ──► 诊断层 ──► 因果层\n'
           '构造系统    归因错误    量化影响')

bullet('设计层：拆子问题、定目标、定参考规则与测点、构造故障实例库、图可达审计（算支持层、分两类模糊组）、审 latent/alias。', bold_lead='')
bullet('诊断层：契约认知重构、解析模型答案、算残差、五类分类、候选集（递归保守归并）、可归因/关联切分。', bold_lead='')
bullet('因果层：反事实修复、算不可分解度、危害度/发生度/传播深度三榜。', bold_lead='')

para('诊断流程（算法形式）：', space_after=2)
code_block('算法 1  过程级诊断\n'
           '输入：模型答案 o，依赖图 G_c，局部规则 φ，契约认知重构器\n'
           '输出：五类标签、候选集、可归因/关联切分、最小修复核\n'
           '1. 契约认知重构：由识别测点回答重构模型认知契约 ĉ\n'
           '2. 逐节点局部重算 õ_v = φ_v(o_Pa(v), x, ĉ)，记录忠实轴与确定性轴读数\n'
           '3. 五类分类：答案对错 × 可归因性（链上是否有 φ=⊥）+ 忠实基线\n'
           '4. 错测点候选集：字段与 φ 一致→只收祖先故障；否则本空注入 ∪ 祖先故障；\n'
           '    祖先须注入点 ⊆ 严格祖先且可达集盖住该测点（超集，不是冲突集）\n'
           '5. 可归因/关联切分：路径 φ 处处有值→可归因；某处 φ=⊥→关联；无规则且无 ⊥→未解开\n'
           '6. 因果层最小修复核 K*(t)（上界 = 终点全部上游）→ 不可分解度、危害度三榜')

para('三层指标（汇总；正式定义与公式归入独立的「评价指标」一节，并只保留核心指标）：')
table(
    ['层', '指标', '回答'],
    [
        ['设计', '可检测性、FIR、模糊组、latent、alias', '数据集能不能诊断'],
        ['诊断', '覆盖率、未认定率、可归因/关联率、传播深度/广度、答案遮蔽率、随机性正确率、不可归因率', '错误能不能被解释、多强'],
        ['因果', '发生度、影响范围、CR、不可分解度、危害度', '哪些错误危害最大'],
    ],
    widths=[2.5, 8.0, 4.5],
)
para('危害度、传播深度、发生度分三榜报告，分歧本身是结论——一个「路径最长」的错误可能危害度很低，'
     '一个「每次出现」的错误可能因到达终点不可分解而无法加分，这本身就是关于错误结构的发现。')

# ---- 方法小结 ----
doc.add_heading('方法小结（核心主张）', level=2)
para('本方法把工程契约从隐含背景提升为数据生成、参考执行、模型推理共同使用的显式输入；通过契约与骨架分离，'
     '确定性程序在多种契约组合下自动生成过程金标；通过契约认知重构，在模型认知契约下做局部重算，区分「识别错」与「执行错」；'
     '通过二值 D 矩阵做可诊断性审计，通过递归保守归并与确定性判定（路径上是否出现 φ=⊥）把归因分为可归因（确定性传播）与关联（矛盾）；'
     '通过反事实核与三榜量化错误影响。方法核心是把「契约遵循」从隐式目标变为显式、可观测、可归因的评测对象。')

# ================= 附录 =================
doc.add_heading('附录：关键术语中英对照（便于翻译）', level=1)
table(
    ['中文', 'English'],
    [
        ['工程契约', 'engineering contract'],
        ['逻辑骨架', 'logical skeleton / computation DAG'],
        ['测点', 'measurement point / probe'],
        ['契约识别测点 / 执行测点 / 数理测点', 'recognition / execution / algebra probe'],
        ['金标轨迹', 'gold / reference trace'],
        ['参考执行程序', 'reference executor / deterministic solver'],
        ['局部规则', 'local rule'],
        ['认知契约', 'cognized contract'],
        ['契约认知重构', 'contract cognition reconstruction'],
        ['直接识别 / 推理识别', 'DIR (direct) / RSN (reasoned) recognition'],
        ['执行 / 数理 / 残差', 'EXE / ALG / RES'],
        ['局部一致性', 'local consistency'],
        ['确定性掩盖', 'deterministic masking'],
        ['忠实传播', 'faithful propagation'],
        ['局部错', 'local error'],
        ['故障签名（二值支持列）', 'fault signature (binary support column)'],
        ['模糊组', 'ambiguity group'],
        ['可达集 / 支持层', 'reachable set / support layer'],
        ['图可达 D 矩阵', 'graph-reachability D matrix'],
        ['设计层审计', 'design-layer audit'],
        ['可检测性 / 可隔离性', 'detectability / isolability'],
        ['候选集 / 递归保守归并', 'candidate set / recursive conservative merging'],
        ['可归因 / 关联', 'attributable / associational'],
        ['可追溯答案遮蔽', 'traceable answer masking'],
        ['随机性正确', 'random / undiagnosed correct'],
        ['反事实修复核', 'counterfactual repair kernel'],
        ['不可分解度', 'indecomposability'],
        ['发生度 / 危害度 / 传播深度', 'occurrence / harm / propagation depth'],
    ],
    widths=[6.0, 9.0],
)

doc.save(OUT)
print('saved:', OUT)

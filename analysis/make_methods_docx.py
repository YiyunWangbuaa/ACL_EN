# -*- coding: utf-8 -*-
"""Generate the ACL Methods chapter as a Word document."""
from docx import Document
from docx.shared import Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

OUT = r"C:\Users\minec\Desktop\ACL_methods_chapter_draft.docx"

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
set_font(normal, 'Times New Roman', 'Times New Roman', 11)
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.25

set_font(doc.styles['Heading 1'], 'Arial', 'Arial', 15, bold=True, color=RGBColor(0, 0, 0))
set_font(doc.styles['Heading 2'], 'Arial', 'Arial', 12.5, bold=True, color=RGBColor(0, 0, 0))
set_font(doc.styles['Heading 3'], 'Arial', 'Arial', 11.5, bold=True, color=RGBColor(0x33, 0x33, 0x33))

for sec in doc.sections:
    sec.top_margin = Cm(2.2)
    sec.bottom_margin = Cm(2.2)
    sec.left_margin = Cm(2.5)
    sec.right_margin = Cm(2.5)

# ---------- helpers ----------
def run_font(run, latin='Times New Roman', east='Times New Roman', size=None, bold=None, italic=None, color=None):
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
    run_font(run, latin='Cambria Math', east='Cambria Math', size=11.5)
    return par

def code_block(text):
    par = doc.add_paragraph()
    par.paragraph_format.space_before = Pt(4)
    par.paragraph_format.space_after = Pt(6)
    for i, line in enumerate(text.split('\n')):
        if i > 0:
            par.add_run().add_break()
        run = par.add_run(line)
        run_font(run, latin='Consolas', east='Consolas', size=9.5)
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
tp = para('ACL Methods chapter: core design (draft)', align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
for run in tp.runs:
    run_font(run, latin='Arial', east='Arial', size=18, bold=True)
para('(Written for LLM reviewers. English draft of the methods core.)', align=WD_ALIGN_PARAGRAPH.CENTER, space_after=12)

# ================= editor's note =================
doc.add_heading('Editor note (delete this page before submission)', level=2)
bullet('Language: English. The term list at the end is the working glossary.')
bullet('Placement: this file is drafted as paper section 3, Methods. Subsections 3.1-3.9 can be dropped in as they stand. If Methods is section 4, shift the numbers.')
bullet('Rewrite for LLM reviewers: engineering phrases such as fault diagnosis, signature matrix, and minimal hitting set are stated in the language of process supervision, step-level verifiers, attribution, and counterfactuals. Each formal definition says which evaluation question it answers.')
bullet('Scope: this draft covers only the three core layers (task formalization, deterministic data generation, error diagnosis and causal attribution). It does not include the A-E evaluation protocol, experimental results, or related work.')
bullet('The benchmark name still needs a new proposal. The current name FD-Eval sounds like fault diagnosis. The name should carry the theme of heterogeneous engineering contracts. Supply a name, or authorize a proposal.')
bullet('Already aligned: fault classes E1-E5 are DIR/RSN/EXE/ALG/RES (recognition is one ability at two depths). The intermediate calculator is contract-cognition reconstruction: execute the parents under the contract the model has cognized, c-hat. See 3.3 and 3.6.')

para('Writing decisions still open:', space_after=2)
bullet('The analogy of a diagnosable system model needs a systematic discussion, not one or two sentences. This draft gives the skeleton. Whether it becomes its own subsection or moves to related work is open.')
bullet('The metrics should be a section of their own, written as formulas, and limited to the core metrics. The three-layer table here is longer than that. It is kept as a placeholder.')
bullet('The body uses error diagnosis and attribution rather than fault diagnosis. English terms are marked on first use. Correct any gloss that drifts.')

# ================= 3 Methods =================
doc.add_heading('3  Methods', level=1)

para('This paper proposes a process-level evaluation framework for heterogeneous engineering contracts. Following an engineering contract is the explicit, observable, and attributable object of evaluation. '
     'The difference between engineering reasoning and ordinary arithmetic or proof is that the correct answer is contract-sensitive: it is fixed by the engineering contract declared in the problem '
     '(axis directions, positive senses of angles, transform conventions, rotation order, installation, and sign conventions). The same physical scene and the same computation, '
     'with one contract field changed, yield a systematically different correct answer. The core skill is therefore not only computing correctly. '
     'It is recognizing the contract and following it consistently to the correct answer.')

para('Most process-level evaluations only separate a correct step from an incorrect step (process supervision), or a correct final answer from an incorrect one. They do not answer three finer questions: '
     '(i) whether an error is a failure to read the contract or a failure to execute a contract that was read; (ii) whether a finally correct answer is reliably correct, '
     'correct under traceable masking (the process is wrong and the answer happens to match), or a chance hit under contradiction; (iii) whether an upstream error affects the final target '
     'causally or only associatively. This framework lifts evaluation data from a problem plus a reference answer to a diagnosable system model. '
     'Each task is a deterministic system (dependency graph, local rules, and a contract). The LLM is the black-box component under diagnosis. '
     'We do not assume its internal structure. We observe its output at each measurement point and, from the system model, isolate faults, attribute errors, and trace how errors propagate. '
     'The system-model analogy runs through the paper: isolated question-answer pairs become an instrument that can be diagnosed, and why the model is wrong becomes a computable diagnosis problem rather than a post-hoc guess. '
     'The method has three layers: a design layer (build the system and audit diagnosability), a diagnosis layer (attribute errors), and a causal layer (measure effects).')

# ---- 3.1 ----
doc.add_heading('3.1  Task formalization: contract, skeleton, and target', level=2)

para('An engineering task is the triple:')
math('T = (x, c, G)')

bullet('the physical scene (aircraft model, camera layout, arm configuration). It states what the world is.', bold_lead='Physical scene x: ')
bullet('a finite set of convention and definition fields (axis directions, positive senses of angles, transform conventions, rotation order, installation, and sign conventions). It states which rules the problem requires.', bold_lead='Engineering contract c = (c_1, ..., c_K): ')
bullet('a directed acyclic graph. A node v in V is a step or answer field to be filled. An edge (u -> v) means that the derivation of v uses the output of u.', bold_lead='Logical skeleton G = (V, E): ')

para('Core design: the contract is separated from the skeleton. For one scene and one skeleton, contract fields can be changed without rewriting the reasoning procedure:')
math('T_c = (x, c, G).')

para('The contract is an independent system variable so that the method can reproduce a real engineering setting in which several contracts coexist. For one physical object, different conventions (frames, positive senses, rotation order, installation) '
     'produce systematically different correct answers. Once the contract is factored out, we can generate counter-intuitive contracts (for example, axial positive rearward, a reversed installation, or a nonstandard rotation order) '
     'and test whether the LLM understands and applies the contract, rather than matching a template from the conventions in its training distribution. The contract enters both the model prompt and the reference executor, '
     'so contract following is measurable: if the model falls back to a default convention, that fallback is observed.')

# ---- 3.2 ----
doc.add_heading('3.2  Reasoning graph and measurement points', level=2)

para('The reasoning graph and the measurement points are explicit because contract following is an internalized skill of the LLM. Right or wrong on the target question does not reveal it. It is coupled to computation, semantic understanding, '
     'and task planning, and it is itself a composite skill: first recognize the contract (read it and choose the route it implies), then execute it. '
     'Those skills appear only in implicit intermediate steps, so the graph needs explicit, targeted question nodes that expose the corresponding failures.')

para('Each reference procedure is a contract-conditioned directed graph:')
math('G_c = (V, E, {phi_v}_{v in V})')
para('V is the set of answer nodes. E is a subset of V x V. An edge (u -> v) means that the derivation of v uses the output of u directly. '
     'phi_v is the local rule of node v under contract c: given the parent values and the contract, what the answer at v should be.')

para('Measurement points fall into four classes by skill, in one-to-one correspondence with the fault classes in section 3.4:')
table(
    ['Measurement class', 'Skill exposed', 'Fault class'],
    [
        ['Contract recognition', 'Read the contract (direct recognition and reasoned understanding)', 'DIR / RSN'],
        ['Contract execution', 'Write the recognized contract into the object', 'EXE'],
        ['Algebraic application', 'Compute from quantities already written', 'ALG'],
        ['Target question', 'The skills above, combined', '—'],
    ],
    widths=[4.5, 8.0, 3.5],
)

para('A measurement point is a node. Every node v in V of the logical skeleton is a measurement point. When the dependency graph is built, the target question is already decomposed to the granularity of a value signature: '
     'matrix entries, vector components, an outer sign, a lever arm, and a numerator or denominator are separate nodes. Each node is an answer field that can be scored on its own. '
     'There is no extra layer that projects a node answer onto a measurement point. Field-level nodes separate the error patterns that directly show a contract-execution failure (a flipped sign, the form of one entry), '
     'so a local choice is directly observable.')

para('Scale of the four domains (by measurement type; the experimental section gives the exact counts):')
table(
    ['Domain', 'Nodes', 'Engineering contracts covered'],
    [
        ['Wind-tunnel aerodynamics', '207', 'Coordinate conventions, attitude transforms, force and moment axes'],
        ['Flight mechanics', '131', 'Aerodynamic quantities and load conversion'],
        ['Computer vision', '66 + 165', 'Pose and stereo geometry'],
        ['Robot kinematics', '156', 'Homogeneous transforms and kinematic solution'],
    ],
    widths=[4.5, 3.0, 8.0],
)

# ---- 3.3 ----
doc.add_heading('3.3  Contract-conditioned deterministic data generation', level=2)

para('The data are produced by deterministic programs in three modules. There is no model call and no annotation noise. The modules run in the order contract, then skeleton, then execution. '
     'The contract instantiator expands contract c into the engineering definitions and method requirements in the problem text. The skeleton generator cuts (x, G) into step-by-step questions and records dependencies. '
     'The contract-conditioned executor runs the local rules along the skeleton and produces the gold trace. All three share the same contract object c and the same skeleton G, so the problem text, the dependencies, and the gold trace agree.')
bullet('c becomes the engineering definitions and method requirements in the problem text.', bold_lead='1. Contract instantiator: ')
bullet('(x, G) becomes the step-by-step questions and their dependencies.', bold_lead='2. Skeleton generator: ')
bullet('under c, run the local rules along the skeleton and produce the gold trace.', bold_lead='3. Contract-conditioned executor: ')

para('A local computation rule depends on the contract and the parent values. Gold is generated under the correct contract c:')
math('g_v = phi_v( g_Pa(v), x, c ).')
para('Gold trace and target quantity:')
math('tau* = ( g_{v_1}, ..., g_{v_{|V|}} ),      y* = Target(tau*).')
para('The problem text, the gold trace, and the target are produced together by q = Gen(x, c, G) and tau* = Exec(x, c, G).')

para('Contract combinations. The combination space Theta is a subset of C_1 x ... x C_K. Each theta in Theta produces:', space_after=2)
math('q_theta = Gen(x, theta, G),      tau*_theta = Exec(x, theta, G).')
para('The physical object and the target stay fixed. Only the contract changes, so the instances are structurally comparable and contractually different. The gold trace is not a label obtained by transforming a standard answer. '
     'It is the result of executing the whole skeleton under that contract. Changing the contract and relabeling are different: the first changes the conditions under which the reasoning itself is valid; the second only changes a right/wrong label.')

para('Two computation layers and contract-cognition reconstruction. The deterministic program computes at two levels:', space_after=2)
bullet('Gold parents and the correct contract c are passed to phi_v. The output is a self-consistent reference.', bold_lead='Gold layer: ')
bullet('The parents the model actually wrote, and the contract c-hat the model has cognized, are passed to phi_v. The output is the conditional result under the model prefix.', bold_lead='Local computation layer: ')
para('Contract-cognition reconstruction is the key step. The local layer does not execute the parents under the correct contract. It first reconstructs the contract c-hat that the model actually cognized, from the answers at the recognition points (DIR/RSN) '
     '(for example, the chirality it cognized is inferred from the axis triple it wrote, not read from the gold chirality), and then executes under c-hat:')
math('o-hat_v = phi_v( o_Pa(v), x, c-hat ).')
para('That separates a recognition error (c-hat differs from c: the model faithfully executed the wrong contract) from an execution error (c-hat equals c: the model did not carry out the contract it had recognized). '
     'When the model parents violate a contract constraint (axes not bound as a group, a repeated product code, an illegal order, or an incomplete triple):')
math('phi_v( o_Pa(v), x, c-hat ) = bottom.')
para('A constraint violation is cut off at the local node and is not propagated deeper. After a shallow violation, deeper reasoning is usually contradictory and not self-consistent. A deterministic system cannot build a self-consistent global explanation of a contradictory prefix, '
     'so it can only treat the deeper nodes as affected by upstream propagation. This local cutoff is a deliberate design choice.')

# ---- 3.4 ----
doc.add_heading('3.4  Fault model: injectable, checkable error hypotheses', level=2)

para('To make attribution a computable claim rather than a judgment, the model error space is a set of injectable, checkable error hypotheses. '
     'Recognition is one ability at two depths. Execution and algebra are later abilities. Class names are the English keyword abbreviations; the E codes are not used:')
table(
    ['Class', 'Ability', 'Meaning'],
    [
        ['DIR', 'Recognition, direct', 'The contract states the mapping. Reading it is enough (Direct).'],
        ['RSN', 'Recognition, reasoned', 'The contract only implies the route. It must be understood (Reason).'],
        ['EXE', 'Execution', 'After recognition, carry it out (Execute).'],
        ['ALG', 'Algebra', 'A computation error while carrying it out (Algebra).'],
        ['RES', 'Residual', 'Not modeled, and not injected (Residual).'],
    ],
    widths=[2.2, 4.0, 8.5],
)

para('Contract classes are recognition (DIR + RSN) plus execution (EXE). DIR tests whether the contract is read. RSN tests whether the route implied by the contract is chosen. EXE tests whether that route is then carried out. '
     'ALG does not depend on the contract. It reads intermediate quantities that are already written, so an upstream contract error does not affect it. That split separates a contract error from an algebra error in the mechanism.')

para('Fault-instance library:')
table(
    ['Class', 'Key', 'One row means'],
    [
        ['DIR', 'Each recognition contract that states a mapping', 'One direct recognition error'],
        ['RSN', 'Each route that must be reasoned (at least 2 blanks)', 'One reasoned recognition error'],
        ['EXE', 'Each derived parent blank', 'One parent blank written wrong (the witness is the wrong text accepted by the most calculators)'],
        ['ALG', 'Each algebra blank', 'One algebra error on a matrix, vector, or expression (one slip is the witness)'],
    ],
    widths=[2.2, 6.5, 6.5],
)

para('Scope: a fault is located at the blank whose answer is wrong, not at a child that read a wrong value. Reading a child is propagation. It is handled by the recursive union and is not a separate row.')
para('Each fault instance f carries, on the dependency graph, a direct-write set Direct(f) (the blanks or measurement points it rewrites) and a reachable set Reach(f) '
     '(the descendant closure of Direct(f), the measurement points it may affect). Both are inputs to the graph-reachability audit in section 3.5. Direct says where the root is. Reach says how far the effect can travel.')

# ---- 3.5 ----
doc.add_heading('3.5  Diagnosability audit: binary D matrix (graph reachability)', level=2)

para('The D matrix uses the binary support-layer semantics of a standard fault signature matrix (FSM). D_{f,m} is in {0,1}, and D_{f,m}=1 if and only if fault f can affect measurement point m. '
     'It is a support relation of whether the point can change, not a prediction of the value it changes to. This method computes that support layer directly from the dependency graph of the deterministic system, with no injection and no recomputation:')
math('D_{f,m} = 1[ m in Reach(f) ].')
para('The construction is transferred from structural diagnosability analysis in model-based fault diagnosis. In a deterministic system with an explicit dependency graph, whether a fault can affect a point is a pure graph property, '
     'so one graph traversal yields every fault-to-point support relation and avoids the cost of injecting and recomputing each instance.')
para('The D matrix therefore sits in the data-review (design) stage. It is a tool that helps the designer place measurement points. It answers whether the current points can detect and isolate each error class, '
     'and which nodes still need a point. The D matrix carries only the binary support layer. It carries no predicted value. The values used in diagnosis come from recomputing phi_v on the model\'s actual wrong values (section 3.6). The two are different and are not mixed.')

para('Ambiguity groups are merged by the binary support column and read as two types:')
table(
    ['Type', 'Pattern', 'Handling'],
    [
        ['Label repeat', 'Members inject the same blank or contract and differ only in the DIR/RSN/EXE label', 'Merge (one blank, one root; do not open a new class)'],
        ['True ambiguity', 'Members inject different blanks but share one reachable set (the same support column)', 'Add a measurement point and split them'],
    ],
    widths=[3.2, 7.5, 5.0],
)

para('After label repeats are merged, the real ambiguity groups remain: different blanks, the same reach. Only those groups need an extra measurement point. Detectability and isolability are both decided by the binary support column: '
     'f is detectable if and only if some m has D_{f,m}=1; f_i and f_j are isolable if and only if some m has D_{f_i,m} different from D_{f_j,m} (different support columns).')
para('(LLM view: this is a diagnosability precheck of the evaluation data. Before any model is called, the graph confirms that the measurement points can expose and separate the error types we care about. It is a static audit of the point design before the data are released.)')

# ---- 3.6 ----
doc.add_heading('3.6  Process-level diagnosis: local recomputation and five classes', level=2)

para('Conditional local recomputation is o-hat_v = phi_v(o_Pa(v), x, c-hat), recomputed under the cognized contract c-hat, which is reconstructed from the recognition-point answers (section 3.3). Local consistency is:')
math('LC(v) = 1[ o_v equiv_v o-hat_v ].')
para('Two axes stay separate. The faithful axis (o_v = o-hat_v versus o_v different from o-hat_v) is the reading (faithful versus local deviation). '
     'The deterministic axis (phi_v has a value versus bottom) is attributability (causal versus associative). Local consistency LC lies only on the faithful axis. '
     'The causal/associative split does not depend on LC. It depends on the deterministic axis.')

para('The five classes are answer correctness times attributability (determinism), plus the faithful baseline:')
table(
    ['Answer', 'Attributability (determinism)', 'Class'],
    [
        ['Correct', 'Faithful (parents all correct)', 'consistent'],
        ['Correct', 'Attributable (phi has a value; the process is wrong)', 'traceable_masking'],
        ['Correct', 'Not attributable (phi = bottom, a contradiction)', 'chance_correct (undiagnosed)'],
        ['Wrong', 'Attributable (phi has a value)', 'attributable'],
        ['Wrong', 'Not attributable (phi = bottom)', 'unattributable'],
    ],
    widths=[2.5, 7.5, 5.5],
)

para('Attributability is the deterministic axis (phi has a value versus bottom) and is decoupled from the faithful axis. A local execution deviation (EXE) is a causal root: even when o_v differs from o-hat_v (LC=0), '
     'if phi_v has a value the error still propagates deterministically along the rule. DIR, RSN, EXE, and local deviation are therefore causal. traceable_masking merges the former deterministic masking '
     '(o = phi = g, with a wrong parent) and unfaithful correctness (o = g and o different from phi). Both have a correct answer, a wrong process, and a system that has not fallen into chance, and both trace to a concrete fault.')
para('(LLM view: this is a stricter reading of step correctness in process supervision. It goes further by crossing final-answer correctness with whether the process has a deterministic explanation, '
     'which yields five classes that cover every case and do not overlap. Correct is no longer one bin. It splits into faithful correctness, traceable masking, and a chance hit.)')

# ---- 3.7 ----
doc.add_heading('3.7  Attribution: candidate set and the attributable/associative split', level=2)

para('Recursive conservative merging. Attribution is not cut off because an intermediate node is correct. For each wrong measurement point m at node v, compare that field with phi_v. '
     'If they agree, collect only faults in the ancestors. If they disagree, there is no rule, or phi_v = bottom, collect the injection at this blank together with the ancestor faults. An ancestor fault must be injected at a strict ancestor of v, '
     'and its binary reachable set must cover the measurement point. The result is a candidate set, a superset, not a conflict set:')
math('F(t) = { local error at t } union { w : w in Anc(t), o_w different from g_w }.')
para('An intermediate correct node cannot tell a recovered causal chain from traceable masking, so recovery is not assumed. Every wrong ancestor is merged recursively (completeness is preferred over precision). '
     'This is different from stopping at the first wrong step, which misses an upstream error that was masked.')

para('Attributable/associative split. On the path from a candidate root w to the target t, the test is determinism (whether some phi on the chain is bottom), not faithfulness (LC):', space_after=2)
math('w in C(t)  iff  the chain w->t has no contradiction (every phi has a value; deterministic propagation);')
math('w in R(t)  iff  the chain w->t has a contradiction (some phi = bottom; not deterministic).')
para('DIR, RSN, EXE, and local deviation are attributable (a local error is a root and still propagates deterministically). Only a contradiction with phi = bottom is associative. The candidate set splits into three blocks:')
table(
    ['Subset', 'Meaning'],
    [
        ['Attributable C(t)', 'The path has no contradiction and propagates deterministically (including a local deviation as the root)'],
        ['Associative R(t)', 'Some phi = bottom on the path, so the evidence chain breaks'],
        ['Unresolved U(t)', 'No rule, or the point cannot be scored, and the path has no bottom'],
    ],
    widths=[4.5, 11.0],
)

para('The diagnosis layer does not compute a minimal hitting set. That is what an online diagnosis engine does when it infers a smallest abnormal group from symptoms. Here the candidate set is already a superset, and the split uses the deterministic axis. '
     'Completeness is the superset (attributable and associative are both kept, so nothing is dropped). Precision is the attributable subset. Honesty is the associative subset: '
     'the chain broke where determinism ended, and that break is itself a conclusion. It is not treated as if the chain were intact. The minimal repair set belongs to the causal layer (3.8).')

# ---- 3.8 ----
doc.add_heading('3.8  Counterfactual repair kernel and indecomposability', level=2)

para('The counterfactual repair kernel of target t is:')
math('K*(t) = argmin_{S subset of F} |S|   s.t.   y_{-S} equiv y*.')
para('That is: which upstream errors, restored to gold, make target t recover gold, with as few restorations as possible. The search ceiling is every upstream node of target t. '
     'The indecomposability delta(t) = |K*(t)| measures whether the terminal error comes from one root or from several roots tangled together. '
     'K*(t) is separate from the diagnosis-layer candidate set F(t). The kernel asks which gold restorations make the terminal recover. The candidate set asks which possible roots still have a deterministic explanation.')
para('A system-side counterfactual (deterministic recomputation: change an upstream node, then recompute phi) is reported separately from a model-side counterfactual (call the model again: replace upstream slots with gold and ask the downstream question). '
     'Only when the two agree can we say that propagation inside the system has moved to the causal layer of the model\'s observable behavior. Disagreements (randomness, prompt sensitivity, and other confounders) are quantified on their own. '
     'Several minimal kernels are reported as a set and as a size distribution. They are not collapsed to one point.')
para('Design audit:', space_after=2)
bullet('the number of direct factors of the target that have no measurement point of their own (hidden error sources).', bold_lead='latent: ')
bullet('the number of pairs of distinct factor assignments that produce the same value at a measurement point (the value cannot be inverted).', bold_lead='alias: ')
para('latent and alias are a pre-audit of diagnosability. They do not by themselves guarantee detectability or unique attribution.')

# ---- 3.9 ----
doc.add_heading('3.9  Three-layer evaluation procedure', level=2)

code_block('Design layer  -->  Diagnosis layer  -->  Causal layer\n'
           'Build the system   Attribute errors    Measure effects')

bullet('Decompose subquestions, fix the target, fix the reference rules and measurement points, build the fault-instance library, audit graph reachability (compute the support layer and separate the two ambiguity groups), and review latent and alias.', bold_lead='Design layer: ')
bullet('Reconstruct contract cognition, parse the model answers, compute residuals, assign the five classes, build the candidate set by recursive conservative merging, and split attributable from associative.', bold_lead='Diagnosis layer: ')
bullet('Counterfactual repair, indecomposability, and the three rankings of harm, occurrence, and propagation depth.', bold_lead='Causal layer: ')

para('Diagnosis procedure, in algorithmic form:', space_after=2)
code_block('Algorithm 1  Process-level diagnosis\n'
           'Input: model answers o, dependency graph G_c, local rules phi, contract-cognition reconstructor\n'
           'Output: five-class labels, candidate set, attributable/associative split, minimal repair kernel\n'
           '1. Contract-cognition reconstruction: rebuild the cognized contract c-hat from the recognition-point answers\n'
           '2. Local recomputation at each node, o-hat_v = phi_v(o_Pa(v), x, c-hat); record the faithful-axis and deterministic-axis readings\n'
           '3. Five classes: answer correctness x attributability (whether some phi on the chain is bottom) + faithful baseline\n'
           '4. Candidate set of a wrong point: if the field agrees with phi, keep ancestor faults only; otherwise this blank\'s injection union ancestor faults;\n'
           '    an ancestor must be injected in the strict ancestors and its reachable set must cover the point (a superset, not a conflict set)\n'
           '5. Attributable/associative split: every phi on the path has a value -> attributable; some phi = bottom -> associative; no rule and no bottom -> unresolved\n'
           '6. Causal-layer minimal repair kernel K*(t) (ceiling = every upstream node of the terminal) -> indecomposability and the three harm rankings')

para('Three-layer metrics (summary; the formal definitions and formulas belong in a separate metrics section and should keep only the core metrics):')
table(
    ['Layer', 'Metrics', 'Question answered'],
    [
        ['Design', 'Detectability, FIR, ambiguity groups, latent, alias', 'Can the dataset be diagnosed?'],
        ['Diagnosis', 'Coverage, uncertified rate, attributable/associative rate, propagation depth and breadth, answer-masking rate, chance_correct rate, unattributable rate', 'Can the error be explained, and how strongly?'],
        ['Causal', 'Occurrence, scope of effect, CR, indecomposability, harm', 'Which errors do the most harm?'],
    ],
    widths=[2.5, 8.0, 4.5],
)
para('Harm, propagation depth, and occurrence are reported as three rankings. Disagreement among them is a conclusion. An error with the longest path may have low harm, '
     'and an error that appears every time may receive no extra score because the terminal it reaches is indecomposable. That disagreement is itself a finding about the error structure.')

# ---- summary ----
doc.add_heading('Method summary (core claim)', level=2)
para('The method lifts an engineering contract from implicit background to an explicit input shared by data generation, reference execution, and model reasoning. Separating the contract from the skeleton lets a deterministic program '
     'generate process gold under many contract combinations. Contract-cognition reconstruction recomputes locally under the contract the model has cognized, and separates a recognition error from an execution error. '
     'The binary D matrix audits diagnosability. Recursive conservative merging and the determinism test (whether some phi on the path is bottom) split attribution into attributable (deterministic propagation) and associative (contradiction). '
     'Counterfactual kernels and the three rankings measure the effect of an error. The core of the method is to make contract following an explicit, observable, and attributable object of evaluation rather than an implicit goal.')

# ================= appendix =================
doc.add_heading('Appendix: key terms', level=1)
table(
    ['Term', 'Meaning in this paper'],
    [
        ['Engineering contract', 'The declared conventions that fix the correct answer'],
        ['Logical skeleton', 'The computation DAG of steps and answer fields'],
        ['Measurement point / probe', 'A node observed for diagnosis'],
        ['Recognition / execution / algebra probe', 'DIR/RSN, EXE, and ALG measurement points'],
        ['Gold / reference trace', 'The trace produced by executing the skeleton under the contract'],
        ['Reference executor', 'The deterministic solver that produces gold'],
        ['Local rule', 'phi_v: parents and contract determine the answer at v'],
        ['Cognized contract', 'The contract reconstructed from the model\'s recognition answers'],
        ['Contract-cognition reconstruction', 'Execute parents under the cognized contract'],
        ['DIR / RSN', 'Direct recognition and reasoned recognition'],
        ['EXE / ALG / RES', 'Execution, algebra, and residual'],
        ['Local consistency', 'Whether the written answer matches the local recomputation'],
        ['Deterministic masking', 'A correct answer whose process is still deterministically wrong'],
        ['Faithful propagation', 'The node matches the value implied by its parents'],
        ['Local error', 'An error introduced at the node itself'],
        ['Fault signature', 'The binary support column of a fault'],
        ['Ambiguity group', 'Faults that share one support column'],
        ['Reachable set / support layer', 'Where a fault can affect a measurement point'],
        ['Graph-reachability D matrix', 'The binary support matrix computed from the graph'],
        ['Design-layer audit', 'Diagnosability checked before any model call'],
        ['Detectability / isolability', 'Whether a fault shows, and whether two faults can be told apart'],
        ['Candidate set', 'The superset from recursive conservative merging'],
        ['attributable / associative', 'Deterministic propagation versus a broken chain'],
        ['traceable_masking', 'A correct answer with a traceable wrong process'],
        ['chance_correct', 'A correct answer that is not diagnosed'],
        ['Counterfactual repair kernel', 'The smallest upstream set whose restoration recovers the target'],
        ['Indecomposability', 'The size of that smallest set'],
        ['Occurrence / harm / propagation depth', 'The three rankings of an error\'s effect'],
    ],
    widths=[6.0, 9.0],
)

doc.save(OUT)
print('saved:', OUT)

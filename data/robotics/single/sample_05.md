# Data 05 tool pitch reversed

Single-definition perturbation: tool pitch

This is the kinematics review manuscript for robotics. The task is attitude control of an aircraft-shaped tool carried by the robot, and a boundary-reachability check of designated points. A three-degree-of-freedom spherical wrist is fixed to the robot base, and a mount rigidly joins the aircraft-shaped tool to the wrist. From the task direction in the base frame, the robot computes the tool attitude and the three task angles, then the positions of the tool points, and decides whether the nose, the right tip, and the left tip can reach the cell boundary. The decision does not cover the whole tool. The aircraft shape is only the geometry of the end tool.

This file contains two independent samples. The forward sample goes from the joint angles to the tool attitude and the three task angles, then to the tool points, the translation, and the boundary reachability of the designated points. The inverse sample is injected alone. It gives only the target task angles and the zero-pose installation. It does not give the joint angles or the forward calculation. The declaration and the text of the two samples are not visible to each other.

Review notes, labels, answers, the cross-layer table, and the generation notes are not sent to the model. The output-format rules have no gold answer and are sent as a shared prompt. The model also receives the declaration of the current sample, the background, the current subprocess name, and the text from the start of that subprocess through the period of the current sentence. Blanks before the current sentence are filled with the answer text, not with the letter. The blank in the current sentence stays open. Each sentence has one open blank. Text after the period is not sent. A blank does not contain the label or the answer. The label is used only for scoring.

In the review manuscript a blank is written {ID:number, options:A: first; B: second, format:format name, label:label, answer:gold}. When there are no options, write none. The model does not see the label or the answer. The model writes the blanks of the current sentence as one JSON object. The key is the number as a string and the value is a string. A choice value is a letter, for example {"1":"B"}. A matrix, a vector, task angles, a formula, joint angles, and an expression are strings. Do not output the original sentence and do not explain.

Scoring looks at three things separately: whether the JSON and the field format are valid; whether the mathematical expressions are equivalent; and whether the number, sign, vector, or matrix is correct. An equivalent expression is mathematically correct. In this example the tool-roll task angle may be written arcsin(1/2) or atan2(1, sqrt(3)) or as an equivalent inverse trigonometric form. Do not mark a mathematical error only because the inverse trigonometric function is written differently.

The forward sample is numbered 1 through 140 and has five subprocesses. A later subprocess does not see the text of an earlier one. The right-tip leading edge is split into two layers: first the tool-frame coordinate, then the base-frame coordinate. The translation of the tool origin is its own layer: first the nose-to-wrist distance, then the translation. The base-frame layer states the tool coordinate and the translation at its own start and does not read them from the previous blanks. Boundary reachability of the designated points is its own layer. It finds the three reachable intervals of the nose and uses the distances from the nose, the right tip, and the left tip to the wrist origin to decide whether these three points can reach the boundary. That test is not a collision conclusion for the whole tool. The forward sample does not also recover the joint angles from the task angles.

The inverse sample restarts its numbering at 1 and has one subprocess. It is paired with the forward sample, but it must be injected independently. The joint angles in the forward declaration are known quantities of the forward problem. The inverse declaration does not write those joint angles or a numerical D. Do not treat a joint angle that appeared in the forward sample as an independent inverse skill.

## Output format

- choice: multiple choice. Write only the option letter. Example: A
- matrix: one 3×3 matrix, as [[a, b, c], [d, e, f], [g, h, i]]. Put exactly one space after each comma. An entry may be an integer, a fraction, sqrt(3), or cos(q1), sin(q1), cos(q2), sin(q2), cos(q3), sin(q3). Do not leave a space inside parentheses. Write the minus sign before the entry. Do not break the line, do not use LaTeX, and do not write degrees. Example: [[cos(q1), -sin(q1), 0], [sin(q1), cos(q1), 0], [0, 0, 1]]
- vector: three components on one line, as [a, b, c]. Put exactly one space after each comma. The order is X, Y, Z of that frame. A component may contain sqrt, an integer, a fraction, and L, L_mount, x_r, b, Λ, Γ, tan, sin, cos, cos(q2), or sin(q2). Use * for multiplication. Do not break the line. Example: [L + L_mount, 0, 0]
- task-angles: write the tool pitch, the tool lateral angle, and the tool roll, in that order, with a comma and one space between items. arctan, arcsin, atan2, and sqrt are allowed. Do not write an approximation or a degree. Example: arctan(2/3), arcsin(1/5), atan2(-1, sqrt(3))
- formula: write three items in the order q1, q2, q3, with exactly one space after each semicolon. Use only atan2 and sqrt. Write a matrix entry as h11, row index first and column index second. Write a square as ^2. Use * for multiplication. Do not write q1=. Example: atan2(h21, h11); atan2(-h31, sqrt(h11^2 + h21^2)); atan2(h32, h33)
- joint-angles: write [a, b, c] in the order q1, q2, q3, in degrees, without a degree sign. Put exactly one space after each comma. Example: [10, -20, 30]
- distance: one length. Use only L and L_mount, with one space on each side of the plus sign. Do not write a unit. Example: L + L_mount
- interval: a closed interval [min, max]. Put exactly one space after the comma. Write the minus sign before the lower or upper bound. The bounds may contain L, L_mount, sqrt(3), a fraction, and *. Do not write a unit or a degree. Example: [-(L + L_mount), L + L_mount]
- expression: one line. Use * for multiplication. Square brackets denote a column. D*[a, b, c] means matrix D times that column on the left. The name D may be kept. sqrt, ^2, max, and a comparison are allowed. Write > or >= with one space on each side. Do not expand D, do not break the line, and do not use LaTeX. Example: L + L_mount >= L_cell/2

# Forward kinematics

## Declaration

Fixed conventions
- Vectors are columns, and a transform matrix multiplies a column on the left. Components in the parent frame equal the attitude matrix times components in the child frame. Components are written in the X, Y, Z order of that frame.
- Absolute directions and coordinate axes are declared separately. Forward, rearward, up, down, right, left, and axial, normal, and lateral positive are absolute directions, not the positive senses of axes X, Y, and Z.
- Wrist rotations use an intrinsic moving-axis sequence. The three angles follow the absolute-axis order and are written q1, q2, q3. The joint matrices are R(q1), R(q2), and R(q3). The first rotation is written on the left. Rz, Ry, and Rx rotate about coordinate axes Z, Y, and X, with rows and columns in X, Y, Z order. These names do not carry an absolute direction. The positive sense of a joint angle follows the right-hand rule about that absolute direction, and it does not flip when a coordinate axis is reversed. The absolute-axis order is a variable definition. The fixed conventions do not set that order.
- The declaration does not give the entries of Rx, Ry, or Rz. The entries are fixed by where the absolute directions go after the rotation, whether each axis agrees with that absolute positive direction, and the right-hand rule about that direction.
- Joint ranges stay on the absolute axes and do not move to another axis when the order changes. The normal axis runs from -60° to 60°, where the cosine is positive. The lateral and axial axes run from -90° to 90°.
- In the calculation the zero-pose installation is represented only by S. Do not multiply by another installation matrix.
- This example ignores deformation and uses rigid, quasistatic kinematics.
- Tool pitch and the tool lateral angle are computed only from the task-direction components in the tool frame. Tool roll uses the zero reference of the task angles and is computed from tool forward, tool up, and the base reference-up direction. The three angles together are the tool task angles. A joint angle is not a task angle.

Absolute directions
The wrist's forward, up, and right are fixed to the wrist and turn with it. They are called the wrist's physical forward, physical up, and physical right. The base's forward, up, and right are fixed to the robot base. At the initial pose, the wrist's physical forward, up, and right coincide with the base's forward, up, and right. After the joints turn, the wrist's physical directions leave that coincidence. The tool's absolute directions are relative to the tool and are not rewritten by this initial coincidence.
The base, the wrist, and the tool each have three direction axes. The three sets use the same positive senses: axial positive is forward, normal positive is up, and lateral positive is right. The six directions are the two senses of these three axes.
Axial positive, normal positive, lateral positive, the positive tool axis, and the base task direction are the positive senses of these three axes, not the positive senses of coordinate axes X, Y, and Z. The wrist's physical axial, normal, and lateral directions are the same three axes, fixed to the wrist. An axis declaration only assigns each coordinate axis to one of these axes and to one sense. That sense may agree with the positive direction or oppose it.
The three frames share one orientation rule: forward × right = down. Looking along forward, right is the right-hand side. Base down is vertically downward. Up and down of the tool and the wrist are directions fixed to each body and are not called vertically downward. This rule does not change with the definition.
The tool's three axes and six directions, relative to the tool:
- Axial: the axis joining the tool rear to the tool nose. The positive sense is forward, toward the nose. Rearward points toward the rear.
- Normal: the tool's up-down axis. The positive sense is up. Down is the opposite sense.
- Lateral: the tool's left-right axis. The positive sense is right. Left is the opposite sense.
The wrist's three axes and six directions are the wrist's physical directions, relative to the wrist:
- Axial: the wrist's physical fore-aft axis. The positive sense is the wrist's physical forward. Rearward is the opposite sense.
- Normal: the wrist's physical up-down axis. The positive sense is the wrist's physical up. Down is the opposite sense.
- Lateral: the wrist's physical left-right axis. The positive sense is the wrist's physical right. Left is the opposite sense.
The three joint axes of the spherical wrist are not always fixed to one set of end directions. The normal, lateral, and axial names of joint i are the corresponding directions in the moving frame before that joint acts. The wrist frame describes only the final attitude after every joint has turned.
The base's three axes and six directions, fixed to the base:
- Axial: the base task axis. The positive sense is forward, the task direction. Rearward is the opposite sense.
- Normal: the base vertical axis. The positive sense is up, upward is vertically up, and downward is vertically down.
- Lateral: the base's horizontal transverse axis. The positive sense is right. Left is the opposite sense.

Scene
The three axes of the spherical wrist meet at a fixed point O. O is the wrist origin and the fixed reference point of the base frame. The wrist center does not translate. The wrist carries a tool mount, which enters the aircraft-shaped tool from its base, and the tool is fixed rigidly to the mount. As the robot moves, the wrist turns the mount and the tool together. The tool origin is at the tool nose, not at the center of mass and not at the wrist origin. After the rotation, the nose leaves the base reference point, and the position between the two points is computed from the tool attitude. The robot base origin is at the center of the cell cross-section. The cell extends along base X_B and the cross-section is a square. The end mechanism must not cross the cell boundary. The later reachability check inspects only three designated points.

Variable definitions
The following seven items can be rewritten one at a time, and the absolute-direction text does not change with them: the base frame, the wrist frame, the tool frame, the absolute-axis order, the positive sense of the joint angles, the zero-pose installation, and the positive sense of the tool task angles. This item gives the default declaration. The absolute-axis order is a prescribed perturbation. It changes only the intrinsic sequence and does not change the positive sense or the range of any axis.

Base frame (right-handed)
The base frame is a right-handed frame fixed to the robot base. X_B is the task direction, Y_B is reference up, and Z_B is reference lateral.
- X_B: along the task direction, pointing base-forward (as declared).
- Y_B: along the base normal, pointing base-up (as declared).
- Z_B: along the base lateral direction, pointing base-right (inferred from chirality).

Wrist frame (right-handed)
- X_W: along the wrist's physical axis, pointing physically forward (as declared).
- Y_W: along the wrist's physical normal, pointing physically up (as declared).
- Z_W: along the wrist's physical lateral direction, pointing physically right (inferred from chirality).
When every joint angle is 0 and the installation matrix is not yet included, X_W, Y_W, and Z_W have the same sense as X_B, Y_B, and Z_B.

Tool frame (right-handed)
The tool frame is fixed to the aircraft-shaped end tool.
- X_T: along the tool axis, pointing tool-forward (as declared).
- Y_T: along the tool normal, pointing tool-up (as declared).
- Z_T: along the tool lateral direction, pointing tool-right (inferred from chirality).
The tool-frame origin is at the tool nose.

Origin and mount
- The wrist origin coincides with the base reference point. Both are O. O does not move with the joint angles.
- The tool-frame origin is at the tool nose.
- L is the tool's axial length along X_T, measured from the tool nose to the tool base.
- The part of the mount inserted into the tool lies inside the tool axial length.
- L_mount is the installation distance from the tool base to the wrist origin. It does not include the part already inserted into the tool.
- O lies behind the tool base. Along tool forward the order is O, the tool base, the tool nose. The three points are collinear on the tool axis.
- After the wrist turns, the tool origin translates relative to the base reference point. The translation is the position of the tool origin in the base frame and changes with the attitude. The declaration does not give the nose-to-O distance or the finished translation. The process computes them.

Work cell
- The robot base origin is at the center of the cell cross-section.
- The cell extends along base X_B. Its cross-section is a square of side L_cell. The sides of the square are parallel to base up and base right. The vertical and lateral boundaries use the same side length.
- The cell boundary limits position along base Y_B and Z_B. It does not limit X_B.
- Boundary reachability checks only the tool nose, the right tip, and the left tip. The right tip is the right-tip leading edge. The left tip matches the right tip in fore-aft and vertical position and is opposite in the lateral direction.
- The mount, the tool surface, and other parts are not checked. The check sweeps the declared joint range and does not replace that range by one joint triple. The conclusion covers only these three points and is not a collision conclusion for the whole tool.
- Touching the boundary counts as reaching it. The declaration does not give the boundary distance, the nose intervals, or the final test. The process computes them.

Rotation order of the absolute-direction axes
This is a variable definition and a perturbation. It states which absolute-direction axis turns first in the intrinsic sequence. It does not rename axes X, Y, and Z. Each of the three axes appears once.
The order in this example is normal, lateral, axial.
- q1 turns first, about the wrist's absolute normal axis. That axis is positive up.
- q2 turns second, about the wrist's absolute lateral axis. That axis is positive right.
- q3 turns last, about the wrist's absolute axial axis. That axis is positive forward.
In this example q1 uses the normal-axis range, q2 the lateral-axis range, and q3 the axial-axis range.

Positive sense of the joint angles
This is a variable definition. The positive sense follows the right-hand rule about the absolute axis being turned, not a separate rule based on axes X, Y, and Z.
- About the normal axis, the positive sense agrees with the right-hand rule about up.
- About the lateral axis, the positive sense agrees with the right-hand rule about right.
- About the axial axis, the positive sense agrees with the right-hand rule about forward.

Matrices
- R is the attitude of the wrist frame relative to the base frame. It maps wrist components into base components.
- S is the fixed installation of the tool frame in the wrist frame. It maps tool components into wrist components.
- D is the attitude of the tool frame relative to the base frame. It maps tool components into base components.

Zero-pose installation
When every joint angle is 0, tool forward coincides with base +X_B and tool up coincides with base +Z_B. Because the three directions must form a right-handed frame, tool right coincides with base -Y_B.

Task direction
The robot task compares a tool reference direction with base +X_B. The task-direction vector is fixed in the base frame and points along base +X_B. Its components on X_B, Y_B, and Z_B follow the base-frame declaration.

Tool task angles
The tool pitch, lateral, and roll task angles together are called the tool task angles. Their positive senses are a variable definition. Write the task direction in the tool frame as [d_x, d_y, d_z], in the order X_T, Y_T, Z_T.
- Tool-pitch task angle α_T: the signed angle between tool forward and the projection of the task direction onto the tool plane of symmetry. The angle is positive when the task direction points upward. α_T=atan2(d_y, d_x).
- Tool-lateral task angle β_T: the signed angle of the task direction away from the tool plane of symmetry. The angle is positive when the task direction points to the right. β_T=atan2(d_z, sqrt(d_x^2 + d_y^2)).
- Tool-roll task angle γ_T: let t be the unit vector of tool forward, n the unit vector of tool up, and e_Y the unit vector of the base reference-up direction. In this example base up is along Y_B, so e_Y=[0, 1, 0]. u=(e_Y-(e_Y·t)*t)/||e_Y-(e_Y·t)*t||. Let r=t×u. γ_T=atan2(n·r, n·u). The positive sense is the same as the right-hand rule about tool forward, and the opposite sense is negative. The roll task angle is undefined when tool forward is parallel to the base reference-up direction. Changing only the tool-roll task angle rotates about the tool axis.

Tool geometry
The aircraft shape is only the geometry of the end tool. The position reference is the right-tip leading edge, used for attitude checks and boundary contact. The trailing edge, the left-tip leading edge, and the wing-root leading edge are not the reference.
The shape parameters are as follows.
- b: the spacing of the left and right leading edges along Z_T. Those two edges match in fore-aft and vertical position, so the distance between them is also b. The lateral coordinate from the plane of symmetry to one tip is b/2. That lateral coordinate does not change with the dihedral.
- Λ: the sweep of the tool leading edge in the X_T-Z_T plane. The angle is positive when the right-tip leading edge lies behind the wing-root leading edge. The fore-aft offset equals that lateral coordinate times tan(Λ), and then changes sign.
- Γ: the dihedral. It is the ratio of the tip's normal offset to its lateral offset from the wing root, tan(Γ). The angle is positive when the right tip is above the wing-root leading edge. Increasing the dihedral increases the slant length and leaves the lateral spacing equal to b.
- x_r: the coordinate of the wing-root leading edge along the tool axis, measured from the tool nose. It is positive forward. It is written on the tool axis that corresponds to the absolute axial positive direction. The absolute normal and absolute lateral coordinates of the wing-root leading edge are both 0.
The fore-aft position of the right-tip leading edge uses the sweep and the lateral coordinate. Its vertical position uses the dihedral and the lateral coordinate. Fore-aft, lateral, and vertical follow the absolute axial, lateral, and normal directions, and then fall on X_T, Y_T, and Z_T according to the axis declaration.

Joint angles of this example
q1=0°, q2=30°, and q3=-60°. The normal axis is 0°, the lateral axis is 30°, and the axial axis is -60°.

## Background

Axial positive, normal positive, and lateral positive are forward, up, and right. The positive tool axis and the base task direction are this absolute forward, not the positive senses of axes X, Y, and Z. The wrist's physical forward, up, and right are fixed to the wrist. At the initial pose they coincide with base forward, up, and right, and they leave that coincidence after the joints turn. A coordinate axis may agree with the positive direction or oppose it. Forward × right = down, and this relation does not change with the axis declaration. Pitch alone rotates about the absolute lateral axis. The lateral task angle alone rotates about the absolute normal axis. Roll alone rotates about the absolute axial axis. Tool roll is measured from the projection of the base reference-up direction onto the plane perpendicular to tool forward. Its positive sense is the same as the right-hand rule about tool forward, and the opposite sense is negative. Tool roll is undefined when tool forward is parallel to the base reference-up direction. The positive sense of a joint angle follows the right-hand rule about the absolute axis it turns about, and it does not flip when a coordinate axis is reversed. The intrinsic moving-axis order is the absolute-direction order, here normal, lateral, axial. Joint ranges are bound to the absolute axes: the normal axis is -60° to 60°, and the lateral and axial axes are -90° to 90°. The positive senses of the task angles follow the variable definition. At the zero pose, tool forward coincides with base +X_B, tool up with base +Z_B, and tool right with base -Y_B. The wrist turns the aircraft-shaped tool as a rigid body through the mount. This example ignores deformation. The task direction points along base +X_B. The same physical vector has different components in different frames. Vectors are columns, and an attitude matrix multiplies a column on the left, mapping the child frame into the parent frame. The transpose maps the parent frame back into the child frame. Tool pitch is positive when the task direction points upward, and it equals atan2(d_y, d_x). The tool lateral angle is positive when the task direction points to the right, and it equals atan2(d_z, sqrt(d_x^2 + d_y^2)). The zero reference and sign of tool roll follow the task-angle definition. Leading-edge sweep places the tip behind the root. Dihedral makes the tip higher than the root. When the tip spacing is a lateral spacing, dihedral does not shrink the lateral projection, and the slant length grows with the dihedral. The wrist origin coincides with the base reference point. The tool origin is at the tool nose. O is behind the tool base, and the order along forward is O, base, nose. The nose-to-wrist distance joins the tool axial length to the mount length from the base to the wrist. The inserted part of the mount is already inside the axial length. After the attitude changes, the tool origin in the base frame is the attitude matrix times this axial offset, and the two origins no longer coincide. The cell extends along base X_B and its cross-section is a square. Its boundary limits the vertical and lateral directions and does not limit the axial direction. A checked point reaches the boundary when it touches the boundary. This is not a collision conclusion for the whole tool.

## Full process


Every slot below carries a label and the gold answer so the manuscript can be reviewed. Those fields are not what the model sees in the blank.

### Tool attitude and task angles

The base frame, the wrist frame, and the tool frame are right-handed.
Each frame first declares the axial and normal axes. The lateral axis is inferred from chirality.
1. The absolute normal direction (up) of the base frame corresponds to {ID:1, options:A: X_B; B: Y_B; C: Z_B; D: X_W; E: Y_W; F: Z_W; G: X_T; H: Y_T; I: Z_T; J: -X_B; K: -Y_B; L: -Z_B; M: -X_W; N: -Y_W; O: -Z_W; P: -X_T; Q: -Y_T; R: -Z_T, format:choice, label:C1, answer:B}.
2. The absolute axial direction (forward) of the base frame corresponds to {ID:2, options:A: X_B; B: Y_B; C: Z_B; D: X_W; E: Y_W; F: Z_W; G: X_T; H: Y_T; I: Z_T; J: -X_B; K: -Y_B; L: -Z_B; M: -X_W; N: -Y_W; O: -Z_W; P: -X_T; Q: -Y_T; R: -Z_T, format:choice, label:C1, answer:A}.
3. The absolute lateral direction (right) of the base frame corresponds to {ID:3, options:A: X_B; B: Y_B; C: Z_B; D: X_W; E: Y_W; F: Z_W; G: X_T; H: Y_T; I: Z_T; J: -X_B; K: -Y_B; L: -Z_B; M: -X_W; N: -Y_W; O: -Z_W; P: -X_T; Q: -Y_T; R: -Z_T, format:choice, label:C1, answer:C}.
4. Physical up of the wrist frame corresponds to {ID:4, options:A: X_B; B: Y_B; C: Z_B; D: X_W; E: Y_W; F: Z_W; G: X_T; H: Y_T; I: Z_T; J: -X_B; K: -Y_B; L: -Z_B; M: -X_W; N: -Y_W; O: -Z_W; P: -X_T; Q: -Y_T; R: -Z_T, format:choice, label:C1, answer:E}.
5. Physical forward of the wrist frame corresponds to {ID:5, options:A: X_B; B: Y_B; C: Z_B; D: X_W; E: Y_W; F: Z_W; G: X_T; H: Y_T; I: Z_T; J: -X_B; K: -Y_B; L: -Z_B; M: -X_W; N: -Y_W; O: -Z_W; P: -X_T; Q: -Y_T; R: -Z_T, format:choice, label:C1, answer:D}.
6. Physical right of the wrist frame corresponds to {ID:6, options:A: X_B; B: Y_B; C: Z_B; D: X_W; E: Y_W; F: Z_W; G: X_T; H: Y_T; I: Z_T; J: -X_B; K: -Y_B; L: -Z_B; M: -X_W; N: -Y_W; O: -Z_W; P: -X_T; Q: -Y_T; R: -Z_T, format:choice, label:C1, answer:F}.
7. The absolute normal direction (up) of the tool frame corresponds to {ID:7, options:A: X_B; B: Y_B; C: Z_B; D: X_W; E: Y_W; F: Z_W; G: X_T; H: Y_T; I: Z_T; J: -X_B; K: -Y_B; L: -Z_B; M: -X_W; N: -Y_W; O: -Z_W; P: -X_T; Q: -Y_T; R: -Z_T, format:choice, label:C1, answer:H}.
8. The absolute axial direction (forward) of the tool frame corresponds to {ID:8, options:A: X_B; B: Y_B; C: Z_B; D: X_W; E: Y_W; F: Z_W; G: X_T; H: Y_T; I: Z_T; J: -X_B; K: -Y_B; L: -Z_B; M: -X_W; N: -Y_W; O: -Z_W; P: -X_T; Q: -Y_T; R: -Z_T, format:choice, label:C1, answer:G}.
9. The absolute lateral direction (right) of the tool frame corresponds to {ID:9, options:A: X_B; B: Y_B; C: Z_B; D: X_W; E: Y_W; F: Z_W; G: X_T; H: Y_T; I: Z_T; J: -X_B; K: -Y_B; L: -Z_B; M: -X_W; N: -Y_W; O: -Z_W; P: -X_T; Q: -Y_T; R: -Z_T, format:choice, label:C1, answer:I}.
Where each joint sends a direction is measured in the moving frame before that joint acts, not in the base frame after the rotation.
10. The first absolute-direction axis in this rotation order is {ID:10, options:A: axial; B: normal; C: lateral, format:choice, label:C1, answer:B}.
11. The current wrist axis corresponding to the first absolute-direction axis is {ID:11, options:A: X_W; B: Y_W; C: Z_W, format:choice, label:C2, answer:B}.
12. The positive sense of q1 and the right-hand rule about this absolute direction are {ID:12, options:A: same; B: opposite, format:choice, label:C1, answer:A}.
13. In the moving frame before this joint acts, when q1 is positive the wrist's physical forward direction turns {ID:13, options:A: upward; B: downward; C: left; D: right, format:choice, label:A1, answer:C}.
14. Rows and columns of R(q1) follow the wrist axes X, Y, Z. The entries follow this joint axis, its positive sense, and where the directions go. The declaration does not give the finished matrix. R(q1)= {ID:14, options:none, format:matrix, label:A2, answer:[[cos(q1), 0, sin(q1)], [0, 1, 0], [-sin(q1), 0, cos(q1)]]}.
15. The second absolute-direction axis in this rotation order is {ID:15, options:A: axial; B: normal; C: lateral, format:choice, label:C1, answer:C}.
16. The current wrist axis corresponding to the second absolute-direction axis is {ID:16, options:A: X_W; B: Y_W; C: Z_W, format:choice, label:C2, answer:C}.
17. The positive sense of q2 and the right-hand rule about this absolute direction are {ID:17, options:A: same; B: opposite, format:choice, label:C1, answer:A}.
18. In the moving frame before this joint acts, when q2 is positive the wrist's physical forward direction turns {ID:18, options:A: upward; B: downward; C: left; D: right, format:choice, label:A1, answer:A}.
19. Rows and columns of R(q2) follow the wrist axes X, Y, Z. The entries follow this joint axis, its positive sense, and where the directions go. The declaration does not give the finished matrix. R(q2)= {ID:19, options:none, format:matrix, label:A2, answer:[[cos(q2), -sin(q2), 0], [sin(q2), cos(q2), 0], [0, 0, 1]]}.
20. The third absolute-direction axis in this rotation order is {ID:20, options:A: axial; B: normal; C: lateral, format:choice, label:C1, answer:A}.
21. The current wrist axis corresponding to the third absolute-direction axis is {ID:21, options:A: X_W; B: Y_W; C: Z_W, format:choice, label:C2, answer:A}.
22. The positive sense of q3 and the right-hand rule about this absolute direction are {ID:22, options:A: same; B: opposite, format:choice, label:C1, answer:A}.
23. In the moving frame before this joint acts, when q3 is positive the wrist's physical up direction turns {ID:23, options:A: upward; B: downward; C: left; D: right, format:choice, label:A1, answer:D}.
24. Rows and columns of R(q3) follow the wrist axes X, Y, Z. The entries follow this joint axis, its positive sense, and where the directions go. The declaration does not give the finished matrix. R(q3)= {ID:24, options:none, format:matrix, label:A2, answer:[[1, 0, 0], [0, cos(q3), -sin(q3)], [0, sin(q3), cos(q3)]]}.
25. At the zero pose, tool forward points {ID:25, options:A: task forward direction; B: base up; C: base right; D: base down, format:choice, label:C1, answer:A}.
26. At the zero pose, tool up coincides with the base absolute direction {ID:26, options:A: upward; B: downward; C: right; D: left, format:choice, label:C1, answer:C}.
At the zero pose the wrist's physical directions coincide with the base directions.
27. Tool up therefore coincides with the wrist's physical direction {ID:27, options:A: forward; B: upward; C: right; D: downward, format:choice, label:C2, answer:C}.
Tool forward coincides with base forward.
Tool up coincides with base right.
28. By tool forward × tool up = tool right, tool right points along the base absolute direction {ID:28, options:A: upward; B: downward; C: left; D: right, format:choice, label:A1, answer:B}.
29. Tool right therefore coincides with the wrist's physical direction {ID:29, options:A: forward; B: upward; C: right; D: downward, format:choice, label:C2, answer:D}.
The columns of S follow X_T, Y_T, Z_T, and the rows follow X_W, Y_W, Z_W.
30. The tool forward axis is this column of the wrist coordinates {ID:30, options:A: [1, 0, 0]; B: [0, 1, 0]; C: [0, 0, 1]; D: [0, -1, 0], format:choice, label:A2, answer:A}.
31. The tool up axis is this column of the wrist coordinates {ID:31, options:A: [1, 0, 0]; B: [0, 0, 1]; C: [0, 1, 0]; D: [0, -1, 0], format:choice, label:A2, answer:B}.
32. The tool right axis is this column of the wrist coordinates {ID:32, options:A: [0, 1, 0]; B: [0, -1, 0]; C: [0, 0, 1]; D: [1, 0, 0], format:choice, label:A2, answer:B}.
33. The installation matrix S is therefore {ID:33, options:none, format:matrix, label:A2, answer:[[1, 0, 0], [0, 0, -1], [0, 1, 0]]}.
34. From the entries of S, it equals a rotation about wrist axis {ID:34, options:A: X_W; B: Y_W; C: Z_W, format:choice, label:A2, answer:A}.
35. That angle is {ID:35, options:A: 0°; B: 90°; C: -90°; D: 180°, format:choice, label:A2, answer:B}.
36. The three joint matrices multiply as an intrinsic moving-axis sequence, with the first rotation on the left. The wrist attitude R(q) is {ID:36, options:A: R(q1)*R(q2)*R(q3); B: R(q3)*R(q2)*R(q1); C: R(q2)*R(q1)*R(q3), format:choice, label:C2, answer:A}.
37. S is the tool installation in the wrist, R is the wrist attitude in the base frame, and the tool attitude D in the base frame is {ID:37, options:A: R*S; B: S*R, format:choice, label:A2, answer:A}.
38. After substituting q1=0°, R(q1) is {ID:38, options:A: the identity; B: R(q2); C: R(q3), format:choice, label:A2, answer:A}.
R(q3) and S rotate about the same axis.
q3=-60°。
39. On this axis, the combination of q3 and the installation angle already obtained is {ID:39, options:A: 30°; B: -30°; C: -150°; D: 90°, format:choice, label:A2, answer:A}.
40. With q2=30° and R(q1) the identity, the tool attitude after multiplying R(q2) by the previous combined angle is {ID:40, options:A: 30° about Z times 30° about X; B: 30° about Y times 30° about X; C: 30° about Z times -60° about X, format:choice, label:A2, answer:A}.
Before the product, expand each factor with the symbols just chosen.
41. The R(q1) just obtained expands to {ID:41, options:none, format:matrix, label:A2, answer:[[cos(q1), 0, sin(q1)], [0, 1, 0], [-sin(q1), 0, cos(q1)]]}.
42. The R(q2) just obtained expands to {ID:42, options:none, format:matrix, label:A2, answer:[[cos(q2), -sin(q2), 0], [sin(q2), cos(q2), 0], [0, 0, 1]]}.
43. The R(q3) just obtained expands to {ID:43, options:none, format:matrix, label:A2, answer:[[1, 0, 0], [0, cos(q3), -sin(q3)], [0, sin(q3), cos(q3)]]}.
44. The S just obtained expands to {ID:44, options:none, format:matrix, label:A2, answer:[[1, 0, 0], [0, 0, -1], [0, 1, 0]]}.
45. The D computed this way is {ID:45, options:none, format:matrix, label:A2, answer:[[sqrt(3)/2, -sqrt(3)/4, 1/4], [1/2, 3/4, -sqrt(3)/4], [0, 1/2, sqrt(3)/2]]}.
46. The task direction points along base +X_B. In the order X_B, Y_B, Z_B, its unit vector in the base frame is {ID:46, options:none, format:vector, label:C1, answer:[1, 0, 0]}.
47. D maps tool components into the base frame. To map this task direction from the base frame into the tool frame, use {ID:47, options:A: D times the base-frame task direction; B: the transpose of D times the base-frame task direction, format:choice, label:A2, answer:B}.
48. In the order X_T, Y_T, Z_T, the three tool-frame components of the task-direction unit vector are {ID:48, options:none, format:vector, label:A2, answer:[sqrt(3)/2, -sqrt(3)/4, 1/4]}.
49. The second component is the Y_T component, and the positive sense of Y_T is the absolute direction {ID:49, options:A: forward; B: rearward; C: upward; D: downward; E: right; F: left, format:choice, label:C1, answer:C}.
50. A negative component means the task direction points along the absolute direction {ID:50, options:A: forward; B: rearward; C: upward; D: downward; E: right; F: left, format:choice, label:A1, answer:D}.
51. The third component is the Z_T component, and the positive sense of Z_T is the absolute direction {ID:51, options:A: forward; B: rearward; C: upward; D: downward; E: right; F: left, format:choice, label:C1, answer:E}.
52. A positive component means the task direction points along the absolute direction {ID:52, options:A: forward; B: rearward; C: upward; D: downward; E: right; F: left, format:choice, label:A1, answer:E}.
53. The task direction relative to the tool therefore points {ID:53, options:A: down and right; B: up and left; C: down and left, format:choice, label:A1, answer:A}.
54. By the variable definition of the task angles, when tool pitch is positive the task direction points along the tool absolute direction {ID:54, options:A: forward; B: rearward; C: upward; D: downward; E: right; F: left, format:choice, label:C1, answer:C}.
55. When the tool lateral task angle is positive, the task direction points along the tool absolute direction {ID:55, options:A: forward; B: rearward; C: upward; D: downward; E: right; F: left, format:choice, label:C1, answer:E}.
56. When the tool-roll task angle is negative, the turn and the right-hand rule about tool forward are {ID:56, options:A: same; B: opposite, format:choice, label:C1, answer:B}.
57. When only the tool-roll task angle changes, the rotation axis is the tool's absolute {ID:57, options:A: axial; B: normal; C: lateral, format:choice, label:C1, answer:A}.
58. This positive sense and the right-hand rule about this absolute axis are {ID:58, options:A: same; B: opposite, format:choice, label:C1, answer:A}.
59. In this example the task direction points tool-down, so the sign of the tool-pitch task angle is {ID:59, options:A: positive; B: negative, format:choice, label:A1, answer:B}.
60. In this example the task direction points tool-right, so the sign of the tool-lateral task angle is {ID:60, options:A: positive; B: negative, format:choice, label:A1, answer:A}.
61. The right tip follows the tool's absolute lateral direction, and its base-frame coordinate is column {ID:61, options:A: column 1; B: column 2; C: column 3, format:choice, label:A1, answer:C} of D.
62. The sign of this column's component along the base absolute normal is {ID:62, options:A: positive; B: negative, format:choice, label:A1, answer:B}.
63. The right tip therefore deflects toward the base absolute direction {ID:63, options:A: upward; B: downward, format:choice, label:A1, answer:B}.
64. By the zero reference and positive sense of the tool-roll task angle, its sign in this example is {ID:64, options:A: positive; B: negative, format:choice, label:A1, answer:A}.
65. Write the tool pitch, lateral, and roll task angles as declared {ID:65, options:none, format:task-angles, label:A2, answer:-arctan(1/2), arcsin(1/4), arcsin(1/2)}.

### Right-tip leading edge in the tool frame

This layer asks only for the right-tip leading edge in the tool frame. It does not map that point into the base frame.

66. The absolute axial direction (forward) of the tool frame corresponds to {ID:66, options:A: X_T; B: Y_T; C: Z_T; D: -X_T; E: -Y_T; F: -Z_T, format:choice, label:C1, answer:A}.
67. The absolute normal direction (up) of the tool frame corresponds to {ID:67, options:A: X_T; B: Y_T; C: Z_T; D: -X_T; E: -Y_T; F: -Z_T, format:choice, label:C1, answer:B}.
68. The absolute lateral direction (right) of the tool frame corresponds to {ID:68, options:A: X_T; B: Y_T; C: Z_T; D: -X_T; E: -Y_T; F: -Z_T, format:choice, label:C1, answer:C}.
69. The position reference of this example is {ID:69, options:A: right-tip leading edge; B: right-tip trailing edge; C: left-tip leading edge; D: wing-root leading-edge point, format:choice, label:C1, answer:A}.
70. The shape parameter used for the fore-aft position of the tip leading edge relative to the root leading edge is {ID:70, options:A: leading-edge sweep Λ; B: dihedral Γ; C: tip spacing b; D: wing-root coordinate x_r, format:choice, label:C1, answer:A}.
71. When this parameter is positive, the tip leading edge relative to the root leading edge points in the absolute direction {ID:71, options:A: forward; B: rearward, format:choice, label:C1, answer:B}.
72. The fore-aft offset is written on the tool axis that corresponds to the absolute axial positive direction {ID:72, options:A: X_T; B: Y_T; C: Z_T; D: -X_T; E: -Y_T; F: -Z_T, format:choice, label:C2, answer:A}.
73. The positive sense of this axial axis is the absolute direction {ID:73, options:A: forward; B: upward; C: rearward; D: downward; E: right; F: left, format:choice, label:A1, answer:A}.
74. The fore-aft offset makes this axial component {ID:74, options:A: increases; B: decreases, format:choice, label:A1, answer:B}.
75. The fore-aft expression is {ID:75, options:A: (b/2)*tan(Λ); B: -(b/2)*tan(Λ); C: (b/2)*tan(Γ); D: -b*tan(Λ), format:choice, label:A2, answer:B}.
76. The longitudinal coordinate of the wing-root leading edge uses {ID:76, options:A: x_r; B: tip spacing b; C: leading-edge sweep Λ; D: dihedral Γ, format:choice, label:C1, answer:A}.
77. The lateral position of the right tip is written on the tool axis corresponding to the absolute lateral positive direction {ID:77, options:A: X_T; B: Y_T; C: Z_T; D: -X_T; E: -Y_T; F: -Z_T, format:choice, label:C2, answer:C}.
78. The positive sense of this lateral axis is the absolute direction {ID:78, options:A: forward; B: rearward; C: upward; D: downward; E: right; F: left, format:choice, label:A1, answer:E}.
79. The shape parameter used for the lateral coordinate is {ID:79, options:A: tip spacing b; B: tip spacing b and dihedral Γ; C: leading-edge sweep Λ; D: wing-root coordinate x_r, format:choice, label:C1, answer:A}.
80. The lateral-coordinate expression is {ID:80, options:A: (b/2)*cos(Γ); B: (b/2)*sin(Γ); C: b/2; D: (b/2)*tan(Γ), format:choice, label:A2, answer:C}.
81. The vertical position of the right tip is written on the tool axis corresponding to the absolute normal positive direction {ID:81, options:A: X_T; B: Y_T; C: Z_T; D: -X_T; E: -Y_T; F: -Z_T, format:choice, label:C2, answer:B}.
82. The positive sense of this normal axis is the absolute direction {ID:82, options:A: forward; B: rearward; C: upward; D: downward; E: right; F: left, format:choice, label:A1, answer:C}.
83. When the dihedral is positive, the wing tip relative to the wing root points in the absolute direction {ID:83, options:A: upward; B: downward, format:choice, label:C1, answer:A}.
84. The vertical-position expression is {ID:84, options:A: (b/2)*sin(Γ); B: -(b/2)*sin(Γ); C: (b/2)*tan(Γ); D: -(b/2)*cos(Γ), format:choice, label:A2, answer:C}.
85. In the order X_T, Y_T, Z_T, the right-tip leading edge in the tool frame is {ID:85, options:none, format:vector, label:A2, answer:[x_r - (b/2)*tan(Λ), (b/2)*tan(Γ), b/2]}.

### Translation of the tool origin

This layer first finds the distance from the tool nose to the wrist origin, then the translation of the tool origin in the base frame. It does not use the wing-tip coordinate.

86. The tool-frame origin is placed at {ID:86, options:A: tool nose; B: tool center of mass; C: wrist origin; D: tool base, format:choice, label:C1, answer:A}.
87. The wrist-frame origin coincides with {ID:87, options:A: base origin; B: tool nose; C: tool center of mass; D: tool base, format:choice, label:C1, answer:A}.
88. The tool axial length L measures {ID:88, options:A: tool nose to the tool base; B: tool base to the wrist origin; C: tool nose to the wrist origin, format:choice, label:C1, answer:A}.
89. The mount length L_mount measures {ID:89, options:A: tool base to the wrist origin; B: tool nose to the tool base; C: the length of the mount inserted into the tool, format:choice, label:C1, answer:A}.
90. The part of the mount that enters the tool from the tool base {ID:90, options:A: already included in the axial length L, and is not added again; B: add it outside L and L_mount; C: subtract it from L, format:choice, label:C1, answer:A}.
91. Along tool forward, the order of the wrist origin, the tool base, and the tool nose is {ID:91, options:A: O, base, nose; B: nose, base, O; C: base, O, nose, format:choice, label:C1, answer:A}.
92. The distance from the tool nose to the wrist origin is therefore {ID:92, options:none, format:distance, label:A2, answer:L + L_mount}.
93. The direction from the wrist origin to the tool nose is the tool absolute direction {ID:93, options:A: forward; B: rearward; C: upward; D: downward, format:choice, label:C1, answer:A}.
94. The tool axis that receives this offset is {ID:94, options:A: X_T; B: Y_T; C: Z_T; D: -X_T; E: -Y_T; F: -Z_T, format:choice, label:C2, answer:A}.
95. The other two components are {ID:95, options:A: 0; B: L; C: L_mount; D: L + L_mount, format:choice, label:C2, answer:A}.
96. In the order X_T, Y_T, Z_T, the tool-frame components from the wrist origin to the tool origin are {ID:96, options:none, format:vector, label:A2, answer:[L + L_mount, 0, 0]}.
97. After the wrist takes its attitude, mapping this tool-frame vector into the base frame uses {ID:97, options:A: D times that vector; B: the transpose of D times that vector; C: R times that vector; D: do not multiply by the attitude matrix, format:choice, label:A2, answer:A}.
98. Without expanding D, and with square brackets for a column, the translation of the tool origin in the base frame is {ID:98, options:none, format:expression, label:A2, answer:D*[L + L_mount, 0, 0]}.

### Right-tip leading edge in the base frame

This layer asks only for base-frame coordinates. The right-tip leading edge in the tool frame, in the order X_T, Y_T, Z_T, is [x_r - (b/2)*tan(Λ), (b/2)*tan(Γ), b/2]. The tool origin is not the base origin. Its translation in the base frame is D*[L + L_mount, 0, 0]. Square brackets denote a column.

99. Mapping this tool coordinate into the base frame uses {ID:99, options:A: D times the tool coordinate; B: the transpose of D times the tool coordinate; C: R times the tool coordinate, format:choice, label:A2, answer:A}.
100. The base-frame position also combines the translation with this rotated result by {ID:100, options:A: add; B: keep only the rotated result; C: keep only the translation, format:choice, label:A2, answer:A}.
101. Without expanding D, and in the order X_B, Y_B, Z_B, the right-tip leading edge in the base frame is {ID:101, options:none, format:expression, label:A2, answer:D*[L + L_mount, 0, 0] + D*[x_r - (b/2)*tan(Λ), (b/2)*tan(Γ), b/2]}.

### Boundary reachability of the designated points

This layer analyzes only three designated points: the tool nose, the right tip, and the left tip. It finds the three reachable base-coordinate intervals of the nose inside the declared joint range, and it decides whether these three points can reach the cell boundary. Touching the boundary counts as reaching it. The result is not a collision conclusion for the whole tool. The vertical and lateral coordinates of the tips are not expanded into closed intervals. The wrist origin is at the center of the cross-section. The right tip is the right-tip leading edge, and its tool-frame coordinate is [x_r - (b/2)*tan(Λ), (b/2)*tan(Γ), b/2]. The nose relative to the wrist origin has tool-frame components [L + L_mount, 0, 0]. This layer cannot see the blanks of earlier subprocesses, and it does not replace the whole range by this example's one joint triple. The test covers only these three points.

102. The absolute directions limited by the cell boundary are {ID:102, options:A: up with down, and right with left; B: forward with rearward, and up with down; C: forward with rearward, and right with left, format:choice, label:C1, answer:A}.
103. Along the base task direction, {ID:103, options:A: no boundary; B: place the boundary forward; C: place the boundary rearward, format:choice, label:C1, answer:A}.
104. The base axis limited by the vertical boundaries is {ID:104, options:A: X_B; B: Y_B; C: Z_B, format:choice, label:C1, answer:B}.
105. The base axis limited by the lateral boundaries is {ID:105, options:A: X_B; B: Y_B; C: Z_B, format:choice, label:C1, answer:C}.
106. The cross-section side is L_cell and the wrist origin is at its center, so the distance from the center to one face is {ID:106, options:none, format:expression, label:A2, answer:L_cell/2}.
107. The two side boundaries lie at {ID:107, options:A: both the positive and negative sides; B: only on the positive side; C: only on the negative side, format:choice, label:A1, answer:A}.
108. The tool-frame vector of the right tip relative to the wrist origin is obtained from the nose offset and the right-tip leading-edge coordinate by {ID:108, options:A: add; B: subtract; C: keep only the wing-tip coordinate, format:choice, label:A2, answer:A}.
109. In the order X_T, Y_T, Z_T, that vector is {ID:109, options:none, format:vector, label:A2, answer:[L + L_mount + x_r - (b/2)*tan(Λ), (b/2)*tan(Γ), b/2]}.
110. Compared with the right tip, the left tip has {ID:110, options:A: fore-aft and vertical unchanged, lateral reversed; B: negate all three components; C: fore-aft reversed, lateral unchanged, format:choice, label:A1, answer:A}.
111. In the order X_T, Y_T, Z_T, the left-tip vector is {ID:111, options:none, format:vector, label:A2, answer:[L + L_mount + x_r - (b/2)*tan(Λ), (b/2)*tan(Γ), -b/2]}.
112. Mapping these three tool-frame vectors into the base frame uses {ID:112, options:A: D times that vector; B: the transpose of D times that vector; C: R times that vector; D: do not multiply by the attitude matrix, format:choice, label:A2, answer:A}.
q3 and the installation angle rotate about the same axis, and the installation angle is +90°.
113. While sweeping the range, the combined angle θ is {ID:113, options:A: q3+90°; B: q3-90°; C: q3; D: q1+90°, format:choice, label:A2, answer:A}.
114. As q3 runs from -90° to 90°, the range of θ is {ID:114, options:A: 0° to 180°; B: -90° to 90°; C: -180° to 0°; D: 0° to 90°, format:choice, label:A2, answer:A}.
115. Let v=R(q2)*R(θ)*r. v is the intermediate vector of the product, with components in X, Y, Z order, before left-multiplication by R(q1). The q1 matrix is on the far left and rotates about base axis {ID:115, options:A: X_B; B: Y_B; C: Z_B, format:choice, label:C2, answer:B}.
116. After left-multiplying by R(q1), the vertical base coordinate of the point and q1 are {ID:116, options:A: unrelated; B: related, format:choice, label:A1, answer:A}.
117. That vertical coordinate therefore equals component {ID:117, options:A: one; B: two; C: three, format:choice, label:A2, answer:B} of v.
118. R(θ) does not change the tool-frame vector of the nose. After left-multiplying by R(q2), the intermediate vector is {ID:118, options:none, format:vector, label:A2, answer:[(L + L_mount)*cos(q2), (L + L_mount)*sin(q2), 0]}.
119. Left-multiplying by R(q1) does not change this second component, so the vertical base coordinate of the nose equals {ID:119, options:A: (L + L_mount)*sin(q2); B: (L + L_mount)*cos(q1); C: -(L + L_mount)*sin(q1), format:choice, label:A2, answer:A}.
120. q2 is the lateral axis, on -90° to 90°, and the values sin(q2) can take cover {ID:120, options:A: -1 to 1; B: -1/2 to 1/2; C: -sqrt(3)/2 to sqrt(3)/2, format:choice, label:A2, answer:A}.
121. q1 is the normal axis, on -60° to 60°, and the minimum of cos(q1) is {ID:121, options:A: 1/2; B: 0; C: sqrt(3)/2; D: 1, format:choice, label:A2, answer:A}.
122. On -90° to 90°, cos(q2) is {ID:122, options:A: not negative; B: not positive; C: always 1, format:choice, label:A1, answer:A}.
123. The greatest absolute value of sin(q1) is {ID:123, options:A: sqrt(3)/2; B: 1/2; C: 1; D: 0, format:choice, label:A2, answer:A}.
124. The axial base coordinate of the nose is {ID:124, options:A: (L + L_mount)*cos(q1)*cos(q2); B: (L + L_mount)*cos(q2); C: (L + L_mount)*sin(q2), format:choice, label:A2, answer:A}.
125. The lateral coordinate is {ID:125, options:A: -(L + L_mount)*sin(q1)*cos(q2); B: (L + L_mount)*sin(q2); C: -(L + L_mount)*sin(q2), format:choice, label:A2, answer:A}.
126. The axial reachable interval is {ID:126, options:none, format:interval, label:A2, answer:[0, L + L_mount]}.
127. The vertical reachable interval is {ID:127, options:none, format:interval, label:A2, answer:[-(L + L_mount), L + L_mount]}.
128. The lateral reachable interval is {ID:128, options:none, format:interval, label:A2, answer:[-(L + L_mount)*sqrt(3)/2, (L + L_mount)*sqrt(3)/2]}.
129. The greatest absolute vertical coordinate of the nose, compared with its greatest absolute lateral coordinate, is {ID:129, options:A: larger; B: smaller; C: equal, format:choice, label:A1, answer:A}.
130. As the nose-to-wrist distance grows, the first boundary-contact threshold corresponds to {ID:130, options:A: vertical boundaries; B: lateral boundaries; C: both axial ends, format:choice, label:A1, answer:A}.
131. Touching the boundary counts as contact. The nose can meet the boundary when {ID:131, options:none, format:expression, label:A2, answer:L + L_mount >= L_cell/2}.
132. Inside the declared joint range, the greatest reachable absolute vertical base coordinate of the point equals {ID:132, options:A: the distance from that point to the wrist origin; B: half of that distance; C: that distance times sqrt(3)/2, format:choice, label:A2, answer:A}.
A θ can be chosen so that the third component after R(θ) is 0, and q2 on -90° to 90° then turns the vector toward the positive or negative sense of Y_B.
q1 rotates about Y_B and does not change the vertical coordinate.
This greatest reachable value therefore equals the distance from that point to the wrist origin.
133. A point at least as far from the wrist origin as the center-to-boundary distance is, for a boundary-touching attitude, {ID:133, options:A: necessary and sufficient; B: sufficient only; C: necessary but not sufficient, format:choice, label:A2, answer:A}.
134. The axial reachable interval {ID:134, options:A: does not enter; B: also enters, format:choice, label:C1, answer:A} the boundary-contact test.
135. If any one of the three points is at least the boundary distance from the wrist origin, the decision is {ID:135, options:A: a boundary contact is possible; B: a boundary contact is impossible, format:choice, label:C2, answer:A}.
136. When a point is closer to the wrist origin than the boundary distance, that point {ID:136, options:A: a boundary contact is impossible; B: a boundary contact is possible, format:choice, label:A2, answer:A}.
137. The sum of squares of the last two components of the right tip's tool-frame vector is {ID:137, options:A: (b/2)^2; B: (b/2)^2*(1 + (tan(Γ))^2); C: b^2; D: (b/2)^2*(tan(Γ))^2, format:choice, label:A2, answer:B}.
138. The distance from the right tip to the wrist origin is {ID:138, options:none, format:expression, label:A2, answer:sqrt((L + L_mount + x_r - (b/2)*tan(Λ))^2 + (b/2)^2*(1 + (tan(Γ))^2))}.
139. This distance of the left tip and the right tip are {ID:139, options:A: same; B: opposite; C: differs by L, format:choice, label:A1, answer:A}.
140. The test covers only the three checked points. Inside the declared joint range, at least one point meets or crosses the boundary when {ID:140, options:none, format:expression, label:A2, answer:max(L + L_mount, sqrt((L + L_mount + x_r - (b/2)*tan(Λ))^2 + (b/2)^2*(1 + (tan(Γ))^2))) >= L_cell/2}.

# Inverse kinematics

This sample is injected alone. Its declaration does not contain the forward-kinematics joint angles or a numerical D.

## Declaration

Fixed conventions
- Vectors are columns, and a transform matrix multiplies a column on the left. Components in the parent frame equal the attitude matrix times components in the child frame. Components are written in the X, Y, Z order of that frame.
- Base frame: X_B points base-forward, the task direction. Y_B points base-up. Z_B points base-right.
- Wrist frame: X_W points physically forward, Y_W points physically up, and Z_W points physically right. These physical directions are fixed to the wrist. At the initial pose they coincide with base forward, up, and right.
- Tool frame: X_T points tool-forward, Y_T points tool-up, and Z_T points tool-right. The tool frame is fixed to the aircraft-shaped end tool.
- Wrist rotations use an intrinsic moving-axis sequence. The absolute-axis order is a variable definition. The order here is normal, lateral, axial, and the angles are q1, q2, q3. The first rotation is written on the left.
- The joint matrices are R(q1), R(q2), and R(q3). Every positive sense agrees with the right-hand rule about the corresponding absolute direction. The declaration does not give their entries. R=R(q1)*R(q2)*R(q3).
- The normal, lateral, and axial names of each joint are the corresponding directions in the moving frame before that joint acts. The wrist frame describes the final attitude. The three joint axes are not always fixed to one set of end directions.
- Joint ranges are bound to the absolute axes. Normal-axis q1 runs from -60° to 60°, where the cosine is positive. Lateral-axis q2 and axial-axis q3 both run from -90° to 90°.
- The inverse for this wrist uses the nonsingular branch -90°<q2<90°. The full mechanical limit still allows q2=±90°, but singular boundary poses are not part of the unique-inverse task.
- When every joint angle is 0, tool forward coincides with base +X_B and tool up coincides with base +Z_B. Because the three directions must form a right-handed frame, tool right coincides with base -Y_B. S is the installation of the tool frame in the wrist frame.
- D is the attitude of the tool frame relative to the base frame. It maps tool components into base components.
- The task direction points along base +X_B. In the order X_B, Y_B, Z_B, its unit vector in the base frame is [1, 0, 0]. In the tool frame it equals the transpose of D times that column, which is the first row of D.
- This sample does not give the joint angles or a numerical D.

Tool task angles
Write the task direction in the tool frame as [d_x, d_y, d_z], in the order X_T, Y_T, Z_T.
- Tool-pitch task angle α_T: the signed angle between tool forward and the projection of the task direction onto the tool plane of symmetry. The angle is positive when the task direction points upward. α_T=atan2(d_y, d_x).
- Tool-lateral task angle β_T: the signed angle of the task direction away from the tool plane of symmetry. The angle is positive when the task direction points to the right. β_T=atan2(d_z, sqrt(d_x^2 + d_y^2)).
- Tool-roll task angle γ_T: let t be the unit vector of tool forward, n the unit vector of tool up, and e_Y the unit vector of the base reference-up direction. e_Y=[0, 1, 0]. u=(e_Y-(e_Y·t)*t)/||e_Y-(e_Y·t)*t||. Let r=t×u. γ_T=atan2(n·r, n·u). The positive sense is the same as the right-hand rule about tool forward, and the opposite sense is negative. The roll task angle is undefined when tool forward is parallel to the base reference-up direction.
The three target task angles of this robot task are arctan(1/2), arcsin(1/4), arcsin(1/2).
Recovering the joint angles from the three task angles is done only for -90°<α_T<90°, -90°<β_T<90°, when the tool forward axis is not parallel to the base reference-up direction, and on the nonsingular lateral branch -90°<q2<90°.

## Background

Vectors are columns, and an attitude matrix multiplies a column on the left. Its transpose maps parent-frame components back into the child frame. The task-direction unit vector is [1, 0, 0] in the base frame, and its tool-frame components are the first row of D. Tool pitch is positive when the task direction points upward. The tool lateral angle is positive when the task direction points to the right. Tool roll is measured from the projection of the base reference-up direction onto the plane perpendicular to tool forward. Its positive sense is the same as the right-hand rule about the tool forward axis, and the opposite sense is negative. Tool roll is undefined when tool forward is parallel to the base reference-up direction. The intrinsic moving-axis order in this example is normal, lateral, axial. cos(q1) is positive inside the declared range. The unique inverse on the lateral axis uses only -90°<q2<90°.

## Full process


### Joint angles from the three task angles

1. If only the tool pitch and lateral task angles are given, and not the tool roll, can the three joints be unique? The answer is {ID:1, options:A: can; B: cannot, format:choice, label:A2, answer:B}.
Both joint triples are inside the declared range: q=(0°, 90°, 0°) and q=(30°, 90°, -30°).
2. Checked with the declared product, the attitude matrices of these two sets are {ID:2, options:A: same; B: different, format:choice, label:A2, answer:A}.
3. In both sets the tool forward axis is parallel to the base reference-up direction. Under this sample's conditions, {ID:3, options:A: still recover the joint angles from the three task angles; B: do not recover the joint angles from the three task angles, format:choice, label:C1, answer:B}.
4. Using the conditions listed for this sample, decide whether the three target task angles determine the joint angles uniquely. The answer is {ID:4, options:A: can; B: cannot, format:choice, label:A2, answer:A}.
5. These three task angles fix the tool attitude relative to the base frame. The corresponding matrix is {ID:5, options:A: D; B: R; C: S, format:choice, label:C1, answer:A}.
6. Let a=cos(α_T)*cos(β_T), bv=sin(α_T)*cos(β_T), and c=sin(β_T). The first row of D is the task direction in the tool frame and equals {ID:6, options:A: [a, bv, c]; B: [a, c, bv]; C: [bv, a, c], format:choice, label:C2, answer:A}.
7. At this example's angles, this row evaluates to {ID:7, options:none, format:vector, label:A2, answer:[sqrt(3)/2, sqrt(3)/4, 1/4]}.
8. Let d=bv*cos(γ_T)-c*sin(γ_T) and N=sqrt(a^2+d^2). On the branch a>0, using the roll zero reference and the right-hand rule about tool forward, the second row is {ID:8, options:A: [-d/N, a*cos(γ_T)/N, -a*sin(γ_T)/N]; B: [d/N, a*cos(γ_T)/N, a*sin(γ_T)/N]; C: [-d/N, a*sin(γ_T)/N, a*cos(γ_T)/N], format:choice, label:A2, answer:A}.
9. The third row is the cross product of the first and second rows, so the computed D is {ID:9, options:none, format:matrix, label:A2, answer:[[sqrt(3)/2, sqrt(3)/4, 1/4], [-sqrt(13)/13, 3*sqrt(13)/13, -sqrt(39)/13], [-3*sqrt(13)/26, 5*sqrt(13)/52, 7*sqrt(39)/52]]}.
10. At the zero pose, tool forward points {ID:10, options:A: task forward direction; B: base up; C: base right; D: base down, format:choice, label:C1, answer:A}.
11. Tool up coincides with the base absolute direction {ID:11, options:A: upward; B: downward; C: right; D: left, format:choice, label:C1, answer:C}.
Tool forward coincides with base forward.
Tool up coincides with base right.
12. By tool forward × tool up = tool right, tool right points along the base absolute direction {ID:12, options:A: upward; B: downward; C: left; D: right, format:choice, label:A1, answer:B}.
The columns of S follow X_T, Y_T, Z_T, and the rows follow X_W, Y_W, Z_W.
At the zero pose the wrist's physical directions coincide with the base directions.
13. The installation matrix S is {ID:13, options:none, format:matrix, label:A2, answer:[[1, 0, 0], [0, 0, -1], [0, 1, 0]]}.
14. To remove the fixed installation, the wrist attitude relative to the base frame should be {ID:14, options:A: D*S^T; B: S^T*D; C: D*S, format:choice, label:A2, answer:A}.
Call the wrist attitude just obtained H. Its entries are hij, with the row index first and the column index second.
15. On the nonsingular branch -90°<q2<90°, the extraction formulas for q1, q2, and q3 are {ID:15, options:none, format:formula, label:A2, answer:atan2(-h31, h11); atan2(h21, sqrt(h11^2 + h31^2)); atan2(-h23, h22)}.
16. The three joint angles obtained this way are {ID:16, options:none, format:joint-angles, label:A2, answer:[26, -16, -60]}.

# Cross-layer correspondence

This section is not sent to the model. The basic-theory questions that used to stand alone are folded into this manuscript and are not stored as a separate theory paper. α, β, and γ are written α_T, β_T, and γ_T here. The L of the theory paper is L + L_mount here.

The forward sample measures conditional ability with step-by-step prompts. The inverse sample is injected alone, to see the inverse after the joint angles and the forward process are removed.

| kinematics | computation that is kept | corresponding operation in the basic theory |
|---|---|---|
| Forward items 1-9, which axis each direction of the base, wrist, and tool frames uses | three axis declarations | no absolute directions |
| Forward items 10-24, the absolute-axis order, positive sense, destination, and matrices. The order here is normal, lateral, axial | the joint axis reads the order and the three wrist axes; R(q1), R(q2), and R(q3) read that axis, the positive sense, and the destination | give Rx, Ry, and Rz directly |
| Forward items 25-35 write S before using the installation matrix. At the zero pose, tool forward coincides with base +X_B, tool up with base +Z_B, and tool right with base -Y_B | the three columns read the zero-pose directions and S is stacked from them; the axis and angle then read S | give the installation matrix directly |
| Forward items 36-53 decide the relation of D to R and S. The expansions copy the joint matrices and S just obtained, and D reads the product choice and those four expansions. Item 46 reads the relation of the task direction to base +X_B | use one joint triple for the tool attitude and the task direction | compose D from the given matrices, then write the direction angles |
| Forward items 54-65, the signs and values of the three task angles. Roll uses the projection of base reference-up onto the plane perpendicular to tool forward as its zero | tool pitch, lateral, and roll task angles | write α, β, and γ from the components |
| Forward items 66-85, the right-tip leading edge in the tool frame | the axis reads the absolute direction, the expression reads that axis's positive sense and the chosen parameter, and the coordinate vector reads the three expressions | no corresponding item |
| Forward items 86-98, the nose-to-wrist distance and the translation of the tool origin in the base frame. The order along forward is O, base, nose | the translation column. The basic theory writes the whole distance as L | the translation column of a homogeneous transform |
| Forward items 99-101, the right-tip leading edge in the base frame, including translation | kept in this manuscript | no corresponding item |
| Forward items 102-140, boundary reachability of the three designated points. The cell extends along X_B and the cross-section is square, so the vertical and lateral limits share L_cell/2. The test covers only these three points, is not a whole-tool collision conclusion, and holds only for the current rotation order and zero-pose installation | kept in this manuscript | no corresponding item |
| Inverse items 1-4 ask separately about the missing freedom of two task angles and about the multiple solutions at the lateral-axis boundary. Inverse items 5-9 build D from the three task angles | recover the attitude from the target task angles | the solvability condition, and recovering D from the coordinate angles |
| Inverse items 10-13 write the matrix from the zero-pose installation before using S | tool right comes from the zero-pose cross product, and S reads these three directions | give the installation matrix directly |
| Inverse item 14 first asks for the attitude after removing the installation. Inverse items 15-16 call that attitude H and then extract the joint angles | the joint angles read the D and S just written, then remove the installation by the chosen product | obtain the joint angles from the given extraction formula |

# Generation notes

This section is not sent to the model.

D is the attitude of the tool frame relative to the base frame. With column vectors and left multiplication, D=R*S numerically. Forward item 37 asks the model to judge that relation. The declaration only says what each of the three matrices maps, and it does not write the product. Inverse item 14 asks for the attitude D*S^T after the installation is removed, and the declaration does not write that product either.

The task direction in the base frame is [1, 0, 0]. Forward item 46 asks the model to write that vector from the phrase pointing along +X_B, so the forward declaration does not give the three numbers. The inverse sample does not ask this, and its declaration may give them.

Injection is bounded by the sentence period, with one open blank in each sentence. There is no separate step id. Blanks before the current sentence are filled with the answer text. Text after the blank in the same sentence is visible, so the answer is not written into that later text.

Inverse item 1 asks the model to judge for itself that three joints cannot be unique when only the pitch and lateral task angles are given. Inverse item 2 asks the model to check whether q=(0°, 90°, 0°) and q=(30°, 90°, -30°) have the same attitude matrix. Those two conclusions are written only here: two task angles do not determine three joints; at q2=±90° one attitude can correspond to several joint triples, and the two triples above are the same attitude. Do not put these two sentences into the background or into the text before items 1 and 2. The declaration only lists the conditions for a unique solution and does not say that the given values already satisfy them. The gold answer of item 4 is still that it can.

Scores are separated by label. C1 is definition recognition. A1 and A2 are spatial inference from the definition. Forward items 25-29, 54-58, and 66-84 contain both kinds. Do not treat the C1 accuracy as independent spatial reasoning.

The inverse sample must use an independent session. Do not attach the whole review manuscript at once, and do not carry the answers, the scoring notes, the cross-layer table, or this section. The forward and inverse samples of the same underlying case stay in the same split. Do not put them on opposite sides of a train and test split. Do not put the forward sample's [0, 30, -60] in a shared prompt. Do not write D=Rz(30°)*Rx(30°) in the inverse declaration, and do not write a numerical D.

The inverse extraction holds only under the current default: the intrinsic moving-axis order is normal, lateral, axial, and the nonsingular branch is -90°<q2<90°. The full mechanical limit still allows q2=±90°, but singular boundary poses are not part of the unique-inverse task. After the rotation order changes, the extraction must be derived again. Replacing axis names and joint indices is not enough. The atan2 pattern in the format example is not the gold answer for these joints. Under the default order the extraction is atan2(-h31, h11); atan2(h21, sqrt(h11^2 + h31^2)); atan2(-h23, h22).

The boundary reachability of the three designated points, and the final test, hold only under the current default: the intrinsic order is normal, lateral, axial; the zero-pose installation puts tool forward on base +X_B and tool up on base +Z_B; and the cell extends along X_B with a square cross-section. Therefore q3 shares an axis with the installation angle, θ=q3+90°, the leftmost matrix rotates about Y_B, and the vertical and lateral limits share L_cell/2. The test covers only the tool nose, the right tip, and the left tip. It is not a collision conclusion for the whole tool. After the installation changes, q3 cannot always be combined directly with the installation angle. After the order changes, the leftmost matrix need not rotate about Y_B, and a point need not be able to reach a boundary normal inside the given limits. A point whose distance from the wrist origin equals the boundary distance then gives only a theoretical length bound and cannot be treated as an attitude that is actually reachable. Do not extend the test to another rotation order, another installation, or a rectangular rather than square cross-section. A new sample has to derive reachability again.
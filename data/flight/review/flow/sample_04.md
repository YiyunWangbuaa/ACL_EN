# Data 04 roll reversed

This is a flight-mechanics review manuscript. Blanks are numbered 1 through 131. The first four subprocesses convert coordinates among the ground, track, airframe, and airflow. The last two convert the six aerodynamic components and transfer the moment reference point. The load subprocess and the reference-point subprocess are injected separately: the reference-point subprocess states the airframe force and moment directly and does not use the answers of the load subprocess. This manuscript does not judge the flight envelope and does not apply an error correction.

Definition of this item: Roll is negative when the aircraft rolls to its right.
Single-definition perturbation: roll

Review notes, labels, answers, and generation notes are not sent to the model. The output-format rules are sent. The first four subprocesses use the declaration and the background above. The last two use the aerodynamic-load and reference-point declaration and do not carry the text of the first four. The model also receives the current subprocess name and the text from the start of that subprocess through the period of the current sentence. Blanks before the current sentence are filled with the answer text. Each sentence has one open blank.

## Output format

- choice: multiple choice. Write only the option letter. Example: A
- vector: three components on one line, as [a, b, c]. Put exactly one space after each comma. The order is X, Y, Z. Example: [1, 0, 0]
- matrix: one 3×3 matrix, as [[a, b, c], [d, e, f], [g, h, i]]. Put exactly one space after each comma. Example: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
- expression: one line. Use * for multiplication. Example: a*cos(α)+b

## Declaration

Fixed conventions
- Coordinate transforms are passive. Attitude angles, track angles, and airflow angles use an intrinsic moving-axis sequence.
- Vectors are column vectors, and a transform matrix multiplies a column vector on the left. Component transforms act from right to left.
- gRb is the attitude of the airframe relative to the ground frame and maps airframe components into the ground frame. The reverse transform is its transpose.
- For the moving-axis attitude yaw, pitch, roll, the component transform from the ground frame to the airframe is bRg=Cx(-Φ)*Cy(θ)*Cz(ψ), and the reverse transform is its transpose. Every other transform takes its product order from the stated start frame, target frame, and intermediate frame.
- Components are written in the X, Y, Z order of each frame.
- Take -90°<α, β, θ, μ<90°, so the attack projection, the horizontal track projection, and the attitude branch used here do not meet a degeneracy that this manuscript does not handle. Azimuth and roll use the principal interval [-180°, 180°).

Mathematical rules
- The transpose of an orthogonal matrix equals its inverse.
- The determinant of any valid three-dimensional rotation matrix is +1.

Absolute directions and chirality
- The ground frame, airframe, stability frame, track frame, and airflow frame are right-handed.
- The positive axial direction is forward, the positive lateral direction is right, and the positive normal direction is down. Forward × right = down.
- Axial, lateral, and normal are physical directions, not fixed coordinate-axis names. The local positive direction of each frame axis does not change with the attitude angles.

Frames
- Ground frame {g}: Xg is forward in the horizontal plane, Yg is right in the horizontal plane, and Zg is vertically down. The ground frame is the fixed reference.
- Airframe {b}: xb follows the nose, yb points right on the airframe, and zb points down on the airframe. The airframe is fixed to the aircraft. The plane of symmetry is the xb-zb plane.
- Stability frame {s}: Xs is the forward projection of the airspeed onto the airframe plane of symmetry, Ys is perpendicular to that plane and points right, and Zs lies in the plane, is perpendicular to Xs, and points down.
- Track frame {p}: Xp follows the airspeed, Zp lies in the vertical plane through the airspeed, is perpendicular to the airspeed, and takes the downward branch, and Yp=Zp×Xp. The track frame has no roll. The track here uses the airspeed direction, not the ground speed.
- Airflow frame {a}: Xa follows the airspeed. Za lies in the airframe plane of symmetry, is perpendicular to Xa, and is the branch whose dot product with Zb is positive. Ya=Za×Xa.
- The airflow frame rolls relative to the track frame about their shared airspeed axis. γ follows the right-hand rule about +Xp, and the sign is then the sign in this angle declaration.

Angles
- Yaw ψ: the angle between the ground projection of the positive airframe axis and the ground axis Xg. When the nose yaws to the right of the ground, the angle is positive.
- Pitch θ: the angle between the positive airframe axis and the ground. When the nose pitches above the ground, the angle is positive.
- Roll Φ: the angle between the positive airframe normal and the vertical plane through the airframe axis. When the aircraft rolls to its right, the angle is negative.
- Track azimuth φ: the angle between the ground projection of the airspeed vector (the positive airflow axis) and the ground axis Xg. When the projection lies to the right of Xg, the angle is positive.
- Track inclination μ: the angle between the airspeed vector (the positive airflow axis) and the ground. When the aircraft climbs, the angle is positive.
- Track roll γ: the angle between the positive airflow normal and the vertical plane through the airspeed. When the aircraft rolls to its right, the angle is positive. It is the roll of the airflow frame relative to the track frame and does not change the track frame itself.
- Angle of attack α: the angle between the projection of the airspeed onto the aircraft plane of symmetry and the positive airframe axis. When that projection lies below the airframe axis, the angle is positive.
- Sideslip β: the angle between the airspeed vector and the aircraft plane of symmetry. When the airspeed lies to the right of the aircraft, the angle is positive.

Elementary rotation matrices
Let cη=cos(η) and sη=sin(η). The following matrices are passive transforms about the current coordinate axis. A positive angle follows the right-hand rule about that axis.
- Cx(η)=[[1, 0, 0], [0, cη, sη], [0, -sη, cη]], about the axial axis X.
- Cy(η)=[[cη, 0, -sη], [0, 1, 0], [sη, 0, cη]], about the lateral axis Y.
- Cz(η)=[[cη, sη, 0], [-sη, cη, 0], [0, 0, 1]], about the normal axis Z.

Airspeed direction
- Airspeed is the velocity of the aircraft relative to the air, with magnitude V>0. This manuscript does not use ground speed and does not introduce a wind.
- Track Xp and airflow Xa both follow the airspeed, so the unit airspeed in both frames is [1, 0, 0].

## Background

Ground Zg points down. A positive ground normal component points down, and a negative component points up. Each angle is taken about the absolute axis named in the declaration, and its positive sense is the sense in the declaration. Coordinate transforms are passive. Attitude, track, and airflow angles use an intrinsic moving-axis sequence.

## Full process

### Subprocess 1: ground frame to airframe

1. The positive normal of the ground frame, the airframe, and the track frame points{ID:1, options:A: upward; B: downward, format:choice, label:C1, answer:B}.
2. The positive lateral direction of the ground frame, the airframe, and the track frame points{ID:2, options:A: right; B: left, format:choice, label:C1, answer:A}.
3. Forward, right, and down on the airframe form{ID:3, options:A: left-handed; B: right-handed, format:choice, label:C1, answer:B}.
4. When yaw is positive, the horizontal projection of the airframe forward axis turns{ID:4, options:A: right on the ground; B: left on the ground, format:choice, label:C1, answer:A}.
5. When pitch is positive, the nose points{ID:5, options:A: downward; B: upward, format:choice, label:C1, answer:B}.
6. Roll Φ is taken about the airframe{ID:6, options:A: axial; B: lateral; C: normal, format:choice, label:C1, answer:A}.
7. Coordinate transforms in this item use{ID:7, options:A: active; B: passive, format:choice, label:C2, answer:B}.
8. Attitude angles use{ID:8, options:A: intrinsic moving axes; B: extrinsic fixed axes, format:choice, label:C2, answer:A}.
9. After the yaw-pitch-roll attitude, bRg=Cx(-Φ)*Cy(θ)*Cz(ψ) maps ground-frame components into the airframe, and the yaw matrix is{ID:9, options:A: the leftmost factor; B: the rightmost factor, format:choice, label:C2, answer:B}.
10. gRb converts airframe coordinates into the ground frame, and its mapping is{ID:10, options:A: airframe to ground frame; B: ground frame to airframe, format:choice, label:C2, answer:A}.
11. A ground-frame vector is converted into the airframe by using gRb's{ID:11, options:A: itself; B: transpose, format:choice, label:C2, answer:B}.
12. By the declaration, the factored form of gRb is{ID:12, options:A: Cz(-ψ)*Cy(-θ)*Cx(Φ); B: Cx(Φ)*Cy(θ)*Cz(ψ); C: Cz(ψ)*Cy(θ)*Cx(Φ); D: Cy(θ)*Cz(ψ)*Cx(Φ), format:choice, label:C2, answer:A}.
13. The airframe coordinates of the airframe forward unit vector are{ID:13, options:none, format:vector, label:C1, answer:[1, 0, 0]}.
14. Map the airframe forward axis into the ground frame by computing{ID:14, options:A: gRb*[1, 0, 0]; B: gRb^T*[1, 0, 0], format:choice, label:C2, answer:A}.
15. At ψ=30°, θ=20°, and Φ=15°, the ground-frame components of the airframe forward axis are{ID:15, options:none, format:vector, label:A2, answer:[cos(20°)*cos(30°), sin(30°)*cos(20°), -sin(20°)]}.
16. From the sign of that ground normal component, the nose points{ID:16, options:A: upward; B: downward, format:choice, label:A1, answer:A}.
17. In the full attitude matrix, the airframe right axis in the ground frame is column{ID:17, options:A: the third column; B: the second column; C: the first column, format:choice, label:C2, answer:B}.
18. The airframe down axis in the ground frame is column{ID:18, options:A: the third column; B: the first column; C: the second column, format:choice, label:C2, answer:A}.
19. The transpose of an orthogonal attitude matrix equals its{ID:19, options:A: negative of the matrix; B: determinant; C: inverse, format:choice, label:C1, answer:C}.
20. The determinant of any valid three-dimensional rotation matrix is{ID:20, options:A: 0; B: +1; C: -1, format:choice, label:C1, answer:B}.

### Subprocess 2: ground frame to track frame

21. Track azimuth φ describes the horizontal airspeed projection relative to{ID:21, options:A: Xg; B: Yg; C: Zg, format:choice, label:C1, answer:A}.
22. Positive track inclination means the aircraft moves{ID:22, options:A: upward; B: downward, format:choice, label:C1, answer:A}.
23. Track roll γ rotates the airflow frame about the shared airspeed axis of the track frame, which is{ID:23, options:A: Zp; B: Xp; C: Yp, format:choice, label:C1, answer:B}.
24. The track attitude seen from the ground frame is{ID:24, options:A: gRp; B: pRg, format:choice, label:C1, answer:A}.
25. By the current declaration, the factored form of gRp without roll is{ID:25, options:A: Cz(-φ)*Cy(-μ); B: Cz(φ)*Cy(μ); C: Cy(-μ)*Cz(-φ); D: Cz(-φ)*Cy(μ); E: Cz(-φ)*Cy(-μ)*Cx(-γ), format:choice, label:C2, answer:A}.
26. The track-frame unit airspeed vector is{ID:26, options:none, format:vector, label:C1, answer:[1, 0, 0]}.
27. Convert the track-frame unit airspeed into the ground frame with{ID:27, options:A: gRp^T; B: gRp, format:choice, label:C2, answer:B}.
Compute the ground-frame airspeed at φ=40° and μ=10°.

28. The ground normal component of the airspeed is{ID:28, options:A: sin(10°); B: cos(10°); C: -sin(10°), format:choice, label:A2, answer:C}.
29. From the sign of that ground normal component, the airspeed points{ID:29, options:A: downward; B: upward, format:choice, label:A1, answer:B}.
30. When track inclination is zero, the ground normal component of the airspeed is{ID:30, options:A: 0; B: +1; C: -1, format:choice, label:A2, answer:A}.
31. At μ=0, changing track azimuth makes the horizontal airspeed projection{ID:31, options:A: stays unchanged; B: turns, format:choice, label:A1, answer:B}.
32. When track roll γ changes, the airspeed direction itself{ID:32, options:A: does not change; B: changes, format:choice, label:A1, answer:A}.
33. Track roll changes{ID:33, options:A: the directions of airflow Ya and Za relative to the track frame; B: the track axes Yp and Zp; C: the airspeed magnitude, format:choice, label:A1, answer:A}.
34. The ground-frame airspeed written from the track direction is{ID:34, options:none, format:vector, label:A2, answer:[cos(φ)*cos(μ), sin(φ)*cos(μ), -sin(μ)]}.

### Subprocess 3: airframe, stability frame, and airflow frame

35. The stability axis Xs follows the airspeed projected onto the aircraft{ID:35, options:A: spanwise wing projection; B: projection in the plane of symmetry, format:choice, label:C1, answer:B}.
36. When angle of attack α is positive, the airspeed in the plane of symmetry points{ID:36, options:A: downward; B: upward, format:choice, label:C1, answer:A}.
37. When sideslip β is positive, the airspeed points{ID:37, options:A: left; B: right, format:choice, label:C1, answer:B}.
38. The attitude difference from the airframe to the stability frame is only{ID:38, options:A: track azimuth; B: angle of attack; C: sideslip, format:choice, label:C2, answer:B}.
39. The attitude difference from the stability frame to the airflow frame is only{ID:39, options:A: sideslip; B: angle of attack; C: yaw, format:choice, label:C2, answer:A}.
40. By the declaration, the matrix from the stability frame to the airframe is{ID:40, options:A: Cy(-α); B: Cz(-β); C: Cy(α), format:choice, label:C2, answer:C}.
41. The matrix from the airflow frame to the stability frame is{ID:41, options:A: Cy(-α); B: Cz(-β); C: Cz(β), format:choice, label:C2, answer:B}.
42. The composite matrix from the airflow frame to the airframe is{ID:42, options:A: Cz(-β)*Cy(-α); B: Cy(α)*Cz(β); C: Cy(α)*Cz(-β); D: Cy(-α)*Cz(-β), format:choice, label:C2, answer:C}.
43. The unit airspeed in the airflow frame is{ID:43, options:none, format:vector, label:C1, answer:[1, 0, 0]}.
44. Transform the airspeed from the airflow frame into the airframe with{ID:44, options:A: bRa; B: bRa^T, format:choice, label:C2, answer:A}.
45. The unit airspeed in the airframe is{ID:45, options:none, format:vector, label:A2, answer:[cos(α)*cos(β), sin(β), sin(α)*cos(β)]}.
46. When the airframe airspeed component yb is positive, the airspeed points{ID:46, options:A: right; B: left, format:choice, label:A1, answer:A}.
47. When the airframe airspeed component zb is positive, the airspeed points{ID:47, options:A: upward; B: downward, format:choice, label:A1, answer:B}.
48. Recover sideslip from the airframe unit airspeed [vx, vy, vz] with{ID:48, options:A: atan2(vy, sqrt(vx^2+vz^2)); B: atan2(-vy, sqrt(vx^2+vz^2)); C: atan2(vz, sqrt(vx^2+vy^2)); D: atan2(vx, vy), format:choice, label:A2, answer:A}.
49. When vx>0 and vy>0, the sign of sideslip is{ID:49, options:A: negative; B: positive, format:choice, label:A1, answer:B}.
50. When vx>0 and vz>0, the sign of angle of attack is{ID:50, options:A: positive; B: negative, format:choice, label:A1, answer:A}.
51. A stable expression for angle of attack is{ID:51, options:A: atan2(vz, vx); B: atan2(-vz, vx); C: atan2(vy, vx), format:choice, label:A2, answer:A}.

### Subprocess 4: ground-frame and airframe closure

vg is the ground-frame unit airspeed from the track azimuth and the track inclination.
gRb and gRp map airframe and track-frame components into the ground frame. bRa maps airflow-frame components into the airframe, and pRa maps airflow-frame components into the track frame.
gRp contains no track roll. pRa contains only the roll of the airflow frame relative to the track frame about their shared airspeed axis.
The attitude angles, track angles, angle of attack, and sideslip in this subprocess describe one flight state, and numerical rounding is ignored. These angles cannot be chosen independently.
In that state, the airframe unit airspeed mapped into the ground frame by gRb equals the track-frame unit airspeed mapped into the ground frame by gRp.
The full attitude also satisfies gRb*bRa=gRp*pRa. Equality of the airspeed vectors checks only the shared airspeed axis, not the roll about that axis.

52. Both gRb and gRp map components of their child frames into{ID:52, options:A: the ground frame; B: the airflow frame, format:choice, label:C1, answer:A}.
53. After the track angles give the ground-frame airspeed, convert it into the airframe by{ID:53, options:A: gRb*vg; B: gRb^T*vg, format:choice, label:C2, answer:B}.
54. When angle of attack and sideslip give the airframe airspeed, γ and Φ{ID:54, options:A: do not enter the airframe airspeed formula directly; B: must be included, format:choice, label:C2, answer:A}.
55. The ground-frame airspeed built from the track angles is{ID:55, options:none, format:vector, label:A2, answer:[cos(φ)*cos(μ), sin(φ)*cos(μ), -sin(μ)]}.
56. After converting that airspeed into the airframe, it should match the airframe airspeed built directly from α and β{ID:56, options:A: generally unrelated; B: same; C: opposite, format:choice, label:A1, answer:B}.
57. When the angles describe one flight state and rounding is ignored, disagreement between the two paths means that the calculation or the use of the convention has{ID:57, options:A: certainly no error; B: at least one error, format:choice, label:A2, answer:B}.
58. Because γ does not change the ground coordinates of the unit airspeed, comparing airspeed vectors checks the shared airspeed axis and not the roll about it, so the airspeed direction can be checked by choosing to{ID:58, options:A: ignore the airspeed axis; B: compare only the roll angle; C: compare only the airspeed vectors, format:choice, label:C2, answer:C}.
59. Closing the unit vectors from the two paths requires the norm of their difference to be{ID:59, options:A: 2; B: 0; C: 1, format:choice, label:A2, answer:B}.

## Aerodynamic loads and the reference point

This section is sent only to the aerodynamic-load subprocess and the reference-point subprocess.

Fixed conventions
- Vectors are column vectors, and a matrix multiplies a column vector on the left.
- T_a←b maps the airframe components of one physical vector to airflow-frame components. The reverse transform is its transpose.
- T_s←b goes from the airframe to the stability frame, and T_a←s goes from the stability frame to the airflow frame. The declaration does not give the entries of these two matrices.
- All three frames are right-handed. Components are written in the order X, Y, Z.
- A force is an aerodynamic force acting on the aircraft. Aerodynamic forces and moments are signed components.
- Changing the coordinate expression of one point, and changing the moment reference point, are two different operations. Force and moment transforms both go from the airflow frame to the airframe, and the reference point stays at A.
- An angle enters a trigonometric function in one angle unit. This manuscript uses -90°<α<90° and -90°<β<90°, so cos(α)>0 and cos(β)>0.

Directions and axes
- Forward, right, and down are the positive axial, lateral, and normal directions. Forward × right = down.
- X_b points forward on the airframe, Y_b points right on the airframe, and Z_b points down on the airframe. The airframe origin is the aerodynamic reference point A.
- X_s is the projection of the airspeed onto the airframe plane of symmetry, Y_s has the same sense as Y_b, and Z_s is fixed by X_s×Y_s. The origin coincides with A.
- X_a follows the velocity of the aircraft relative to the air. Z_a lies in the airframe plane of symmetry, is perpendicular to X_a, and is the branch whose dot product with Z_b is positive. Y_a is fixed by Z_a×X_a. The origin coincides with A.
- Airspeed is the velocity of the aircraft relative to the air. The direction of the air relative to the aircraft is the opposite direction.

Airflow angles
- Angle of attack α: the angle between the projection of the airspeed onto the aircraft plane of symmetry and the positive airframe axis. When that projection lies below the airframe axis, the angle is positive.
- Sideslip β: the angle between the airspeed vector and the aircraft plane of symmetry. When the airspeed lies to the right of the aircraft, the angle is positive.
- Set β=0 when only angle of attack is analyzed. Set α=0 when only sideslip is analyzed.

Aerodynamic force and moment
- D is the drag magnitude, D≥0, along -X_a. Y is the signed side force, positive along +Y_a. L is the signed lift, positive along -Z_a.
- l_a, m_a, and n_a are positive by the right-hand rule about +X_a, +Y_a, and +Z_a.
- X, Y_b, and Z are positive along +X_b, +Y_b, and +Z_b. Y is the airflow side force, and Y_b is the airframe lateral force.
- l_A, m_A, and n_A are moments about A, positive by the right-hand rule about +X_b, +Y_b, and +Z_b.
- In the forward-right-down airframe, a positive pitch moment about +Y_b is nose-up, and a negative value is nose-down.

Dynamic pressure and coefficients
- ρ>0, V>0, and the dynamic pressure is q_bar=ρ*V^2/2. S>0 is the reference area, b>0 is the reference span, and c_bar>0 is the reference mean aerodynamic chord.
- D=q_bar*S*C_D, Y=q_bar*S*C_Ya, and L=q_bar*S*C_L.
- l_a=q_bar*S*b*C_la, m_a=q_bar*S*c_bar*C_ma, and n_a=q_bar*S*b*C_na.
- l_A=q_bar*S*b*C_lA, m_A=q_bar*S*c_bar*C_mA, and n_A=q_bar*S*b*C_nA.

Reference point
- A is the moment reference point of the original aerodynamic data. G is the center of mass and the moment reference point after the transfer. The airframe axes do not change during the transfer.
- N is the nose reference. N, A, and a tail reference are collinear on the airframe axis. L_f is the distance from N to the tail reference, L_A is the distance from A to the tail reference, and 0<L_A<L_f.
- X_N is the distance from A to N and is computed in the process.
- x_Ge is the axial projection from N rearward to G, and rearward is positive. y_Ge and z_Ge are measured from A along +Y_b and +Z_b.
- x_Ge is opposite +X_b. y_Ge and z_Ge have the same sense as the corresponding airframe components. The declaration does not give the coordinates of G relative to A, and it does not give the reference-point transfer formula.

### Subprocess 5: aerodynamic force and moment from the airflow frame to the airframe

60. The known aerodynamic components are in the airflow frame and the output is required in the airframe, so the transform is{ID:60, options:A: airflow frame to airframe; B: airframe to airflow frame; C: airflow frame to stability frame, then stop, format:choice, label:C2, answer:A}.
61. Across this subprocess, the moment reference point{ID:61, options:A: changes from A to G; B: changes from G to A; C: remains A, format:choice, label:C2, answer:C}.
62. The positive absolute airframe axis corresponds to{ID:62, options:A: Z_b; B: X_b; C: Y_b, format:choice, label:C1, answer:B}.
63. The positive absolute airframe lateral direction corresponds to{ID:63, options:A: Y_b; B: X_b; C: Z_b, format:choice, label:C1, answer:A}.
64. The positive absolute airframe normal corresponds to{ID:64, options:A: X_b; B: Y_b; C: Z_b, format:choice, label:C1, answer:C}.
65. Drag in the airflow frame is along{ID:65, options:A: -Z_a; B: -X_a; C: +X_a, format:choice, label:C1, answer:B}.
66. Positive lift in the airflow frame is along{ID:66, options:A: -Z_a; B: +Z_a; C: +Y_a, format:choice, label:C1, answer:A}.
67. Positive side force in the airflow frame is along{ID:67, options:A: -Y_a; B: +Z_a; C: +Y_a, format:choice, label:C1, answer:C}.
68. In the order X_a, Y_a, Z_a, the airflow-frame aerodynamic force is{ID:68, options:none, format:vector, label:A2, answer:[-D, Y, -L]}.
69. Positive airflow roll moment l_a is about{ID:69, options:A: +X_a; B: +Y_a; C: +Z_a, format:choice, label:C1, answer:A}.
70. Positive airflow pitch moment m_a is about{ID:70, options:A: +X_a; B: +Z_a; C: +Y_a, format:choice, label:C1, answer:C}.
71. In the order X_a, Y_a, Z_a, the airflow-frame moment is{ID:71, options:none, format:vector, label:A2, answer:[l_a, m_a, n_a]}.
72. With only a positive angle of attack, the stability forward axis relative to the airframe forward axis points{ID:72, options:A: downward; B: upward; C: right, format:choice, label:A1, answer:A}.
73. The Z_b component of that forward axis in the airframe is therefore{ID:73, options:A: negative; B: zero; C: positive, format:choice, label:A1, answer:C}.
74. The passive matrix T_s←b from the airframe to the stability frame is{ID:74, options:none, format:matrix, label:A2, answer:[[cos(α), 0, sin(α)], [0, 1, 0], [-sin(α), 0, cos(α)]]}.
75. With only a positive sideslip, the airflow forward axis relative to the stability forward axis points{ID:75, options:A: right; B: left; C: downward, format:choice, label:A1, answer:A}.
76. The Y_s component of that forward axis in the stability frame is therefore{ID:76, options:A: negative; B: zero; C: positive, format:choice, label:A1, answer:C}.
77. The passive matrix T_a←s from the stability frame to the airflow frame is{ID:77, options:none, format:matrix, label:A2, answer:[[cos(β), sin(β), 0], [-sin(β), cos(β), 0], [0, 0, 1]]}.
78. The full transform T_a←b from the airframe to the airflow frame is{ID:78, options:A: T_a←s*T_s←b; B: T_s←b*T_a←s, format:choice, label:C2, answer:A}.
Before the product is formed, write the expansion of each factor in the symbols just chosen.

79. The expanded T_a←s is{ID:79, options:none, format:matrix, label:A2, answer:[[cos(β), sin(β), 0], [-sin(β), cos(β), 0], [0, 0, 1]]}.
80. The expanded T_s←b is{ID:80, options:none, format:matrix, label:A2, answer:[[cos(α), 0, sin(α)], [0, 1, 0], [-sin(α), 0, cos(α)]]}.
81. Expanding that product gives T_a←b ={ID:81, options:none, format:matrix, label:A2, answer:[[cos(α)*cos(β), sin(β), sin(α)*cos(β)], [-cos(α)*sin(β), cos(β), -sin(α)*sin(β)], [-sin(α), 0, cos(α)]]}.
82. Convert the airflow-frame force into the airframe force with{ID:82, options:A: the transpose of T_a←b; B: T_a←b, format:choice, label:C2, answer:A}.
83. Call that reverse matrix B and expand it to{ID:83, options:none, format:matrix, label:A2, answer:[[cos(α)*cos(β), -cos(α)*sin(β), -sin(α)], [sin(β), cos(β), 0], [sin(α)*cos(β), -sin(α)*sin(β), cos(α)]]}.
84. The airframe axial force X is{ID:84, options:none, format:expression, label:A2, answer:-D*cos(α)*cos(β) - Y*cos(α)*sin(β) + L*sin(α)}.
Keep only lift. Set β=0, Y=0, D=0, L>0, and 0°<α<90°.

85. The airframe lateral force Y_b is{ID:85, options:none, format:expression, label:A2, answer:-D*sin(β) + Y*cos(β)}.
86. The airframe normal force Z is{ID:86, options:none, format:expression, label:A2, answer:-D*sin(α)*cos(β) - Y*sin(α)*sin(β) - L*cos(α)}.
87. The airframe axial component produced by lift is{ID:87, options:A: positive; B: negative; C: zero, format:choice, label:A1, answer:A}.
88. The airframe normal component produced by lift is{ID:88, options:A: positive; B: zero; C: negative, format:choice, label:A1, answer:C}.
89. When β=0, Y=0, D>0, and L>0, these conditions alone make the sign of the airframe axial force X{ID:89, options:A: certainly negative; B: indeterminate; C: certainly positive, format:choice, label:A1, answer:B}.
90. At α=30°, β=0°, D=2, Y=0, and L=2*sqrt(3), the airframe aerodynamic force is{ID:90, options:none, format:vector, label:A2, answer:[0, 0, -4]}.
91. After the coordinate transform, the Euclidean length of the aerodynamic force is{ID:91, options:A: generally increases; B: generally decreases; C: same, format:choice, label:A1, answer:C}.
92. At the same reference point A, the aerodynamic moment is moved from the airflow frame to the airframe with{ID:92, options:A: an extra moment-arm translation; B: the same matrix B; C: the negative of B, format:choice, label:C2, answer:B}.
93. The airframe roll moment l_A is{ID:93, options:none, format:expression, label:A2, answer:l_a*cos(α)*cos(β) - m_a*cos(α)*sin(β) - n_a*sin(α)}.
94. The airframe pitch moment m_A is{ID:94, options:none, format:expression, label:A2, answer:l_a*sin(β) + m_a*cos(β)}.
95. The airframe yaw moment n_A is{ID:95, options:none, format:expression, label:A2, answer:l_a*sin(α)*cos(β) - m_a*sin(α)*sin(β) + n_a*cos(α)}.
96. At β=0, the axis of the positive airflow pitch moment and the airframe lateral axis are{ID:96, options:A: aligned; B: perpendicular; C: direction indeterminate, format:choice, label:A1, answer:A}.
97. When α=0, 0°<β<90°, l_a>0, and m_a=n_a=0, the airframe pitch moment m_A is{ID:97, options:A: negative; B: zero; C: positive, format:choice, label:A1, answer:C}.
98. Dynamic pressure q_bar from ρ and V is{ID:98, options:none, format:expression, label:A2, answer:ρ*V^2/2}.
99. The dimensional airflow-frame force from the aerodynamic coefficients is{ID:99, options:none, format:vector, label:A2, answer:[-q_bar*S*C_D, q_bar*S*C_Ya, -q_bar*S*C_L]}.
100. The three force components share one scale, so the signed coefficient vector [-C_D, C_Ya, -C_L] can{ID:100, options:A: convert directly with the same matrix B; B: must be multiplied by different reference lengths before conversion, format:choice, label:C2, answer:A}.
This subprocess gives the resultant F_b=[X, Y_b, Z] in the airframe and the moment M_A=[l_A, m_A, n_A] about A.
Every component is already in the same airframe. This subprocess changes only the moment reference point and does not change the axis directions.

101. When the three moment coefficients use reference lengths b, c_bar, and b, and b is not equal to c_bar,{ID:101, options:A: multiply B directly by [C_la, C_ma, C_na]; B: recover the dimensional moment, rotate it, then normalize by the output-axis reference length, format:choice, label:C2, answer:B}.
102. Under the current normalization, the airframe pitch-moment coefficient C_mA is{ID:102, options:none, format:expression, label:A2, answer:(b/c_bar)*C_la*sin(β) + C_ma*cos(β)}.

### Subprocess 6: moment transfer from the aerodynamic reference point to the center of mass

103. Before the transfer, the moment reference point is{ID:103, options:A: G; B: N; C: A, format:choice, label:C1, answer:C}.
104. After the transfer, the moment reference point is{ID:104, options:A: N; B: G; C: A, format:choice, label:C1, answer:B}.
105. From the collinear lengths, the distance X_N from A to the nose reference N is{ID:105, options:none, format:expression, label:A2, answer:L_f - L_A}.
106. The positive engineering axial position x_Ge points{ID:106, options:A: rearward; B: forward, format:choice, label:C1, answer:A}.
107. That direction and the airframe +X_b direction are{ID:107, options:A: same; B: opposite, format:choice, label:A1, answer:B}.
108. The axial coordinate of the center of mass G in the airframe with origin A is therefore{ID:108, options:none, format:expression, label:A2, answer:X_N - x_Ge}.
109. In the order X_b, Y_b, Z_b, the vector r_AG from A to G is{ID:109, options:none, format:vector, label:A2, answer:[X_N - x_Ge, y_Ge, z_Ge]}.
110. The vector r_GA from G to A is{ID:110, options:none, format:vector, label:A2, answer:[x_Ge - X_N, -y_Ge, -z_Ge]}.
111. After the reference point moves from A to G, the resultant force of the same aerodynamic system{ID:111, options:A: stays unchanged; B: must be reversed; C: must gain a moment-arm term, format:choice, label:C2, answer:A}.
Below, r_AG is written [x, y, z].
This part checks only the direction of the added moment. Set x>0, y>0, and z>0, and keep one positive force component at a time. Every force component that is not named is zero.

112. To transfer the moment about A to the moment about G, the added moment is{ID:112, options:A: r_AG×F_b; B: r_GA×M_A; C: r_GA×F_b, format:choice, label:C2, answer:C}.
113. Using the vector r_AG from A to G, the transfer is{ID:113, options:A: M_G=F_b-r_AG×M_A; B: M_G=M_A-r_AG×F_b; C: M_G=M_A+r_AG×F_b, format:choice, label:A2, answer:B}.
114. With only a positive axial force X, the added pitch moment at G is{ID:114, options:A: negative; B: positive; C: zero, format:choice, label:A1, answer:A}.
115. With only a positive axial force X, the added yaw moment at G is{ID:115, options:A: negative; B: zero; C: positive, format:choice, label:A1, answer:C}.
116. With only a positive lateral force Y_b, the added roll moment at G is{ID:116, options:A: zero; B: positive; C: negative, format:choice, label:A1, answer:B}.
117. With only a positive lateral force Y_b, the added yaw moment at G is{ID:117, options:A: negative; B: positive; C: zero, format:choice, label:A1, answer:A}.
Restore every signed force component. Do not assume that they are positive.

118. With only a positive normal force Z, the added roll moment at G is{ID:118, options:A: positive; B: zero; C: negative, format:choice, label:A1, answer:C}.
119. With only a positive normal force Z, the added pitch moment at G is{ID:119, options:A: zero; B: positive; C: negative, format:choice, label:A1, answer:B}.
120. The roll moment l_G transferred to the center of mass is{ID:120, options:none, format:expression, label:A2, answer:l_A + z*Y_b - y*Z}.
121. The pitch moment m_G transferred to the center of mass is{ID:121, options:none, format:expression, label:A2, answer:m_A + x*Z - z*X}.
122. The yaw moment n_G transferred to the center of mass is{ID:122, options:none, format:expression, label:A2, answer:n_A + y*X - x*Y_b}.
123. After substituting the engineering position, the pitch moment m_G at the center of mass is{ID:123, options:none, format:expression, label:A2, answer:m_A + (X_N - x_Ge)*Z - z_Ge*X}.
124. If G is offset from A only forward on the airframe, so x>0 and y=z=0, and the force is only upward on the airframe, the added pitch moment is{ID:124, options:A: nose-up; B: zero; C: nose-down, format:choice, label:A1, answer:C}.
125. If r_AG is parallel to F_b, the moment added by the reference-point transfer is{ID:125, options:A: certainly negative; B: zero; C: certainly positive, format:choice, label:A1, answer:B}.
126. If G moves from A only along the airframe axis, the roll moments l_G and l_A are{ID:126, options:A: same; B: necessarily opposite; C: no relation can be formed, format:choice, label:A1, answer:A}.
127. If every reference length, reference area, and dynamic pressure stays fixed, and [X, Y_b, Z]=q_bar*S*[C_X, C_Yb, C_Z], the roll-moment coefficient C_lG at the center of mass is{ID:127, options:none, format:expression, label:A2, answer:C_lA + (z/b)*C_Yb - (y/b)*C_Z}.
128. Under the same convention, the pitch-moment coefficient C_mG at the center of mass is{ID:128, options:none, format:expression, label:A2, answer:C_mA + (x/c_bar)*C_Z - (z/c_bar)*C_X}.
129. Under the same convention, the yaw-moment coefficient C_nG at the center of mass is{ID:129, options:none, format:expression, label:A2, answer:C_nA + (y/b)*C_X - (x/b)*C_Yb}.
130. If one aerodynamic system is both rotated from the airflow frame into the airframe and transferred from A to G, the reference-point transfer may be done first in the airflow frame, but the same moment arm must also be expressed in{ID:130, options:A: the airframe without converting it; B: an arbitrary frame; C: the airflow frame, format:choice, label:C2, answer:C}.
131. In this right-handed orthogonal frame, if the force, moment, and arm agree on both paths, the final airframe moment at the center of mass from rotating first and from transferring the point first is{ID:131, options:A: generally unrelated; B: same; C: negatives of each other, format:choice, label:A2, answer:B}.

# Generation notes

This section is not sent to the model. The elementary rotations are passive. gRb maps airframe components into the ground frame and equals the transpose of bRg. For this item gRb=Cz(-ψ)*Cy(-θ)*Cx(Φ), bRg=Cx(-Φ)*Cy(θ)*Cz(ψ). The track frame has no roll, gRp=Cz(-φ)*Cy(-μ), pRa=Cx(-γ), and gRa=gRp*pRa. bRs=Cy(α), bRa=Cy(α)*Cz(-β). After an angle sign changes, these factors, the airspeed components, and the aerodynamic matrices are recomputed together. The declaration does not write gRb, gRp, bRs, bRa, or the unit-airspeed components, so a derived result cannot be copied from it.

The aerodynamic loads use one orthogonal basis. T_i←j maps components of frame j into frame i, and its rows are the axes of frame i written in frame j. The declaration does not write the entries of T_s←b or T_a←s. At the standard angle of attack, the side-force term in the airframe normal force is -Y*sin(α)*sin(β). The reference-point transfer in right-handed components uses M_G=M_A-r_AG×F_b, and the declaration does not write that formula.

The checks are: the airspeed axes close in one flight state, and one track roll makes gRb*bRa=gRp*pRa; the airflow basis is orthogonal and right-handed; the product of the two stages equals the full transform; the force and the moment at one point keep their lengths; the reference-length scale of a moment coefficient agrees with the dimensional moment; the reference-point transfer can be reversed; rotating first and transferring the point first agree.


# 会话分叉：机器视觉物体与双目 ID41

样本 scene/sample_01，走到 ID41。模型 deepseek-flash，思考 enabled，强度 low。
核槽 ['37', '38', '39', '40']。对照 session-model；暂不分叉。

计划 3 趟，完成 3：待分叉 3，残留 0，干净 0，失败 0。

| 槽 | 对照答错 |
| --- | ---: |
| ID37 | 3/3 = 100% |
| ID38 | 3/3 = 100% |
| ID39 | 3/3 = 100% |
| ID40 | 3/3 = 100% |
| ID41 | 3/3 = 100% |

核槽全对时 ID41 仍错：—（没有核槽全对的趟）。

| 趟 | 状态 | 错的核槽 | 对照 ID41 |
| ---: | --- | --- | --- |
| 1 | 待分叉 | ID37、ID38、ID39、ID40 | [[-sin(ψ1)*cos(ψ1)*cos(φ1)+sin(φ1)*(cos(ψ1)*cos(θ1)+sin(θ1)*sin(ψ1)^2), sin(ψ1)*cos(ψ1)*sin(φ1)+cos(φ1)*(cos(ψ1)*cos(θ1)+sin(θ1)*sin(ψ1)^2), -sin(θ1)*cos(ψ1)+cos(θ1)*sin(ψ1)^2], [cos(φ1)*(-sin(θ1)*cos(ψ1)^2+cos(θ1)*sin(ψ1))+sin(φ1)*(-cos(θ1)*sin(θ1)*sin(ψ1)+sin(θ1)*cos(ψ1)*(sin(θ1)*sin(ψ1)+cos(θ1))), -sin(φ1)*(-sin(θ1)*cos(ψ1)^2+cos(θ1)*sin(ψ1))+cos(φ1)*(-cos(θ1)*sin(θ1)*sin(ψ1)+sin(θ1)*cos(ψ1)*(sin(θ1)*sin(ψ1)+cos(θ1))), sin(θ1)^2*sin(ψ1)+cos(θ1)*cos(ψ1)*sin(θ1)*sin(ψ1)+cos(θ1)^2*cos(ψ1)], [cos(φ1)*(cos(θ1)*cos(ψ1)^2+sin(θ1)*sin(ψ1))+sin(φ1)*(cos(θ1)^2*sin(ψ1)+sin(θ1)*cos(ψ1)*(sin(θ1)-cos(θ1)*sin(ψ1))), -sin(φ1)*(cos(θ1)*cos(ψ1)^2+sin(θ1)*sin(ψ1))+cos(φ1)*(cos(θ1)^2*sin(ψ1)+sin(θ1)*cos(ψ1)*(sin(θ1)-cos(θ1)*sin(ψ1))), -sin(θ1)*cos(θ1)*sin(ψ1)+cos(θ1)*cos(ψ1)*sin(θ1)-cos(ψ1)*cos(θ1)^2*sin(ψ1)]] |
| 2 | 待分叉 | ID37、ID38、ID39、ID40 | [[-sin(φ1), cos(θ1)*cos(φ1), sin(θ1)*cos(φ1)], [sin(ψ1)*cos(φ1), cos(θ1)*sin(ψ1)*sin(φ1) - sin(θ1)*cos(ψ1), sin(θ1)*sin(ψ1)*sin(φ1) + cos(θ1)*cos(ψ1)], [cos(ψ1)*cos(φ1), cos(θ1)*cos(ψ1)*sin(φ1) + sin(θ1)*sin(ψ1), sin(θ1)*cos(ψ1)*sin(φ1) - cos(θ1)*sin(ψ1)]] |
| 3 | 待分叉 | ID37、ID38、ID39、ID40 | [[-sin(φ1)*cos(θ1), cos(φ1)*cos(θ1), sin(θ1)], [sin(ψ1)*cos(φ1)+cos(ψ1)*sin(θ1)*sin(φ1), sin(ψ1)*sin(φ1)-cos(ψ1)*sin(θ1)*cos(φ1), cos(ψ1)*cos(θ1)], [cos(ψ1)*cos(φ1)-sin(ψ1)*sin(θ1)*sin(φ1), cos(ψ1)*sin(φ1)+sin(ψ1)*sin(θ1)*cos(φ1), -sin(ψ1)*cos(θ1)]] |

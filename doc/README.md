# 契约条件化过程评测

四个领域：风洞、飞行力学、机器视觉、机器人运动学。

设计定稿按这三份读，不要再从过程笔记回推：

- [全流程总结](全流程总结.md)：阶段 0–7。D 矩阵是阶段 1 的审计结果，不参与阶段 5–7。
- [核心设计形式化](核心设计形式化.md)：任务分解、生成、故障库、五态、因果/关联、反事实。
- [评价指标](评价指标.md)：设计层、诊断层、因果层。

诊断主体是图递归祖先加局部重算。注入对象是模型，不是确定性系统。反事实在阶段 7，系统侧重算和模型侧会话分叉分开报。

常用入口：

```text
python analysis/validate_pipeline.py
python analysis/run_dmatrix.py
python cloze_eval.py --inventory
python cloze_eval.py --domain wind_tunnel --condition session-model --model deepseek-flash --thinking disabled --runs 1 --out results/validate/wind_tunnel
python cloze_eval.py --domain flight --condition session-model --model deepseek-flash --thinking disabled --runs 1 --out results/validate/flight
python cloze_eval.py --domain cv --condition session-model --model deepseek-flash --thinking disabled --runs 1 --out results/validate/cv
python cloze_eval.py --domain robotics --condition session-model --model deepseek-flash --thinking disabled --runs 1 --out results/validate/robotics
```

`cloze_eval` 默认评主实验：每个领域 1 条标准定义和 9 条单定义扰动。题面、定义声明和选项是英文。每条扰动稿写明 `Single-definition perturbation`，机器可读字段是 `factor`。风洞在 `data/wind_tunnel/single/samples.jsonl`，标准行是 000-001，factor 为 `none`。飞行力学在 `data/flight/review/flow`，01 是 common default，02–10 各改一项定义；气动力和参考点仍单独注入。机器视觉在 `data/vision/single`，只采用物体与双目长篇幅稿，一次注入，不含基础理论。机器人学在 `data/robotics/single`，01 是 upright wrist；正运动学和逆运动学分开送入。分档批量、两段视觉旧稿和同时改多个关节的稿仍留在原目录，不作为主实验入口。

数据在 `data/`，生成与诊断在 `domains/`。

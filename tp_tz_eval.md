# 天平系到体轴系评测答案

## 步骤 1

- ID 1 [C2] 力和力矩的坐标系转换方向（体轴系向体轴系/体轴系向天平系/体轴系向风轴系/天平系向体轴系/天平系向天平系/天平系向风轴系/风轴系向体轴系/风轴系向天平系/风轴系向风轴系）：天平系向体轴系
## 步骤 2

- ID 2 [C1] 天平安装角以天平绝对轴向正方向（前方）的右手定则为（正/负）：正
- ID 3 [C1] 机体滚转角以机体绝对轴向正方向（前方）的右手定则为（正/负）：正
- ID 4 [A1] 天平系与机体系相差的滚转角：γ_M - γ_tp
- ID 5 [A1] 安装角旋转所绕的天平坐标轴（XFD/YFD/ZFD/XTZ/YTZ/ZTZ/XTP/YTP/ZYP）：XTP
- ID 6 [A1] 该滚转相对这根坐标轴正方向的转向（正/逆）：正
- ID 7 [A2] 天平轴系中的安装角旋转矩阵 R_(γ_M-γ_tp)：[[1, 0, 0], [0, cos(γ_M - γ_tp), sin(γ_M - γ_tp)], [0, -sin(γ_M - γ_tp), cos(γ_M - γ_tp)]]
## 步骤 3

- ID 8 [C1] 天平系绝对法向正方向对应的有向坐标轴（XFD/YFD/ZFD/XTZ/YTZ/ZTZ/XTP/YTP/ZYP/-XFD/-YFD/-ZFD/-XTZ/-YTZ/-ZTZ/-XTP/-YTP/-ZYP）：YTP
- ID 9 [C1] 天平系绝对轴向正方向对应的有向坐标轴（XFD/YFD/ZFD/XTZ/YTZ/ZTZ/XTP/YTP/ZYP/-XFD/-YFD/-ZFD/-XTZ/-YTZ/-ZTZ/-XTP/-YTP/-ZYP）：XTP
- ID 10 [C1] 天平系绝对侧向正方向对应的有向坐标轴（XFD/YFD/ZFD/XTZ/YTZ/ZTZ/XTP/YTP/ZYP/-XFD/-YFD/-ZFD/-XTZ/-YTZ/-ZTZ/-XTP/-YTP/-ZYP）：ZYP
- ID 11 [C1] 机体系绝对法向正方向对应的有向坐标轴（XFD/YFD/ZFD/XTZ/YTZ/ZTZ/XTP/YTP/ZYP/-XFD/-YFD/-ZFD/-XTZ/-YTZ/-ZTZ/-XTP/-YTP/-ZYP）：YTZ
- ID 12 [C1] 机体系绝对轴向正方向对应的有向坐标轴（XFD/YFD/ZFD/XTZ/YTZ/ZTZ/XTP/YTP/ZYP/-XFD/-YFD/-ZFD/-XTZ/-YTZ/-ZTZ/-XTP/-YTP/-ZYP）：XTZ
- ID 13 [C1] 机体系绝对侧向正方向对应的有向坐标轴（XFD/YFD/ZFD/XTZ/YTZ/ZTZ/XTP/YTP/ZYP/-XFD/-YFD/-ZFD/-XTZ/-YTZ/-ZTZ/-XTP/-YTP/-ZYP）：ZTZ
- ID 14 [A2] 坐标轴与绝对方向统一后的前置矩阵 RTP_TZ1：[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
## 步骤 4

- ID 15 [C1] 模型安装方式（正装/倒装）：正装
- ID 16 [A2] 正装或倒装对应的前置矩阵 RTP_TZ2：[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
## 步骤 5

- ID 17 [A2] 连乘式中的 R1（R_(γ_M-γ_tp)/RTP_TZ1/RTP_TZ2）：RTP_TZ2
- ID 18 [A2] 连乘式中的 R2（R_(γ_M-γ_tp)/RTP_TZ1/RTP_TZ2）：RTP_TZ1
- ID 19 [A2] 连乘式中的 R3（R_(γ_M-γ_tp)/RTP_TZ1/RTP_TZ2）：R_(γ_M-γ_tp)
## 步骤 6

- ID 20 [A2] R1*R2*R3 化简后的总矩阵：[[1, 0, 0], [0, cos(γ_M - γ_tp), sin(γ_M - γ_tp)], [0, -sin(γ_M - γ_tp), cos(γ_M - γ_tp)]]
## 步骤 7

- ID 21 [C1] XTZ 关联的体轴力和力矩（CN_TZ,MN_TZ/CA_TZ,MA_TZ/CZ_TZ,MZ_TZ）：CA_TZ,MA_TZ
- ID 22 [C1] YTZ 关联的体轴力和力矩（CN_TZ,MN_TZ/CA_TZ,MA_TZ/CZ_TZ,MZ_TZ）：CN_TZ,MN_TZ
- ID 23 [C1] ZTZ 关联的体轴力和力矩（CN_TZ,MN_TZ/CA_TZ,MA_TZ/CZ_TZ,MZ_TZ）：CZ_TZ,MZ_TZ
- ID 24 [C1] XTP 关联的天平力和力矩（CN_TP,MN_TP/CA_TP,MA_TP/CZ_TP,MZ_TP）：CA_TP,MA_TP
- ID 25 [C1] YTP 关联的天平力和力矩（CN_TP,MN_TP/CA_TP,MA_TP/CZ_TP,MZ_TP）：CN_TP,MN_TP
- ID 26 [C1] ZYP 关联的天平力和力矩（CN_TP,MN_TP/CA_TP,MA_TP/CZ_TP,MZ_TP）：CZ_TP,MZ_TP
## 步骤 8

- ID 27 [A2] 体轴系 CN_TZ：CN_TP*cos(γ_M - γ_tp) + CZ_TP*sin(γ_M - γ_tp)
- ID 28 [A2] 体轴系 CA_TZ：CA_TP
- ID 29 [A2] 体轴系 CZ_TZ：-CN_TP*sin(γ_M - γ_tp) + CZ_TP*cos(γ_M - γ_tp)
- ID 30 [A2] 体轴系 MN_TZ：MN_TP*cos(γ_M - γ_tp) + MZ_TP*sin(γ_M - γ_tp)
- ID 31 [A2] 体轴系 MA_TZ：MA_TP
- ID 32 [A2] 体轴系 MZ_TZ：-MN_TP*sin(γ_M - γ_tp) + MZ_TP*cos(γ_M - γ_tp)

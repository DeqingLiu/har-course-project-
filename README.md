# 手机传感器人体活动识别课程项目

比较三种方法（逻辑回归、一维 CNN、随机森林）与两种传感器组合，回答：加入陀螺仪是否改善分类？在相同统计特征上，非线性分类器能否缩小与 CNN 的差距？

已完成六组真实实验、十四份最终模型及六页 PDF。仅交付 PDF，不交付 Word。姓名：雷煜；学号：2026439081。项目仓库：[DeqingLiu/har-course-project-](https://github.com/DeqingLiu/har-course-project-)。身份信息保存于 `report_identity.json`，报告脚本默认读取，可用命令行参数覆盖。未发送提交邮件。

GitHub 仓库提供源码、复现说明、正式 PDF，以及 `published_results/` 中的结果摘要和图表。原始数据、十四份权重和完整逐窗口预测位于本地完整材料包，未上传 Git；新克隆目录请按下方“在空目录从头复现”运行。`outputs/` 的既有记录核验命令仅适用于完整包或已完成训练的目录。公开结果摘要放在独立目录，不会被训练入口误认为已冻结实验。

## 阅读入口

- `submission/课程设计报告.pdf`：新版六页正式报告。
- `submission/2026439081-雷煜.pdf`：内容相同的提交命名版本。
- `submission/archive/课程设计报告_四组实验原版.pdf`：原版仅保留在本地完整材料包。
- `中文讲解与复现.md`：方法解释、负面结果和答辩要点。
- `published_results/summary.csv`：GitHub 中的六组结果摘要；本地完整证据位于 `outputs/expanded/`。
- `验收记录.md`：模型、指标、排版和打包核验记录。

## 实际测试结果

| 组别 | 方法 | 通道 | 准确率 % | Macro-F1 % |
|---|---|---:|---:|---:|
| A | 逻辑回归 | 3 | 80.32 | 79.99 |
| B | 逻辑回归 | 6 | 85.04 | 84.76 |
| C | 一维 CNN | 3 | 83.92 ± 2.31 | 83.80 ± 2.37 |
| D | 一维 CNN | 6 | 88.56 ± 1.76 | 88.44 ± 1.74 |
| E | 随机森林 | 3 | 75.96 ± 0.13 | 75.58 ± 0.13 |
| F | 随机森林 | 6 | 81.07 ± 0.12 | 80.68 ± 0.15 |

三通道是总加速度；六通道再加入三轴角速度。± 为种子 42、43、44 的样本标准差，不是置信区间。三种方法加入陀螺仪的 Macro-F1 增益依次为 4.77、4.64、5.11 个百分点。随机森林没有缩小与 CNN 的差距；六通道随机森林训练得分 99.92%，测试 80.68%，提示跨人泛化仍不足，但不能仅凭差距确定唯一原因。

## 核验现有完整项目

在 PowerShell 中运行：

```powershell
cd D:\数据挖掘\har-course-project
& 'D:/Huawei Cup/.venv/Scripts/python.exe' har_rf.py validate
```

该命令需要本地完整材料包或已完成训练的目录；它检查原始与新增冻结清单、相同受试者划分、四配置的验证均值选优，重新加载全部十四个模型并复核预测。本次通过 76 项检查。

完整扩充入口如下。已有冻结权重和结果时，跳过训练、首次评价，重新核验模型并生成图表及训练/验证/测试对照：

```powershell
.\run_expanded.ps1
```

不要在这个已完成目录中重跑原版 `har.py evaluate` 或 `har.py validate`：它们会重写原实验记录，而新增清单保存了这些文件的原始校验值。`run.ps1` 已增加相应保护。不要编辑 `har.py`、`config.json`、`har_rf.py`、`config_rf.json`、数据、选模记录或权重。报告和说明文档可独立更新。

## 在空目录从头复现

解压源码 ZIP 到新的空目录，或克隆仓库后进入项目目录：

```powershell
git clone https://github.com/DeqingLiu/har-course-project-.git
cd har-course-project-
```

然后创建环境并依次运行原实验和补充实验：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\run.ps1 -Python .\.venv\Scripts\python.exe -Stage all
.\run_expanded.ps1 -Python .\.venv\Scripts\python.exe
```

先完成 A–D，再进行 E–F。随机森林拟合 24 个候选模型，保留验证均值最优配置的六个权重，加上原有八个模型，共十四份最终记录。未选候选权重仅在 `tmp/rf_candidates` 暂存，不作为最终模型打包；候选参数和分数全部保留。

依赖见 `requirements.txt`，实际版本见原版及扩充环境记录。本机使用 Python 3.12.10、scikit-learn 1.9.1、PyTorch 2.14.0+cu130 和 RTX 5060；CNN 自动选择可用 CUDA，随机森林在 CPU 上运行。新电脑安装 CUDA 版本时请按 PyTorch 官方说明匹配驱动。不同硬件/版本的重训可能产生数值差异，不能保证逐位相同；本次验收是保存权重的可重载性与指标一致性。

官方数据下载约 58 MB，脚本处理内外两层 ZIP，并保存本地 SHA-256 和 CRC 检查。这是本地完整性记录，不是官方独立校验值。下载失败可重新运行 prepare。不要删除完整实验的冻结记录来重训，应使用新的源码目录。

## 实验协议

- 官方训练 7,352 个窗口、测试 2,947 个。验证受试者固定为 3、16、22、23，最终训练/验证/测试为 5,952/1,400/2,947 个窗口，17/4/9 人互不重叠。
- 输入只使用 total_acc 和可选 body_gyro，不使用 body_acc 或官方 561 维特征。
- 统计量为每通道均值、总体标准差、最小值、最大值、中位数、四分位距、均方根，按统计量再按通道拼接。两种传统方法复用相同的 21/42 维特征。
- 逻辑回归的标准化器仅拟合训练集，C 从 0.1、1、10 中按验证 Macro-F1 选择，平局选较小值。
- CNN 三层卷积通道数 64/128/128、核宽 7/5/3，前两层最大池化 2，随后全局平均池化、Dropout 0.3、六分类层。ReLU，无 BatchNorm、增强或预训练。通道标准化仅使用训练集。
- CNN 使用 Adam、学习率 0.001、批大小 64、最多 60 轮；验证 Macro-F1 连续 10 轮不改善则早停，保存验证最优权重。
- 随机森林直接使用未标准化统计特征，固定 300 棵树、Gini、max_features="sqrt"、bootstrap=True、class_weight=None。深度 {12, None} × 最小叶样本数 {3, 1}，每配置种子 42/43/44。按验证均值选优，完全相同时优先深度 12，再优先叶样本数 3。
- E 选中深度不限、叶样本数 3，验证均值 86.74%；F 选中深度 12、叶样本数 1，验证均值 87.57%。选中模型直接评价，不合并训练和验证重训。
- 原 A–D 测试结果在补充随机森林前已查看。新增协议先固定搜索范围，再验证选模、冻结、评价测试；没有根据新测试成绩追加调参。报告如实披露这种后续比较的局限。
- CNN 和随机森林各使用三个种子；图示和案例固定种子 42。病例选择沿用 C/D 的原规则，再添加 F 预测，不按新方法的好坏重新筛选。

## 代码与证据位置

| 路径 | 内容 |
|---|---|
| `har.py`、`config.json` | 保持原样的 A–D 代码与协议 |
| `har_rf.py train/evaluate/validate`、`config_rf.json` | 扩充实验独立入口与固定搜索范围 |
| `outputs/frozen_manifest.json`、`outputs/runs/` | 原八个模型与冻结记录，保持不变 |
| `outputs/expanded/protocol.json`、`original_snapshot.json` | 新协议与原实验快照，均位于 expanded 中 |
| `outputs/expanded/search/` | 两种输入的全部候选参数、各种子验证分数 |
| `outputs/expanded/runs/`、`frozen_manifest.json` | 新增六个模型及独立冻结清单 |
| `outputs/expanded/predictions/` | 十四份逐窗口预测与概率，每份 2,947 行 |
| `outputs/expanded/results.json`、`summary.csv`、`seed_results.csv` | 逐模型结果、六组汇总及逐种子明细 |
| `outputs/expanded/per_class.csv`、`per_subject.csv` | 类别指标和受试者表现 |
| `outputs/expanded/generalization*.csv`、`generalization.json` | 保存模型的训练/验证/测试描述性对照，不用于重新选模 |
| `outputs/expanded/figures/`、`analysis.json` | 实际结果图、传感器增益、类别汇总与案例 |
| `outputs/expanded/validation.json`、`delivery_audit.json` | 76 项模型复核与 295 项独立交付检查 |
| `data/split_manifest.json` | 受试者划分、标签分布与数据校验值 |

表格中只列文件名的配对文件与该行第一个文件位于同一目录。`joblib` 只加载本项目可信且通过校验的模型。CNN 使用 weights_only=True 读取权重。

`make_figures.py`、`analyze_expanded.py`、`make_report.py`、`audit_project.py` 支持 `--results-dir outputs/expanded`。新版报告需要六组完整数据；原四组 PDF 已归档。

## 重新生成报告与打包

本机文档环境独立于训练环境：

```powershell
$docPython = 'C:/Users/Leiyu/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $docPython make_report.py --results-dir outputs/expanded
& $docPython render_report.py
& $docPython audit_project.py --results-dir outputs/expanded
& $docPython package_project.py
```

只有已经公开的真实仓库才传 `--github '真实链接'`。其他 Windows 电脑安装 `requirements-documents.txt`，需要 Microsoft Word 导出和 Poppler 页图工具；当前机器使用 Codex 内置渲染工具。排版要求安装黑体（SimHei）、宋体（SimSun）、Times New Roman：中文标题黑体，中文正文宋体，数字及英文字母 Times New Roman，图内文字同样区分中西文字体。临时 DOCX 仅作排版中间文件，位于 tmp，绝不进入交付包。最终 PDF 必须重新检查六页且无遮挡，修改姓名后也应重新检查并打包。

`dist/har-course-project-complete.zip` 包含离线数据、十四份最终权重、预测、图表、正式和归档 PDF。`dist/har-course-project-source.zip` 包含代码、配置和说明，适合整理上传 GitHub。两个包不包含临时文件、虚拟环境或 Word；CRC、大小、文件数和 SHA-256 记录在 `dist/package_manifest.json`。

## 来源与提交

数据、论文和工具出处见 `SOURCES.md`。使用 Codex AI 辅助代码、实验、图表和报告，数值均来自实际本地运行；报告已披露。理解方法仍需本人阅读和讲解。

截止日期为 2026 年 12 月 20 日。提交命名版本已生成，提交前核对姓名学号并按课程要求自行提交。代码仓库与提交报告不能代替课程要求的正式提交，本项目没有发送邮件。

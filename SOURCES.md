# 数据 算法与软件来源

原资料核对日期：2026-10-26。随机森林扩展资料核对日期：2026-11-02。

1. Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X. (2013). Human Activity Recognition Using Smartphones [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C54S4K
   - 官网：https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
   - 下载地址记录于 `data/raw/source.json` 与 `config.json`。
   - 使用官方预处理时序窗口、标签及受试者编号；未使用官方 561 维特征。
   - 当前 UCI 页面标注 CC BY 4.0；压缩包中 2013 年旧 README 仍写有禁止商业用途的文字。本项目保留原始说明，仅用于课程研究；公开代码包不附原始大数据，数据由下载脚本获取。这里没有对两个版本许可的差异作法律判断。
2. Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013). A Public Domain Dataset for Human Activity Recognition Using Smartphones. ESANN, 437–442.
   - 原论文：https://www.esann.org/sites/default/files/proceedings/legacy/es2013-84.pdf
3. PyTorch 官方 API 与复现文档。
   - https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
   - https://docs.pytorch.org/docs/stable/notes/randomness.html
   - 使用 Conv1d、ReLU、MaxPool1d、AdaptiveAvgPool1d、Dropout、Linear、Adam、CrossEntropyLoss。
4. scikit-learn 官方 API。
   - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
   - 使用 StandardScaler、LogisticRegression、Pipeline 与评价指标。
5. NumPy、SciPy、Matplotlib、joblib 用于数值处理、图表与模型保存；python-docx、Microsoft Word 与 Poppler 用于本地报告排版、导出及视觉检查。

实验脚本为本项目实现，不声称提出新的网络结构或新的基础算法。Codex AI 辅助编写代码、执行实验、制作图表和整理报告；训练权重、预测和数值均由本地真实运行产生。报告末尾包含相同的工具使用披露。

## 随机森林扩展

- scikit-learn RandomForestClassifier 官方文档：https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- 使用随机森林对同一套 21/42 维统计特征分类。300 棵树、四种深度/叶节点组合、三个随机种子在本轮训练前固定；参数仅依据验证集选择。
- 这是在首轮 A–D 测试结果已查看之后开展的补充比较。报告如实说明这一研究过程，并未将同一测试集描述为从未查看的全新独立验证。

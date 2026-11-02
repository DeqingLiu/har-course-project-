"""Six-page content for the expanded comparison; all numerical claims use artifacts."""
from pathlib import Path
import json
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt


def add_content(doc,p,heading,table,caption,picture,args,out,results,analysis,split,audit):
    load=lambda path:json.loads(path.read_text(encoding="utf-8"))
    agg={r['experiment']:r for r in results['aggregate']}
    runs={(r['experiment'],r['seed']):r for r in results['runs']}
    cls={(r['experiment'],r['activity']):r['f1_mean'] for r in analysis['class_summary']}
    gen={r['experiment']:r for r in load(out/'generalization.json')['aggregate']}
    cfg=load(out.parent.parent/'config_rf.json')
    names={'logistic':'逻辑回归','cnn':'一维 CNN','random_forest':'随机森林'}
    pct=lambda x:f'{100*x:.2f}'
    diff=lambda a,b:100*(agg[a]['macro_f1_mean']-agg[b]['macro_f1_mean'])
    expanded='F' in agg
    assert expanded, 'Expanded report requires six-group data; the original PDF is archived.'
    seeds=cfg['seeds']
    best=max(agg,key=lambda e:agg[e]['macro_f1_mean'])
    title=doc.add_paragraph('基于一维卷积神经网络的人体活动识别','Title')
    title.alignment=WD_ALIGN_PARAGRAPH.CENTER
    sub=p('传感器组合与传统方法的对比研究');sub.style=doc.styles['Subtitle'];sub.alignment=WD_ALIGN_PARAGRAPH.CENTER
    for r in sub.runs:r.font.size=Pt(12)
    p(f'姓名：{args.name}    学号：{args.student_id}    课程：深度学习课程设计',True)
    p(f'GitHub：{args.github}',True)
    heading('摘要',level=2)
    p(f'本文在 UCI HAR 数据集上比较逻辑回归、随机森林与一维卷积神经网络（CNN），分别使用三轴总加速度和加速度加陀螺仪，以研究传感器融合及非线性统计特征分类的作用。采用互不重叠的训练、验证、测试受试者划分；CNN 与随机森林各使用三个种子。六通道 CNN 的测试 Macro-F1 为 {pct(agg["D"]["macro_f1_mean"])}%±{pct(agg["D"]["macro_f1_std"])}%，随机森林为 {pct(agg["F"]["macro_f1_mean"])}%±{pct(agg["F"]["macro_f1_std"])}%。加入陀螺仪对三种方法均有总体收益，但随机森林未缩小与 CNN 的差距。类别分析与训练测试差异表明，模型复杂度增加并不保证跨人泛化改善。')
    p('关键词：人体活动识别；一维卷积；随机森林；传感器消融；跨受试者评价',True)
    heading('1 研究问题与实验目标')
    p('手机加速度计包含线性运动与重力方向信息，陀螺仪补充角速度信息。本文将应用范围限定为腰部佩戴手机时的六类活动识别，研究两个问题：加入陀螺仪是否改善分类；当传统分类器从线性逻辑回归换为非线性随机森林时，与 CNN 的差距是否缩小。')
    p('逻辑回归与随机森林共享相同统计特征，使二者的差异主要体现分类器选择。CNN 直接学习窗口内的时间模式。该设计能区分“加入传感器”和“更换方法”的实验效果，但不能把 CNN 的全部优势归因于某一单独结构，因为表示、优化与模型容量也同时不同。')
    table(['组别','输入','方法','统计特征维度'],[[e,'总加速度' if a['channels']==3 else '总加速度＋角速度',names[a['method']],str(a['channels']*7) if a['method']!='cnn' else '直接输入时序'] for e,a in agg.items()],[1.2,5.7,4.5,4.9])
    p('比较 A→B、C→D、E→F 分析传感器增益；比较 A 与 E、B 与 F 分析非线性分类器是否改善相同特征；再在相同输入下比较 CNN。全部方法使用同一批窗口和标签，不额外使用官方 561 维特征。',True)

    heading('2 数据来源与实验环境',True)
    p('UCI HAR 数据来自 30 名年龄为 19 至 48 岁的志愿者，使用腰部佩戴的 Samsung Galaxy S II 手机采集[1–2]。加速度计与陀螺仪采样率均为 50 Hz；官方完成滤波后，以 2.56 秒窗口、50% 重叠率切分，每个窗口为 128 个时间点。类别为步行、上楼、下楼、坐姿、站姿和躺卧。本研究使用公开的预处理窗口，并非自行采集的连续原始信号。')
    table(['集合','人数','窗口数','用途'],[[label,len(split['splits'][key]['subjects']),split['splits'][key]['n'],use] for key,label,use in [('train','训练','拟合参数与预处理统计量'),('val','验证','选择参数配置和最佳训练轮'),('test','测试','选模冻结后的统一评价')]],[2,1.5,2,10.8])
    val_ids='、'.join(map(str,split['splits']['val']['subjects']))
    p(f'保留官方测试受试者。从官方训练受试者中使用 NumPy default_rng(42) 无放回抽取 4 人验证，编号为 {val_ids}。训练、验证、测试受试者互不重叠，避免同一人的相邻重叠窗口进入不同集合。若随机按行划分，模型可能利用高度相似的窗口获得偏乐观分数。',True)
    picture('data_overview')
    caption('图 1 各集合的类别分布与训练集步行窗口的三轴总加速度')
    heading('2.1 输入与预处理',level=2)
    p('三通道使用 total_acc 的 x、y、z 分量，保留重力信息，单位为 g；六通道再加入 body_gyro 的三个角速度分量，单位为 rad/s。不加入分离后的 body_acc。CNN 输入为 N×3×128 或 N×6×128；统计特征从这些相同窗口计算。所有样本通过有限性检查。')
    p('CNN 的各通道均值和标准差只在训练集估计；逻辑回归的 StandardScaler 也仅拟合训练特征。随机森林直接使用未标准化的统计特征，不引入额外的学习型预处理。验证与测试始终使用已经确定的处理规则，未将验证集并入训练集重新拟合。')
    heading('2.2 环境与补充实验边界',level=2)
    p('运行环境为 Windows、Python 3.12.10、NumPy 2.5.3、scikit-learn 1.9.1 和 PyTorch 2.14.0+cu130。CNN 使用 RTX 5060；传统方法使用 CPU。随机森林在首轮 A–D 测试结果已查看后补充，搜索范围在本轮训练前固定，仅用验证集选模，再冻结权重评价测试集。因此本轮属于补充比较，而非全新未查看测试集的独立验证。',True)

    heading('3 方法设计与实现',True)
    heading('3.1 共享统计特征与逻辑回归',level=2)
    p('每通道提取均值、总体标准差、最小值、最大值、中位数、四分位距和均方根，形成 21 或 42 维特征。统计量概括幅值和波动，但不显式保留时间点的先后顺序。逻辑回归使用 L2 正则化及 lbfgs 求解，在 C∈{0.1, 1, 10} 中按验证 Macro-F1 选择，最多迭代 3000 次[4]。')
    p(f'A、B 最终 C 分别为 {runs["A",42]["best_C"]:g} 和 {runs["B",42]["best_C"]:g}。线性分类器作为基线，可以检验简单统计信息是否已经足够；在相同特征上增加随机森林，则进一步检验非线性分类能力的作用。')
    heading('3.2 一维卷积网络',level=2)
    p('CNN 沿时间轴学习局部变化模式。前两层卷积后使用最大池化压缩时间长度，末端全局平均池化汇总每个特征通道，再经 Dropout 与全连接层分类；未使用预训练、数据增强或 BatchNorm[3]。')
    table(['模块','配置','输出形状 不含批维'],[['输入','按通道标准化','(3 或 6) × 128'],['卷积块 1','64 通道 核 7 ReLU 池化 2','64 × 64'],['卷积块 2','128 通道 核 5 ReLU 池化 2','128 × 32'],['卷积块 3','128 通道 核 3 ReLU','128 × 32'],['分类头','全局平均池化 Dropout 0.3 全连接','6 类 logits']],[3,8.5,4.8],9.3)
    p(f'所有卷积步长为 1、对称填充。C、D 参数量为 {runs["C",42]["parameters"]:,} 和 {runs["D",42]["parameters"]:,}；六通道增加首层参数，传感器对比并非严格等容量。交叉熵损失、Adam、学习率 0.001、批大小 64；最多 60 轮，验证 Macro-F1 连续 10 轮不改善则停止，保存验证最佳轮。')
    heading('3.3 随机森林与统一选模',level=2)
    p('随机森林将多棵决策树的类别概率平均，利用有放回样本抽样和每次分裂的随机特征子集降低单树的相关性[5]。固定 300 棵树、Gini 分裂、max_features=sqrt，不加类别权重；深度取 12 或不限，叶节点最少样本数取 3 或 1，共四种配置。')
    rows=[]
    for e in ['E','F']:
        r=runs[e,42];pr=r['selected_params'];rows.append([e,'不限' if pr['max_depth'] is None else pr['max_depth'],pr['min_samples_leaf'],pct(r['mean_selection_val_f1'])])
    table(['组别','所选最大深度','最小叶节点样本数','验证 Macro-F1 均值 %'],rows,[1.4,4.3,4.8,5.8])
    p('每种随机森林配置运行种子 42、43、44，以三次验证 Macro-F1 均值选优，平局优先深度 12，再优先叶节点样本数 3。选中配置的三个模型直接测试，不重训合并集合。CNN 也使用这三个种子；两种方法的调参机制不同，并非等预算搜索。候选记录、原权重和新增权重分别留档。',True)

    heading('4 实验结果与总体比较',True)
    p('主指标 Macro-F1 是六类 F1 的等权平均；某类 F1=2TP/(2TP+FP+FN)，兼顾误报和漏报。准确率为正确窗口比例。下表中随机方法报告三个种子的均值与样本标准差（ddof=1），逻辑回归为一次确定性拟合；标准差描述训练随机性，不是置信区间。')
    rows=[]
    for e,a in agg.items():
        fmt=lambda m:pct(a[m+'_mean'])+(f' ± {pct(a[m+"_std"])}' if a['runs']>1 else '')
        rows.append([e,names[a['method']],a['channels'],fmt('accuracy'),fmt('macro_f1')])
    caption('表 1 六组方法在相同官方测试集上的结果 单位为 %')
    table(['组别','方法','通道数','准确率','Macro-F1'],rows,[1.1,3.8,1.4,5,5])
    picture('overall_results')
    caption('图 2 三种方法的传感器组合比较 误差线为三个种子的样本标准差')
    p(f'在当前协议下，随机森林 E、F 分别比同输入的逻辑回归低 {diff("A","E"):.2f}、{diff("B","F"):.2f} 个百分点，未缩小与 CNN 的差距。六通道 CNN 高于随机森林 {diff("D","F"):.2f} 个百分点。结果说明，把统计特征分类器换为更复杂的非线性方法，并不必然改善跨人识别。')
    caption('表 2 六通道方法的训练 验证 测试 Macro-F1 均值 单位为 %')
    table(['方法','训练','验证','测试'],[[names[agg[e]['method']],pct(gen[e]['train_macro_f1']),pct(gen[e]['val_macro_f1']),pct(gen[e]['test_macro_f1'])] for e in ['B','D','F']],[4.3,4,4,4])
    p(f'随机森林训练得分达到 {pct(gen["F"]["train_macro_f1"])}%，测试为 {pct(gen["F"]["test_macro_f1"])}%，相差 {100*(gen["F"]["train_macro_f1"]-gen["F"]["test_macro_f1"]):.2f} 个百分点。这与模型对训练受试者拟合较充分、跨人泛化不足的解释一致；但验证与测试人群不同，不能仅凭差距确定过拟合的唯一原因。训练分数是对保存模型的事后描述，未用于再次选模。',True)

    heading('5 传感器消融与错误分析',True)
    p(f'加入陀螺仪后，逻辑回归、CNN、随机森林的 Macro-F1 分别提高 {diff("B","A"):.2f}、{diff("D","C"):.2f}、{diff("F","E"):.2f} 个百分点。三种方法的增益方向一致，支持角速度在本数据上具有补充价值；但更高的相对增益不等于更高的最终得分，随机森林仍落后。')
    picture('class_results')
    caption('图 3 六组类别 F1 逻辑回归为单次结果 CNN 与随机森林为三次均值')
    p(f'六通道 CNN 的步行、上楼 F1 分别为 {pct(cls["D","步行"])}%、{pct(cls["D","上楼"])}%，随机森林为 {pct(cls["F","步行"])}%、{pct(cls["F","上楼"])}%。相反，随机森林的坐姿和站姿得分高于 CNN 均值，说明统计特征仍适合部分静态姿态。CNN 加入陀螺仪后，下楼 F1 从 {pct(cls["C","下楼"])}% 降至 {pct(cls["D","下楼"])}%，并非所有类别同步改善。')
    picture('confusion_seed42')
    caption('图 4 固定种子 42 的六通道模型 数字为窗口数 颜色按真实类别行归一化')
    cmD=runs['D',42]['confusion_matrix'];cmF=runs['F',42]['confusion_matrix']
    p(f'固定种子 42 下，随机森林把 {cmF[0][1]} 个步行窗口判为上楼，CNN 对应为 {cmD[0][1]} 个；这类动态活动混淆是两者差距的重要组成。CNN 仍把 {cmD[3][4]} 个坐姿判为站姿、{cmD[4][3]} 个站姿判为坐姿。汇总统计可能丢失节律与顺序，这是待检验的解释，尚不能据此证明 CNN 的全部增益来自时序建模。')

    heading('6 案例分析与结论',True)
    p('沿用原先按 C 与 D 预测选择的三个案例，在“由错变对”“由对变错”“两者都错”各类别中取官方测试行号最小者，再补充 F 的预测。案例不按随机森林结果重新筛选，所有图示固定种子 42，不选择测试最高的种子。')
    labels=['由错变对','由对变错','两者都错']
    table(['案例','测试行','真实','C 加速度 CNN','D 六通道 CNN','F 六通道 RF'],[[labels[i],c['test_row'],c['true'],c['C_prediction'],c['D_prediction'],c['F_prediction']] for i,c in enumerate(analysis['cases'])],[2.8,1.6,1.9,3.4,3.4,3.2],9.3)
    p(f'上述三个案例均来自受试者 {analysis["cases"][0]["subject"]}，用于解释具体决策，不代表全部人群。C→D 共纠正 {analysis["corrected_windows"]} 个窗口，同时新增 {analysis["new_errors"]} 个错误。完整十四份逐窗口预测、类别指标和逐种子表随代码提供，可按 test_row 回查。')
    heading('6.1 局限与进一步验证',level=2)
    p('本研究只有一个受试者划分、单一设备和固定腰部佩戴位置；相邻窗口重叠，窗口数不能视为独立人数。三个种子的标准差没有覆盖受试者重新划分的变化，也未测量设备功耗或部署延迟。随机森林只搜索四种配置，结果不代表该算法的能力上限。')
    p('随机森林是在首轮测试已查看后补充，虽然新增参数只通过验证集选择，仍存在研究方案受已有结果影响的可能。更强的后续证据应来自训练受试者交叉验证及新增独立人群，不能继续围绕同一测试集反复改模型。可进一步比较频域统计，并做时序打乱对照，检验时间顺序是否解释 CNN 的优势。')
    heading('6.2 结论',level=2)
    p(f'本实验中，{names[agg[best]["method"]]}的{agg[best]["channels"]}通道组合取得最高平均测试 Macro-F1，为 {pct(agg[best]["macro_f1_mean"])}%。随机森林没有超越线性基线，说明复杂度与泛化表现不能简单对应；三种方法均从陀螺仪获得总体收益，但各类别收益不同。新增与原有模型重载复核通过 {audit["n_checks"]} 项检查，原实验校验值保持不变。')
    heading('参考文献与来源',level=2)
    refs=[
        '[1] Reyes-Ortiz J, Anguita D, et al. Human Activity Recognition Using Smartphones [Dataset]. UCI, 2013. https://doi.org/10.24432/C54S4K.',
        '[2] Anguita D, Ghio A, Oneto L, et al. A Public Domain Dataset for Human Activity Recognition Using Smartphones. ESANN, 2013: 437–442. https://www.esann.org/sites/default/files/proceedings/legacy/es2013-84.pdf.',
        '[3] PyTorch. Conv1d documentation. https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html.',
        '[4] scikit-learn. LogisticRegression documentation. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html.',
        '[5] scikit-learn. RandomForestClassifier documentation. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html. 新增资料核对于 2026-11-02。'
    ]
    for ref in refs:p(ref,True)
    p('工具说明：使用 Codex AI 辅助编写代码、执行实验、制图与整理报告；数值来自本地真实运行，未将现成方法冒充原创算法。数据、开源工具及原论文来源如上，完整复现材料随项目提供。',True)

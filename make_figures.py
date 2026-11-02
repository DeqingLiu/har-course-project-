"""Generate publication figures and deterministic case analysis from saved results."""
import csv
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from har import ROOT, OUT, CN, CLASSES, EXPERIMENTS, read_json, write_json, load_split, csv_write

FIG = OUT / "figures"
available = {f.name for f in font_manager.fontManager.ttflist}
required_fonts={"Times New Roman", "SimSun", "SimHei"}
if not required_fonts <= available:
    raise RuntimeError(f"Report typography requires installed fonts: {sorted(required_fonts-available)}")
plt.rcParams.update({"font.family": ["Times New Roman", "SimSun"], "axes.unicode_minus": False, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 220})
COLORS = ["#285F8F", "#E08B36", "#418A73", "#9568A6", "#A36D4F", "#667F94"]
NAMES = {"logistic":"统计特征＋逻辑回归", "cnn":"一维 CNN", "random_forest":"统计特征＋随机森林"}


def save(fig, name):
    for ax in fig.axes:
        ax.title.set_fontfamily(["Times New Roman", "SimHei"])
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    global OUT, FIG
    parser=argparse.ArgumentParser()
    parser.add_argument("--results-dir",type=Path,default=OUT)
    args=parser.parse_args()
    OUT=args.results_dir.resolve()
    FIG=OUT/"figures"
    FIG.mkdir(parents=True,exist_ok=True)
    results = read_json(OUT / "results.json")
    runs = results["runs"]
    experiments=[r["experiment"] for r in results["aggregate"]]
    by_key={(r["method"],r["channels"]):r for r in results["aggregate"]}
    methods=[m for m in NAMES if any(r["method"]==m for r in results["aggregate"])]
    fixed = {r["experiment"]: r for r in runs if r["seed"]==42}
    train, test = load_split("train"), load_split("test")
    manifest = read_json(ROOT / "data/split_manifest.json")
    # Overview includes no test-derived input example.
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 2.7), gridspec_kw={"width_ratios": [1,1.5]})
    for i,(key, label) in enumerate([("train","训练"),("val","验证"),("test","测试")]):
        axes[0].bar(np.arange(6)+(i-1)*.24, manifest["splits"][key]["class_counts"], .24, label=label, color=COLORS[i])
    axes[0].set_xticks(range(6),CN,rotation=25)
    axes[0].set_ylabel("窗口数")
    axes[0].legend(fontsize=8)
    idx = int(np.flatnonzero(train["y"]==0)[0])
    for axis in range(3):
        axes[1].plot(np.arange(128)/50,train["X"][idx,axis],label="XYZ"[axis],lw=1.3)
    axes[1].set(xlabel="时间 / s",ylabel="总加速度 / g",title="训练集步行窗口示例")
    axes[1].legend(fontsize=8,ncol=3)
    fig.tight_layout()
    save(fig,"data_overview")

    fig, ax = plt.subplots(figsize=(9.3, 2.8))
    for ci,(channels,label,color) in enumerate([(3,"仅加速度",COLORS[0]),(6,"加速度＋陀螺仪",COLORS[1])]):
        for i,method in enumerate(methods):
            r=by_key[method,channels];x=i+(ci-.5)*.32
            ax.bar(x,100*r["macro_f1_mean"],yerr=100*r["macro_f1_std"] if r["runs"]>1 else None,color=color,capsize=4,width=.30,label=label if i==0 else None)
            ax.text(x,100*r["macro_f1_mean"]+100*r["macro_f1_std"]+1,f"{100*r['macro_f1_mean']:.2f}",ha="center",fontsize=9)
    ax.set_xticks(range(len(methods)),[NAMES[m] for m in methods])
    ax.legend(loc="lower right",fontsize=8)
    ax.set(ylim=(0,106),ylabel="测试集 Macro-F1 / %")
    ax.grid(axis="y",alpha=.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig,"overall_results")

    fig, axes = plt.subplots(1,2,figsize=(9.3,3.45))
    displayed=["D","F"] if "F" in fixed else ["C","D"]
    for ax,e in zip(axes,displayed):
        cm=np.array(fixed[e]["confusion_matrix"])
        ax.imshow(cm/cm.sum(1,keepdims=True),cmap="Blues",vmin=0,vmax=1)
        for i in range(6):
            for j in range(6):
                ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=8,color="white" if cm[i,j]/cm[i].sum()>.55 else "black")
        ax.set_xticks(range(6),CN,rotation=25)
        ax.set_yticks(range(6),CN)
        ax.set(title=f"{e}  {NAMES[fixed[e]['method']]}  {fixed[e]['channels']}通道",xlabel="预测类别",ylabel="真实类别")
    fig.tight_layout()
    save(fig,"confusion_seed42")

    fig, axes = plt.subplots(1,2,figsize=(9.3,2.8))
    for e,color in zip("CD",COLORS[2:4]):
        history=read_csv((ROOT/fixed[e]["model_path"]).parent/"history.csv")
        epochs=[int(x["epoch"]) for x in history]
        axes[0].plot(epochs,[float(x["train_loss"]) for x in history],label=f"{e} 训练",color=color)
        axes[0].plot(epochs,[float(x["val_loss"]) for x in history],label=f"{e} 验证",color=color,ls="--")
        axes[1].plot(epochs,[100*float(x["val_macro_f1"]) for x in history],label=e,color=color)
        axes[1].axvline(fixed[e]["best_epoch"],color=color,lw=.8,ls=":")
    axes[0].set(xlabel="训练轮数",ylabel="交叉熵损失")
    axes[1].set(xlabel="训练轮数",ylabel="验证集 Macro-F1 / %")
    for ax in axes:
        ax.legend(fontsize=8,ncol=2)
        ax.grid(alpha=.2)
    fig.tight_layout()
    save(fig,"training_seed42")

    fig, ax = plt.subplots(figsize=(9.3,3.0))
    per_class=[]
    matrix=[]
    for i,e in enumerate(experiments):
        group=[r for r in runs if r["experiment"]==e]
        values=np.array([[c["f1"] for c in r["per_class"]] for r in group])*100
        matrix.append(values.mean(0))
        for j in range(6):
            per_class.append({"experiment":e,"activity":CN[j],"f1_mean":values[:,j].mean()/100,"f1_std":values[:,j].std(ddof=1)/100 if len(group)>1 else 0})
    matrix=np.array(matrix)
    im=ax.imshow(matrix,cmap="Blues",vmin=50,vmax=100,aspect="auto")
    for i in range(len(experiments)):
        for j in range(6):
            ax.text(j,i,f"{matrix[i,j]:.2f}",ha="center",va="center",fontsize=9,color="white" if matrix[i,j]>84 else "black")
    ax.set_xticks(range(6),CN)
    ax.set_yticks(range(len(experiments)),[f"{e}  {NAMES[fixed[e]['method']]}  {fixed[e]['channels']}通道" for e in experiments],fontsize=9)
    fig.colorbar(im,ax=ax,label="类别 F1 / %",fraction=.025,pad=.03)
    fig.tight_layout()
    save(fig,"class_results")
    csv_write(OUT/"class_summary.csv",per_class)

    pred={e:read_csv(OUT/"predictions"/f"{e}_seed42.csv") for e in experiments}
    c=np.array([int(x["pred_label"])-1 for x in pred["C"]])
    d=np.array([int(x["pred_label"])-1 for x in pred["D"]])
    y=test["y"]
    categories={"加入陀螺仪后纠正":(c!=y)&(d==y),"加入陀螺仪后出错":(c==y)&(d!=y),"两种输入都预测错误":(c!=y)&(d!=y)}
    cases=[]
    for category,mask in categories.items():
        indices=np.flatnonzero(mask)
        if len(indices):
            i=int(indices[0])
            cases.append({"category":category,"test_row":int(test["source_rows"][i]),"subject":int(test["subjects"][i]),
                "true":CN[y[i]],"C_prediction":CN[c[i]],"D_prediction":CN[d[i]],"F_prediction":CN[int(pred['F'][i]['pred_label'])-1] if 'F' in pred else '',"category_count":int(mask.sum())})
    csv_write(OUT/"case_studies.csv",cases)
    gains=[{"method":m,"macro_f1_delta_pp":100*(by_key[m,6]["macro_f1_mean"]-by_key[m,3]["macro_f1_mean"])} for m in methods]
    csv_write(OUT/"sensor_gains.csv",gains)
    write_json(OUT/"analysis.json",{"case_selection":"First official test row per category selected from C/D; fixed seed 42, not best seed", "sensor_gains":gains,"confusion_experiments":displayed,
        "cases":cases,"corrected_windows":int(((c!=y)&(d==y)).sum()),"new_errors":int(((c==y)&(d!=y)).sum()),
        "class_summary":per_class,
        "cnn_paired_deltas_pp":[100*(next(r["macro_f1"] for r in runs if r["experiment"]=="D" and r["seed"]==s)-next(r["macro_f1"] for r in runs if r["experiment"]=="C" and r["seed"]==s)) for s in [42,43,44]]})
    print("Figures and case analysis generated:",FIG)


if __name__=="__main__":
    main()

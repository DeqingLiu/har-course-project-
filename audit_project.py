"""Independent delivery audit. Uses csv/standard arithmetic, not sklearn metrics."""
import ast
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from pypdf import PdfReader
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"outputs"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument("--results-dir",type=Path,default=ROOT/"outputs/expanded")
    OUT=parser.parse_args().results_dir.resolve()
    checks=[]
    def check(name,condition):
        assert condition,name
        checks.append(name)
    result=read(OUT/"results.json")
    check("14 saved runs",len(result["runs"])==14 and len({r["run_id"] for r in result["runs"]})==14)
    check("six experiments",{r["experiment"] for r in result["aggregate"]}==set("ABCDEF"))
    split=read(ROOT/"data/split_manifest.json")
    seal=read(OUT/"frozen_manifest.json")
    for path,digest in seal["files"].items():
        check("hash "+path,hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest)
    original=read(ROOT/"outputs/results.json")
    check("A-D original result records",result["runs"][:8]==original["runs"])
    with np.load(ROOT/"data/processed/test.npz") as z:
        true_y=z["y"].copy();true_subjects=z["subjects"].copy()
    for script in ROOT.glob("*.py"):
        ast.parse(script.read_text(encoding="utf-8"))
    check("Python entrypoints parse",True)
    for r in result["runs"]:
        with (OUT/"predictions"/(r["run_id"]+".csv")).open(encoding="utf-8-sig") as f:
            rows=list(csv.DictReader(f))
        check(r["run_id"]+" unique rows",[int(x["test_row"]) for x in rows]==list(range(1,2948)))
        check(r["run_id"]+" test subjects",sorted({int(x["subject"]) for x in rows})==split["splits"]["test"]["subjects"])
        check(r["run_id"]+" labels match official test",np.array_equal([int(x["true_label"])-1 for x in rows],true_y))
        check(r["run_id"]+" subject rows match official test",np.array_equal([int(x["subject"]) for x in rows],true_subjects))
        cm=[[0]*6 for _ in range(6)]
        valid_probs=True
        for x in rows:
            y,p=int(x["true_label"])-1,int(x["pred_label"])-1
            cm[y][p]+=1
            prob=[float(v) for k,v in x.items() if k.startswith("p_")]
            valid_probs &= len(prob)==6 and all(math.isfinite(v) and 0<=v<=1 for v in prob) and abs(sum(prob)-1)<1e-5
            valid_probs &= max(range(6),key=lambda i:prob[i])==p and abs(float(x["confidence"])-max(prob))<1e-6
        check(r["run_id"]+" probability distributions",valid_probs)
        check(r["run_id"]+" confusion counts",cm==r["confusion_matrix"])
        scores=[]
        for k in range(6):
            tp=cm[k][k]; actual=sum(cm[k]); predicted=sum(row[k] for row in cm)
            score=2*tp/(actual+predicted) if actual+predicted else 0
            scores.append(score)
            check(r["run_id"]+f" class {k} metric",abs(score-r["per_class"][k]["f1"])<1e-12 and actual==r["per_class"][k]["support"])
        check(r["run_id"]+" independent macro F1",abs(statistics.mean(scores)-r["macro_f1"])<1e-12)
        check(r["run_id"]+" independent accuracy",abs(sum(cm[i][i] for i in range(6))/len(rows)-r["accuracy"])<1e-12)
    for a in result["aggregate"]:
        group=[r for r in result["runs"] if r["experiment"]==a["experiment"]]
        for key in ["accuracy","macro_f1"]:
            vals=[r[key] for r in group]
            check(a["experiment"]+" "+key+" aggregate",abs(statistics.mean(vals)-a[key+"_mean"])<1e-12 and abs((statistics.stdev(vals) if len(vals)>1 else 0)-a[key+"_std"])<1e-12)
    pdf=PdfReader(ROOT/"submission/课程设计报告.pdf")
    check("exactly six report pages",len(pdf.pages)==6)
    expected=["研究问题与实验目标","数据来源与实验环境","方法设计与实现","实验结果与总体比较","传感器消融与错误分析","案例分析与结论"]
    texts=["".join(p.extract_text().split()) for p in pdf.pages]
    for i,title in enumerate(expected):
        check(f"page {i+1} section",title in texts[i])
    report="".join(texts)
    for a in result["aggregate"]:
        for key in ["accuracy_mean","macro_f1_mean"]:
            check("report number "+a["experiment"]+" "+key,f"{100*a[key]:.2f}" in report)
    check("AI assistance disclosed","CodexAI辅助" in report)
    check("supplementary experiment disclosed","首轮" in report and "测试结果已查看后补充" in report)
    check("report contains no Word deliverable",not list((ROOT/"submission").glob("*.docx")))
    check("case table matches recorded cases",all(str(c["test_row"]) in texts[5] for c in read(OUT/"analysis.json")["cases"]))
    check("model restoration audit passed",read(OUT/"validation.json")["passed"])
    payload={"passed":True,"n_checks":len(checks),"checks":checks,"pdf_pages":len(pdf.pages),
        "note":"Independent metric/structural audit. Visual page inspection is additionally recorded in 验收记录.md."}
    (OUT/"delivery_audit.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"PASS: {len(checks)} independent checks; report has {len(pdf.pages)} pages.")


if __name__=="__main__":
    main()

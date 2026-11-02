"""Post-hoc descriptive analysis only; no model fitting or selection."""
import argparse
from pathlib import Path
import statistics
import torch
from har import ROOT, read_json, write_json, csv_write, load_split, inference, metrics, set_seed
from har_rf import rf_predict, verify_seal


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--results-dir",type=Path,default=ROOT/"outputs/expanded")
    out=parser.parse_args().results_dir.resolve()
    verify_seal()
    res=read_json(out/"results.json")
    train=load_split("train")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    records=[]
    for r in res["runs"]:
        pred,_=rf_predict(r,train) if r["method"]=="random_forest" else inference(r,train,device)
        record={"run_id":r["run_id"],"experiment":r["experiment"],"train_macro_f1":metrics(train["y"],pred)["macro_f1"],
            "val_macro_f1":r["val_macro_f1"],"test_macro_f1":r["macro_f1"]}
        records.append(record)
    csv_write(out/"generalization_by_seed.csv",records)
    summary=[]
    for exp in sorted({r["experiment"] for r in records}):
        group=[r for r in records if r["experiment"]==exp]
        summary.append({"experiment":exp,**{k:statistics.mean(r[k] for r in group) for k in ["train_macro_f1","val_macro_f1","test_macro_f1"]}})
    csv_write(out/"generalization_summary.csv",summary)
    write_json(out/"generalization.json",{"note":"Training scores evaluated at saved checkpoints after test evaluation; descriptive only, no refitting or tuning", "by_seed":records,"aggregate":summary})
    print(summary)


if __name__=="__main__":main()

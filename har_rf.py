"""Supplementary random forest protocol; existing har.py and A-D evidence are immutable.

Commands: train -> evaluate -> validate. Candidate selection uses validation only.
"""
import argparse
import csv
import json
import platform
import shutil
import statistics
import time
from pathlib import Path
import joblib
import numpy as np
import sklearn
import torch
from sklearn.ensemble import RandomForestClassifier
from har import (ROOT, OUT as BASE, CLASSES, read_json, write_json, csv_write,
                 sha256, utc_now, verify_seal as verify_original, load_split,
                 features, metrics, inference, set_seed)

OUT = BASE / "expanded"
CFG = ROOT / "config_rf.json"


def verify_snapshot(snapshot):
    for name, digest in snapshot.items():
        assert sha256(ROOT/name) == digest, f"Original evidence changed: {name}"


def verify_seal():
    verify_original()
    seal = read_json(OUT/"frozen_manifest.json")
    verify_snapshot(seal["files"])
    return seal


def train():
    original = verify_original()
    cfg = read_json(CFG)
    if (OUT/"protocol.json").exists():
        raise RuntimeError("An extension protocol already exists. Preserve it and use a new copy for another run.")
    OUT.mkdir(parents=True, exist_ok=True)
    snapshot = dict(original["files"])
    evidence = [BASE/"frozen_manifest.json", BASE/"results.json", BASE/"summary.csv", BASE/"validation.json"]
    evidence += list((BASE/"predictions").glob("*.csv"))
    for p in evidence:
        snapshot[p.relative_to(ROOT).as_posix()] = sha256(p)
    write_json(OUT/"original_snapshot.json", snapshot)
    write_json(OUT/"protocol.json", {"registered_utc": utc_now(), "config": cfg,
        "config_sha256": sha256(CFG), "training_code_sha256": sha256(Path(__file__)),
        "test_history": "A-D test results already inspected before this supplementary comparison",
        "rule": "Fit and select on original train/val only; freeze six RF models before supplementary test evaluation"})
    write_json(OUT/"environment.json", {"python": platform.python_version(), "numpy": np.__version__,
        "sklearn": sklearn.__version__, "platform": platform.platform(), "device": "CPU", "n_jobs": cfg["fixed"]["n_jobs"]})
    # No test split is opened here. The original split checksum is verified only.
    tr, va = load_split("train"), load_split("val")
    selected = []
    for exp, channels in cfg["experiments"].items():
        ft, fv = features(tr["X"][:, :channels]), features(va["X"][:, :channels])
        candidates = []
        for ci, candidate in enumerate(cfg["candidate_order"]):
            seed_records = []
            for seed in cfg["seeds"]:
                t0 = time.perf_counter()
                model = RandomForestClassifier(**cfg["fixed"], **candidate, random_state=seed)
                model.fit(ft, tr["y"])
                pred = model.predict(fv)
                score = metrics(va["y"], pred)["macro_f1"]
                path = ROOT/"tmp/rf_candidates"/f"{exp}_candidate{ci}_seed{seed}.joblib"
                path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, path, compress=3)
                record = {"seed": seed, "val_macro_f1": score, "training_seconds": time.perf_counter()-t0,
                    "candidate_model": path.relative_to(ROOT).as_posix()}
                seed_records.append(record)
                print(f"{exp} candidate={ci} depth={candidate['max_depth']} leaf={candidate['min_samples_leaf']} seed={seed} val_f1={score:.6f}", flush=True)
            candidates.append({"candidate_index": ci, "params": candidate, "seeds": seed_records,
                "mean_val_macro_f1": statistics.mean(r["val_macro_f1"] for r in seed_records)})
        # Python max returns the first item in an exact tie: shallow then larger leaf.
        winner = max(candidates, key=lambda x: x["mean_val_macro_f1"])
        write_json(OUT/"search"/f"{exp}.json", {"experiment": exp, "channels": channels,
            "candidates": candidates, "selected_candidate_index": winner["candidate_index"],
            "selected_mean_val_macro_f1": winner["mean_val_macro_f1"]})
        for seed_run in winner["seeds"]:
            seed = seed_run["seed"]
            run_id = f"{exp}_seed{seed}"
            folder = OUT/"runs"/run_id
            folder.mkdir(parents=True, exist_ok=True)
            path = folder/"model.joblib"
            shutil.copy2(ROOT/seed_run["candidate_model"], path)
            record = {"run_id": run_id, "experiment": exp, "method": "random_forest", "channels": channels,
                "seed": seed, "val_macro_f1": seed_run["val_macro_f1"], "selected_params": winner["params"],
                "selected_candidate_index": winner["candidate_index"], "mean_selection_val_f1": winner["mean_val_macro_f1"],
                "training_seconds": seed_run["training_seconds"], "n_estimators": cfg["fixed"]["n_estimators"],
                "model_path": path.relative_to(ROOT).as_posix(), "model_sha256": sha256(path)}
            write_json(folder/"selection.json", record)
            selected.append(record)
        print(f"SELECTED {exp}: {winner['params']} mean_val_f1={winner['mean_val_macro_f1']:.6f}", flush=True)
    verify_snapshot(snapshot)
    files = [CFG, Path(__file__), OUT/"protocol.json", OUT/"original_snapshot.json", OUT/"environment.json"]
    files += list((OUT/"search").glob("*.json")) + list((OUT/"runs").glob("*/*"))
    frozen = {**snapshot, **{p.relative_to(ROOT).as_posix(): sha256(p) for p in files}}
    write_json(OUT/"frozen_manifest.json", {"sealed_utc": utc_now(), "files": frozen, "runs": selected,
        "rule": cfg["selection"], "protocol": cfg["protocol"]})
    print("SEALED six final RF models before new test evaluation", flush=True)


def rf_predict(run, dataset):
    model = joblib.load(ROOT/run["model_path"])
    x = features(dataset["X"][:, :run["channels"]])
    prob = model.predict_proba(x)
    assert np.array_equal(model.classes_, np.arange(6))
    return prob.argmax(1), prob


def evaluate():
    seal = verify_seal()
    test = load_split("test")
    original = read_json(BASE/"results.json")
    results = list(original["runs"])
    (OUT/"predictions").mkdir(exist_ok=True)
    for run in original["runs"]:
        shutil.copy2(BASE/"predictions"/f"{run['run_id']}.csv", OUT/"predictions"/f"{run['run_id']}.csv")
    for run in seal["runs"]:
        pred, prob = rf_predict(run, test)
        result = {**run, **metrics(test["y"], pred)}
        results.append(result)
        rows = [{"test_row": int(test["source_rows"][i]), "subject": int(test["subjects"][i]),
            "true_label": int(test["y"][i])+1, "true_activity": CLASSES[test["y"][i]],
            "pred_label": int(pred[i])+1, "pred_activity": CLASSES[pred[i]], "confidence": float(prob[i].max()),
            **{f"p_{c}": float(prob[i,j]) for j,c in enumerate(CLASSES)}} for i in range(len(pred))]
        csv_write(OUT/"predictions"/f"{run['run_id']}.csv", rows)
        print(f"{run['run_id']} accuracy={result['accuracy']:.6f} macro_f1={result['macro_f1']:.6f}", flush=True)
    aggregate = []
    for exp in sorted({r["experiment"] for r in results}):
        group = [r for r in results if r["experiment"] == exp]
        aggregate.append({"experiment": exp, "method": group[0]["method"], "channels": group[0]["channels"], "runs": len(group),
            **{f"{m}_mean": statistics.mean(r[m] for r in group) for m in ["accuracy", "macro_f1"]},
            **{f"{m}_std": statistics.stdev(r[m] for r in group) if len(group)>1 else 0.0 for m in ["accuracy", "macro_f1"]}})
    write_json(OUT/"results.json", {"evaluated_utc": utc_now(), "frozen_manifest_sha256": sha256(OUT/"frozen_manifest.json"),
        "original_results_sha256": sha256(BASE/"results.json"), "protocol": seal["protocol"], "runs": results, "aggregate": aggregate})
    csv_write(OUT/"summary.csv", aggregate)
    csv_write(OUT/"per_class.csv", [{"run_id":r["run_id"], **c} for r in results for c in r["per_class"]])
    csv_write(OUT/"seed_results.csv", [{"run_id":r["run_id"],"experiment":r["experiment"],"seed":r["seed"],
        "val_macro_f1":r["val_macro_f1"],"test_accuracy":r["accuracy"],"test_macro_f1":r["macro_f1"],
        "best_epoch":r.get("best_epoch",""),"selected_params":json.dumps(r.get("selected_params",{}))} for r in results])
    per_subject=[]
    for run in results:
        with (OUT/"predictions"/f"{run['run_id']}.csv").open(encoding="utf-8-sig") as f:
            rows=list(csv.DictReader(f))
        pred=np.array([int(x["pred_label"])-1 for x in rows])
        for person in np.unique(test["subjects"]):
            mask=test["subjects"]==person
            m=metrics(test["y"][mask],pred[mask])
            per_subject.append({"run_id":run["run_id"],"subject":int(person),"n":int(mask.sum()),"accuracy":m["accuracy"],"macro_f1":m["macro_f1"]})
    csv_write(OUT/"per_subject.csv",per_subject)


def validate():
    seal=verify_seal()
    cfg=read_json(CFG)
    res=read_json(OUT/"results.json")
    original=read_json(BASE/"results.json")
    checks=[]
    def check(name,value):
        assert value,name
        checks.append(name)
    check("original results unchanged",res["runs"][:8]==original["runs"])
    expected={r["run_id"] for r in original["runs"]}|{f"{e}_seed{s}" for e in "EF" for s in cfg["seeds"]}
    check("14 final runs and 6 experiments",{r["run_id"] for r in res["runs"]}==expected and len(res["aggregate"])==6)
    protocol=read_json(OUT/"protocol.json")
    check("protocol before freeze before evaluation",protocol["registered_utc"]<=seal["sealed_utc"]<=res["evaluated_utc"])
    parts={s:load_split(s) for s in ["train","val","test"]}
    sets=[set(x["subjects"].tolist()) for x in parts.values()]
    check("subject isolation",all(not sets[i]&sets[j] for i in range(3) for j in range(i)))
    for exp in "EF":
        search=read_json(OUT/"search"/f"{exp}.json")
        for c in search["candidates"]:
            check(f"{exp} candidate {c['candidate_index']} mean",abs(statistics.mean(s["val_macro_f1"] for s in c["seeds"])-c["mean_val_macro_f1"])<1e-12)
        winner=max(search["candidates"],key=lambda c:c["mean_val_macro_f1"])
        check(exp+" validation-only winning candidate",winner["candidate_index"]==search["selected_candidate_index"])
    set_seed(42)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for run in res["runs"]:
        rid=run["run_id"]
        if run["method"]=="random_forest":
            model=joblib.load(ROOT/run["model_path"])
            params=model.get_params()
            check(rid+" feature width",model.n_features_in_==run["channels"]*7)
            check(rid+" selected settings",all(params[k]==v for k,v in {**cfg["fixed"],**run["selected_params"],"random_state":run["seed"]}.items()))
            vpred,_=rf_predict(run,parts["val"])
            pred,_=rf_predict(run,parts["test"])
        else:
            vpred,_=inference(run,parts["val"],device)
            pred,_=inference(run,parts["test"],device)
            check(rid+" original prediction bytes",sha256(OUT/"predictions"/f"{rid}.csv")==sha256(BASE/"predictions"/f"{rid}.csv"))
        check(rid+" restored validation",abs(metrics(parts["val"]["y"],vpred)["macro_f1"]-run["val_macro_f1"])<1e-12)
        with (OUT/"predictions"/f"{rid}.csv").open(encoding="utf-8-sig") as f:
            saved=list(csv.DictReader(f))
        check(rid+" restored test predictions",np.array_equal(pred,np.array([int(r["pred_label"])-1 for r in saved])))
        m=metrics(parts["test"]["y"],pred)
        check(rid+" restored metrics",abs(m["macro_f1"]-run["macro_f1"])<1e-12 and abs(m["accuracy"]-run["accuracy"])<1e-12)
    write_json(OUT/"validation.json",{"passed":True,"n_checks":len(checks),"checked_utc":utc_now(),"checks":checks})
    print(f"PASS {len(checks)} expanded model checks; original evidence intact",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["train","evaluate","validate"])
    globals()[parser.parse_args().command]()

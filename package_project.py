"""Package complete offline evidence and a separate clean source distribution."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parent
DIST=ROOT/"dist"


def main():
    DIST.mkdir(exist_ok=True)
    root_files={p for p in ROOT.iterdir() if p.is_file()}
    sources=sorted(root_files | set((ROOT/"data/raw").glob("*.txt")) | {ROOT/"data/raw/source.json"})
    complete=sorted(p for p in ROOT.rglob("*") if p.is_file() and not any(part in {"tmp","dist","__pycache__",".venv"} for part in p.relative_to(ROOT).parts))
    manifests={}
    for label,files in [("source",sources),("complete",complete)]:
        output=DIST/f"har-course-project-{label}.zip"
        with zipfile.ZipFile(output,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for path in files:
                relative=path.relative_to(ROOT).as_posix()
                if path.suffix.lower() in {".docx", ".doc", ".docm"}:raise AssertionError("Word files must not be packaged")
                z.write(path,"har-course-project/"+relative)
        with zipfile.ZipFile(output) as z:
            assert z.testzip() is None
            names=z.namelist()
            assert not any(Path(n).suffix.lower() in {".docx", ".doc", ".docm"} for n in names)
            for name in ["har.py","config.json","har_rf.py","config_rf.json","run_expanded.ps1","report_content.py","README.md","中文讲解与复现.md"]:
                assert "har-course-project/"+name in names
            if label=="complete":
                result=json.loads(z.read("har-course-project/outputs/expanded/results.json"))
                assert len(result["runs"])==14
                for run in result["runs"]:
                    relative=run["model_path"]
                    assert z.read("har-course-project/"+relative)==(ROOT/relative).read_bytes()
                for relative in ["submission/课程设计报告.pdf","submission/archive/课程设计报告_四组实验原版.pdf"]:
                    assert z.read("har-course-project/"+relative)==(ROOT/relative).read_bytes()
        manifests[label]={"file":output.name,"bytes":output.stat().st_size,"n_files":len(files),"sha256":hashlib.sha256(output.read_bytes()).hexdigest()}
    (DIST/"package_manifest.json").write_text(json.dumps(manifests,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(manifests,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()

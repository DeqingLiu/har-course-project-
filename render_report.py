"""Windows renderer: Word PDF conversion + packaged render_docx.py PNG stage.

The Codex runtime on this host does not bundle LibreOffice. Microsoft Word is
installed, so use its own PDF exporter. No external document service is used.
"""
import argparse
import json
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, default=Path.home()/".cache/codex-runtimes/codex-primary-runtime")
    parser.add_argument("--renderer", type=Path, default=None)
    parser.add_argument("--poppler", type=Path, default=None, help="Directory containing pdftoppm and pdfinfo")
    args = parser.parse_args()
    docx = ROOT/"tmp/report_source/课程设计报告.docx"
    out = ROOT/"tmp/report_render"
    poppler = args.poppler or args.runtime/"dependencies/native/poppler/Library/bin"
    if poppler.exists():
        os.environ["PATH"] = str(poppler)+os.pathsep+os.environ.get("PATH", "")
    renderer = args.renderer or args.runtime/"plugins/openai-primary-runtime/plugins/documents/skills/documents/render_docx.py"
    def word_convert(input_path, user_profile, output_dir, stem, verbose=False):
        pdf=Path(output_dir)/(stem+".pdf")
        result=subprocess.run(["powershell.exe","-NoProfile","-ExecutionPolicy","Bypass","-File",str(ROOT/"export_word_pdf.ps1"),
            "-InputDocx",str(input_path),"-OutputPdf",str(pdf)], capture_output=True, text=True, check=True)
        return str(pdf), result.stdout+result.stderr
    if renderer.exists():
        spec=importlib.util.spec_from_file_location("codex_docx_renderer", renderer)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.convert_to_pdf=word_convert
        pages=module.rasterize(str(docx),str(out),150,False,True)
        backend="Word + render_docx.py + Poppler"
    else:
        from pdf2image import convert_from_path
        out.mkdir(parents=True,exist_ok=True)
        pdf,_=word_convert(str(docx),None,str(out),docx.stem)
        pages=[]
        for i,img in enumerate(convert_from_path(pdf,dpi=150)):
            page=out/f"page-{i+1}.png"
            img.save(page)
            pages.append(str(page))
        backend="Word + pdf2image + Poppler"
    pdf=out/(docx.stem+".pdf")
    destination=ROOT/"submission/课程设计报告.pdf"
    shutil.copy2(pdf,destination)
    n=len(PdfReader(destination).pages)
    print(f"Rendered {n} pages using {backend}")
    assert n==6, f"Expected 6 pages, got {n}; revise report layout before delivery"
    identity=json.loads((ROOT/"submission/report_claims.json").read_text(encoding="utf-8"))["report_identity"]
    if all(identity[k] and identity[k]!="待填写" for k in ["name","student_id"]):
        filename=f"{identity['student_id']}-{identity['name']}.pdf"
        if any(c in filename for c in '<>:"/\\|?*'):
            raise ValueError("Identity contains invalid filename characters")
        shutil.copy2(destination, destination.parent/filename)
    for p in pages:
        print(p)


if __name__=="__main__":
    main()

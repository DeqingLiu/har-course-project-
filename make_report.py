"""Build the Chinese DOCX report from verified experimental JSON, using python-docx.

Run with the Codex bundled document Python. Does not import training packages.
"""
import argparse
import json
from pathlib import Path
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent
OUT = ROOT/"outputs"
SUB = ROOT/"submission"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def set_fonts(target, chinese="SimSun"):
    target.font.name="Times New Roman"
    fonts=target._element.get_or_add_rPr().rFonts
    # Theme mappings take precedence over explicit fonts in some Word styles.
    for attr in list(fonts.attrib):
        if attr.endswith("Theme"):
            del fonts.attrib[attr]
    for key in ["ascii","hAnsi","cs"]:
        fonts.set(qn("w:"+key),"Times New Roman")
    fonts.set(qn("w:eastAsia"),chinese)


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument("--results-dir",type=Path,default=ROOT/"outputs/expanded")
    identity=load(ROOT/"report_identity.json") if (ROOT/"report_identity.json").exists() else {}
    parser.add_argument("--name",default=identity.get("name","待填写"))
    parser.add_argument("--student-id",default=identity.get("student_id","待填写"))
    parser.add_argument("--github",default=identity.get("github","未公开发布，代码随本地项目提供"))
    args=parser.parse_args()
    OUT=args.results_dir.resolve()
    results=load(OUT/"results.json")
    analysis=load(OUT/"analysis.json")
    split=load(ROOT/"data/split_manifest.json")
    audit=load(OUT/"validation.json")
    assert audit["passed"]
    aggregate={r["experiment"]:r for r in results["aggregate"]}
    runs={(r["experiment"],r["seed"]):r for r in results["runs"]}
    class_avg={(r["experiment"],r["activity"]):r["f1_mean"] for r in analysis["class_summary"]}
    pct=lambda x:f"{100*x:.2f}"
    delta=lambda e1,e2:100*(aggregate[e1]["macro_f1_mean"]-aggregate[e2]["macro_f1_mean"])
    doc=Document()
    section=doc.sections[0]
    section.page_width=Cm(21)
    section.page_height=Cm(29.7)
    section.top_margin=Cm(1.9)
    section.bottom_margin=Cm(1.8)
    section.left_margin=Cm(2.1)
    section.right_margin=Cm(2.1)
    section.footer_distance=Cm(.8)
    for sty in ["Normal","Title","Subtitle","Heading 1","Heading 2","Caption","Footer"]:
        s=doc.styles[sty]
        set_fonts(s,"SimHei" if sty in ["Title","Subtitle","Heading 1","Heading 2"] else "SimSun")
        s.font.color.rgb=RGBColor(0,0,0)
    # Remove decorative borders inherited from the bundled default template.
    for border in list(doc.styles.element.iter(qn("w:pBdr"))):
        border.getparent().remove(border)
    normal=doc.styles["Normal"]
    normal.font.size=Pt(10.5)
    normal.paragraph_format.line_spacing=1.23
    normal.paragraph_format.space_after=Pt(5)
    doc.styles["Title"].font.size=Pt(17)
    doc.styles["Subtitle"].font.italic=False
    doc.styles["Title"].paragraph_format.space_after=Pt(7)
    doc.styles["Heading 1"].font.size=Pt(14)
    doc.styles["Heading 1"].paragraph_format.space_before=Pt(2)
    doc.styles["Heading 1"].paragraph_format.space_after=Pt(7)
    doc.styles["Heading 2"].font.size=Pt(11.5)
    doc.styles["Heading 2"].paragraph_format.space_before=Pt(7)
    doc.styles["Heading 2"].paragraph_format.space_after=Pt(4)
    doc.styles["Caption"].font.size=Pt(9)
    doc.styles["Caption"].paragraph_format.space_after=Pt(6)
    doc.styles["Caption"].paragraph_format.line_spacing=1.05
    doc.core_properties.title="基于一维卷积神经网络的人体活动识别"
    doc.core_properties.subject="传感器组合与传统方法的对比研究"
    doc.core_properties.author="" if args.name=="待填写" else args.name
    doc.core_properties.keywords="UCI HAR, 一维CNN, 陀螺仪, 消融实验, Macro-F1"
    footer=section.footer.paragraphs[0]
    footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("第 ").font.size=Pt(9)
    field=OxmlElement("w:fldSimple");field.set(qn("w:instr"),"PAGE")
    footer._p.append(field)
    footer.add_run(" 页").font.size=Pt(9)

    def p(text, small=False):
        para=doc.add_paragraph(text)
        if small:
            para.paragraph_format.line_spacing=1.12
            for run in para.runs:run.font.size=Pt(9)
        return para
    def heading(text, newpage=False, level=1):
        h=doc.add_heading(text,level)
        if newpage:h.paragraph_format.page_break_before=True
        return h
    def caption(text):
        c=doc.add_paragraph(text,"Caption")
        c.alignment=WD_ALIGN_PARAGRAPH.CENTER
    def picture(name,width=16.3):
        para=doc.add_paragraph()
        para.alignment=WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.space_after=Pt(2)
        para.paragraph_format.keep_with_next=True
        para.add_run().add_picture(str(OUT/"figures"/(name+".png")),width=Cm(width))
    def table(headers, rows, widths, size=9.3):
        t=doc.add_table(rows=1,cols=len(headers))
        t.alignment=WD_TABLE_ALIGNMENT.CENTER
        t.autofit=False
        for cell,width in zip(t.rows[0].cells,widths):cell.width=Cm(width)
        for col,width in zip(t.columns,widths):col.width=Cm(width)
        for cell,text in zip(t.rows[0].cells,headers):cell.text=str(text)
        for row in rows:
            for cell,text,width in zip(t.add_row().cells,row,widths):
                cell.width=Cm(width);cell.text=str(text)
        borders=OxmlElement("w:tblBorders")
        for name in ["top","left","bottom","right","insideH","insideV"]:
            x=OxmlElement("w:"+name);x.set(qn("w:val"),"single");x.set(qn("w:sz"),"4");x.set(qn("w:color"),"D9D9D9");borders.append(x)
        t._tbl.tblPr.append(borders)
        for i,row in enumerate(t.rows):
            props=row._tr.get_or_add_trPr();props.append(OxmlElement("w:cantSplit"))
            if i==0:props.append(OxmlElement("w:tblHeader"))
            for cell in row.cells:
                cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                tc=cell._tc.get_or_add_tcPr()
                margins=OxmlElement("w:tcMar")
                for side,value in [("top",65),("bottom",65),("left",85),("right",85)]:
                    x=OxmlElement("w:"+side);x.set(qn("w:w"),str(value));x.set(qn("w:type"),"dxa");margins.append(x)
                tc.append(margins)
                if i==0:
                    shade=OxmlElement("w:shd");shade.set(qn("w:fill"),"E8EEF4");tc.append(shade)
                for para in cell.paragraphs:
                    para.alignment=WD_ALIGN_PARAGRAPH.CENTER
                    para.paragraph_format.space_after=Pt(0)
                    para.paragraph_format.line_spacing=1.1
                    for run in para.runs:
                        run.font.size=Pt(size);run.bold=i==0
                        set_fonts(run,"SimHei" if i==0 else "SimSun")
        doc.add_paragraph().paragraph_format.space_after=Pt(0)
        return t

    from report_content import add_content
    add_content(doc,p,heading,table,caption,picture,args,OUT,results,analysis,split,audit)
    for para in list(doc.paragraphs)+list(footer._parent.paragraphs):
        chinese="SimHei" if para.style.name in ["Title","Subtitle","Heading 1","Heading 2"] else "SimSun"
        for run in para.runs:
            set_fonts(run,chinese)
    SUB.mkdir(exist_ok=True)
    source_dir=ROOT/"tmp/report_source"
    source_dir.mkdir(parents=True,exist_ok=True)
    path=source_dir/"课程设计报告.docx"
    doc.save(path)
    claims={"macro_f1":{e:aggregate[e]['macro_f1_mean'] for e in aggregate},"cnn_delta_pp":delta('D','C'),
            "report_identity":{k:v for k,v in vars(args).items() if k!="results_dir"},"source_results":str((OUT/"results.json").relative_to(ROOT)).replace(chr(92),"/"),"target_pages":6}
    (SUB/"report_claims.json").write_text(json.dumps(claims,ensure_ascii=False,indent=2),encoding="utf-8")
    print(path)


if __name__=="__main__":
    main()

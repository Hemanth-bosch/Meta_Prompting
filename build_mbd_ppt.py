r"""[wrapper-only] Fill slides 9-32 of the MBD-UTA deck (structured content + matplotlib plots + tables +
diagrams + reference footers). Slides 1-8 and 33 are left as-is. Saves to a NEW file (original draft preserved).
Run: .meta-prompting\Scripts\python.exe build_mbd_ppt.py
Data sources: apples_compare/FINAL_COMPOSITE.csv, true_midas_baseline JSON, PRESERVED_GEPA run_summaries.
"""
import os, csv, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = r"C:\Users\BHS1RNG\Downloads\Final--2026-07-14_MBD-UTA_Multi_Prompt_Optimization.pptx"
OUT = r"C:\Users\BHS1RNG\Downloads\Final--2026-07-15_MBD-UTA_Multi_Prompt_Optimization_AIQ212.pptx"
ASSETS = os.path.join(ROOT, "ppt_assets"); os.makedirs(ASSETS, exist_ok=True)

# ── palette ──────────────────────────────────────────────────────────────────────────────────────
NAVY=RGBColor(0x00,0x45,0x82); BLUE=RGBColor(0x00,0x7D,0xC3); GREEN=RGBColor(0x00,0x96,0x60)
ORANGE=RGBColor(0xFF,0x6B,0x00); AMBER=RGBColor(0xFF,0xC0,0x00); GRAY=RGBColor(0x33,0x33,0x33)
RED=RGBColor(0xC0,0x00,0x00); WHITE=RGBColor(0xFF,0xFF,0xFF); MEDGRAY=RGBColor(0x59,0x59,0x59)
LBLUE=RGBColor(0xE7,0xF1,0xF9); LGRAY=RGBColor(0xF3,0xF5,0xF7); LGREEN=RGBColor(0xE7,0xF5,0xEF)
LOR=RGBColor(0xFC,0xEC,0xDE)
mNAVY="#004582"; mBLUE="#007DC3"; mGREEN="#009660"; mOR="#FF6B00"; mAMB="#E0A000"; mRED="#C00000"
FONT="Bosch Office Sans"

# ── data ─────────────────────────────────────────────────────────────────────────────────────────
FC = list(csv.DictReader(open(os.path.join(ROOT,"apples_compare","FINAL_COMPOSITE.csv"), encoding="utf-8")))
TM = json.load(open(os.path.join(ROOT,"apples_compare","true_midas_baseline","true_midas_baseline_final_20260715_114414.json"), encoding="utf-8"))
def comp(run, var, model=None):
    for r in FC:
        if r["run"]==run and r["variety"]==var and (model is None or r["model"]==model):
            return float(r["comp_mean"])
    return None
V5=["V1","V2","V3","V4","V5"]
VAL = [("GearLvrPosn_Vd#0","getBit branch"),("InjRly_PwrCtrl_1_1_0#0","Srv_TrnOffDly delay"),
       ("EGSSig_LamSymcp_1_6_0#2","LowpassT filter"),("oilsepactr_coorr#20","Srv_TrnOnDly delay"),
       ("Rail_TempLim_50.8.0#2","Srv_RampSwitch (pinned)")]

# ── pptx helpers ───────────────────────────────────────────────────────────────────────────────────
def _setpara(p, text, size, color, bold=False, align=PP_ALIGN.LEFT, italic=False, font=FONT, sa=3):
    p.alignment=align; p.space_after=Pt(sa); p.space_before=Pt(0)
    r=p.add_run(); r.text=text
    r.font.size=Pt(size); r.font.bold=bold; r.font.italic=italic; r.font.color.rgb=color; r.font.name=font
    return r
def textbox(sl,l,t,w,h,lines,size=12,color=GRAY,bold=False,align=PP_ALIGN.LEFT,anchor=None,sa=3):
    tb=sl.shapes.add_textbox(Inches(l),Inches(t),Inches(w),Inches(h)); tf=tb.text_frame; tf.word_wrap=True
    tf.margin_left=Inches(0.04); tf.margin_right=Inches(0.04); tf.margin_top=Inches(0.02); tf.margin_bottom=Inches(0.02)
    if anchor is not None: tf.vertical_anchor=anchor
    lines = lines if isinstance(lines,list) else [lines]
    for i,ln in enumerate(lines):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph()
        if isinstance(ln,tuple):
            o=ln[1]; _setpara(p,ln[0],o.get("size",size),o.get("color",color),o.get("bold",bold),
                              o.get("align",align),o.get("italic",False),sa=o.get("sa",sa))
        else:
            _setpara(p,ln,size,color,bold,align,sa=sa)
    return tb
def box(sl,l,t,w,h,fill,line=None,rounded=True,line_w=1.0):
    shp=sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
                            Inches(l),Inches(t),Inches(w),Inches(h))
    if fill is None: shp.fill.background()
    else: shp.fill.solid(); shp.fill.fore_color.rgb=fill
    if line is None: shp.line.fill.background()
    else: shp.line.color.rgb=line; shp.line.width=Pt(line_w)
    shp.shadow.inherit=False
    try: shp.adjustments[0]=0.06
    except Exception: pass
    return shp
def boxtext(sl,l,t,w,h,fill,lines,size=12,color=GRAY,bold=False,align=PP_ALIGN.LEFT,line=None,anchor=MSO_ANCHOR.TOP,pad=0.11):
    box(sl,l,t,w,h,fill,line)
    return textbox(sl,l+pad,t+0.05,w-2*pad,h-0.1,lines,size,color,bold,align,anchor)
def table(sl,rows,l,t,w,h,col_w,header_fill=NAVY,fsize=11,hsize=11,zebra=LGRAY):
    nr=len(rows); nc=len(rows[0])
    gt=sl.shapes.add_table(nr,nc,Inches(l),Inches(t),Inches(w),Inches(h)).table
    gt.first_row=False; gt.horz_banding=False
    for ci,cw in enumerate(col_w): gt.columns[ci].width=Inches(cw)
    for ri,row in enumerate(rows):
        for ci,val in enumerate(row):
            c=gt.cell(ri,ci); c.margin_left=Inches(0.06); c.margin_right=Inches(0.05)
            c.margin_top=Inches(0.01); c.margin_bottom=Inches(0.01); c.vertical_anchor=MSO_ANCHOR.MIDDLE
            c.fill.solid(); c.fill.fore_color.rgb = header_fill if ri==0 else (WHITE if ri%2 else zebra)
            p=c.text_frame.paragraphs[0]; c.text_frame.word_wrap=True
            _setpara(p,str(val),hsize if ri==0 else fsize,WHITE if ri==0 else GRAY,
                     ri==0 or (ci==0 and nc>2),PP_ALIGN.LEFT if ci==0 else PP_ALIGN.CENTER,sa=0)
    return gt
def set_subtitle(sl,text):
    for sh in sl.shapes:
        if sh.name=="TextBox 7" and sh.has_text_frame:
            sh.width=Inches(7.2)  # end before the top-right MiDAS/BGSW logos (~7.7in)
            p=sh.text_frame.paragraphs[0]
            if p.runs:
                p.runs[0].text=text; p.runs[0].font.size=Pt(13.5); p.runs[0].font.color.rgb=BLUE; p.runs[0].font.bold=False
                for r in p.runs[1:]: r.text=""
            else: _setpara(p,text,13.5,BLUE)
            return
def set_title(sl,text):
    for sh in sl.shapes:
        if sh.name=="TextBox 6" and sh.has_text_frame:
            p=sh.text_frame.paragraphs[0]
            if p.runs:
                p.runs[0].text=text
                for r in p.runs[1:]: r.text=""
            return
def set_note(sl,refs):
    for sh in sl.shapes:
        if sh.has_text_frame and sh.text_frame.text.strip().startswith("*File"):
            tf=sh.text_frame; tf.clear(); _setpara(tf.paragraphs[0],"* Ref: "+refs,9.5,MEDGRAY,italic=True,sa=0)
            return
    textbox(sl,0.16,6.07,11.6,0.3,[("* Ref: "+refs,{"size":9.5,"color":MEDGRAY,"italic":True})])
def add_img(sl,path,l,t,w):
    return sl.shapes.add_picture(path,Inches(l),Inches(t),Inches(w),None)

# ── matplotlib ──────────────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.size":11,"font.family":"DejaVu Sans","axes.edgecolor":"#999",
                     "axes.linewidth":0.8,"axes.grid":True,"grid.color":"#e6e6e6","grid.linewidth":0.7})
def save(fig,name,w,h):
    fig.set_size_inches(w,h); path=os.path.join(ASSETS,name)
    fig.savefig(path,dpi=170,bbox_inches="tight",facecolor="white"); plt.close(fig); return path
def barlabels(ax,bars,fmt="%.1f",size=9,dy=0.6):
    for b in bars:
        ax.annotate(fmt%b.get_height(),(b.get_x()+b.get_width()/2,b.get_height()+dy),
                    ha="center",va="bottom",fontsize=size,color="#222")

def plot_weights():
    labels=["M1 I/O .20","M4 Coverage .20","M2 Signal .15","M3 Structural .15","M7 DLL solve .15","M8 Functional .15"]
    sizes=[20,20,15,15,15,15]; cols=[mNAVY,mBLUE,mGREEN,mOR,mAMB,"#8E44AD"]
    fig,ax=plt.subplots(); ax.grid(False)
    ax.pie(sizes,labels=labels,colors=cols,startangle=90,counterclock=False,
           wedgeprops=dict(width=0.42,edgecolor="white",linewidth=2),textprops=dict(fontsize=9.5,color="#222"))
    ax.text(0,0,"Composite\n0-100%",ha="center",va="center",fontsize=12,fontweight="bold",color=mNAVY)
    return save(fig,"p_weights.png",4.6,4.0)
def plot_truemidas():
    rep=TM["report"]; mets=["m1","m2","m3","m4","m7","m8"]
    names=["M1\nI/O","M2\nSignal","M3\nStruct","M4\nCoverage","M7\nDLL","M8\nFunc"]
    vals=[rep[m]["mean"]*100 for m in mets]; errs=[rep[m]["std"]*100 for m in mets]
    fig,ax=plt.subplots(); cols=[mNAVY,mGREEN,mOR,mBLUE,mAMB,"#8E44AD"]
    bars=ax.bar(names,vals,color=cols,width=0.66,yerr=errs,capsize=3,ecolor="#888"); barlabels(ax,bars,dy=1.2)
    cm=rep["composite"]["mean"]; ax.axhline(cm,color=mRED,ls="--",lw=1.6)
    ax.text(5.4,cm+1.5,f"Composite {cm:.1f}%",color=mRED,fontsize=10.5,fontweight="bold",ha="right")
    ax.set_ylim(0,105); ax.set_ylabel("score (%)"); ax.set_title("true-MIDAS (run.py prompts) - per metric, 3 reps",fontsize=11)
    return save(fig,"p_truemidas.png",6.6,3.7)
def plot_baselines():
    labels=["#3 mini\nfinal","#1 mini\ninterm","#4 g4.1\nfinal","#2 g4.1\ninterm",
            "#5 mini\nnocot-fin","#6 mini\nnocot-int","#5 g4.1\nnocot-fin","#6 g4.1\nnocot-int"]
    rl=lambda run,m: sum(comp(run,v,m) for v in V5)/5
    vals=[rl("3","azure/gpt-4.1-mini"),rl("1","azure/gpt-4.1-mini"),rl("4","azure/gpt-4.1"),rl("2","azure/gpt-4.1"),
          comp("5","V1","azure/gpt-4.1-mini"),comp("6","V1","azure/gpt-4.1-mini"),
          comp("5","V1","azure/gpt-4.1"),comp("6","V1","azure/gpt-4.1")]
    cols=[mNAVY,mBLUE,mNAVY,mBLUE,mGREEN,"#5DADE2",mGREEN,"#5DADE2"]
    fig,ax=plt.subplots(); bars=ax.bar(labels,vals,color=cols,width=0.7); barlabels(ax,bars)
    cm=TM["report"]["composite"]["mean"]; ax.axhline(cm,color=mRED,ls="--",lw=1.4)
    ax.text(7.4,cm+0.6,"run.py 51.5",color=mRED,fontsize=9,ha="right")
    ax.set_ylim(0,80); ax.set_ylabel("composite (%)"); ax.set_title("dspy inference baselines (no GEPA) - run-level mean",fontsize=11)
    return save(fig,"p_baselines.png",9.6,3.5)
def plot_gepa():
    runs=["150-call\nCoT final","150-call\nCoT interm","300-call\nCoT final","300-call\nNoCoT final"]
    B=[51.42,60.35,61.30,58.93]; O=[70.59,67.60,71.87,65.73]
    x=np.arange(len(runs)); wdt=0.38
    fig,ax=plt.subplots()
    b1=ax.bar(x-wdt/2,B,wdt,label="baseline (seed)",color="#B0BEC5")
    b2=ax.bar(x+wdt/2,O,wdt,label="GEPA optimized",color=mGREEN); barlabels(ax,b1); barlabels(ax,b2)
    for i in range(len(runs)):
        ax.annotate(f"+{O[i]-B[i]:.1f}",(x[i]+wdt/2,O[i]+4),ha="center",fontsize=9,color=mRED,fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(runs); ax.set_ylim(0,90)
    ax.set_ylabel("composite (%)"); ax.legend(fontsize=9,loc="upper left"); ax.set_title("GEPA optimized runs (gpt-4.1-mini)",fontsize=11)
    return save(fig,"p_gepa.png",7.2,3.7)
def plot_combo(run, model, gp, nocot_val, fname, title, with_runpy):
    labels=[]; vals=[]; cols=[]
    if with_runpy:
        labels.append("run.py\nbaseline"); vals.append(TM["report"]["composite"]["mean"]); cols.append(mRED)
    for v in V5: labels.append(v); vals.append(comp(run,v,model)); cols.append(mBLUE)
    labels+=["V1\nno-CoT","GEPA\nbase","GEPA\nopt"]; vals+=[nocot_val,gp[0],gp[1]]; cols+=[mGREEN,"#B0BEC5",mNAVY]
    fig,ax=plt.subplots(); bars=ax.bar(labels,vals,color=cols,width=0.72); barlabels(ax,bars)
    ax.set_ylim(0,90); ax.set_ylabel("composite (%)"); ax.set_title(title,fontsize=11)
    return save(fig,fname,9.4,3.6)
def plot_variety(run,model,nocot_val,fname,title):
    labels=list(V5)+["V1\nno-CoT"]; vals=[comp(run,v,model) for v in V5]+[nocot_val]; cols=[mBLUE]*5+[mGREEN]
    fig,ax=plt.subplots(); bars=ax.bar(labels,vals,color=cols,width=0.66); barlabels(ax,bars)
    ax.set_ylim(0,85); ax.set_ylabel("composite (%)"); ax.set_title(title,fontsize=11)
    return save(fig,fname,7.6,3.6)
def plot_m4kb():
    labels=["run.py\nbaseline","dspy\nno-inject","inject\nNEW KB (V4)","inject NEW\n+retry (V5)"]
    vals=[21.9,48.8,62.8,75.7]; cols=[mRED,"#B0BEC5",mGREEN,mNAVY]
    fig,ax=plt.subplots(); bars=ax.bar(labels,vals,color=cols,width=0.62); barlabels(ax,bars,dy=1.0)
    ax.set_ylim(0,92); ax.set_ylabel("M4 coverage (%)"); ax.set_title("M4 rises with KB guideline injection (gpt-4.1, final-KB)",fontsize=10.5)
    return save(fig,"p_m4kb.png",6.4,2.55)
def plot_perexample(base, fname):
    f3=lambda d: sum(d[:3])/len(d[:3])
    tmk=[k for k in TM["per_example_composite"] if k.split("#")[0]==base][0]
    before=f3(TM["per_example_composite"][tmk])
    b3=json.load(open(os.path.join(ROOT,"apples_compare","batch_meanstd","batch_meanstd_final_20260714_164247.json"),encoding="utf-8"))
    pe=b3["per_example_composite"]["azure/gpt-4.1-mini"]["V5"]
    v5k=[k for k in pe if k.split("#")[0]==base][0]; after=f3(pe[v5k])
    fig,ax=plt.subplots(); bars=ax.bar(["run.py\nbaseline","dspy V5\n(injNEW)"],[before,after],color=[mRED,mNAVY],width=0.5)
    barlabels(ax,bars,dy=0.8); ax.set_ylim(0,max(before,after)*1.3+5); ax.set_ylabel("composite (%)")
    ax.set_title("per-example composite",fontsize=10.5)
    return save(fig,fname,3.4,3.0), before, after

print("generating plots...")
P_W=plot_weights(); P_TM=plot_truemidas(); P_BASE=plot_baselines(); P_GEPA=plot_gepa()
P_C_FIN=plot_combo("3","azure/gpt-4.1-mini",(51.42,71.87),comp("5","V1","azure/gpt-4.1-mini"),"p_combo_fin.png","Final-KB . gpt-4.1-mini . run.py vs varieties vs GEPA",True)
P_C_INT=plot_combo("1","azure/gpt-4.1-mini",(60.35,67.60),comp("6","V1","azure/gpt-4.1-mini"),"p_combo_int.png","Intermediate-KB . gpt-4.1-mini . varieties vs GEPA",False)
P_G41_FIN=plot_variety("4","azure/gpt-4.1",comp("5","V1","azure/gpt-4.1"),"p_g41_fin.png","gpt-4.1 . Final-KB . V1-V5 + no-CoT")
P_G41_INT=plot_variety("2","azure/gpt-4.1",comp("6","V1","azure/gpt-4.1"),"p_g41_int.png","gpt-4.1 . Intermediate-KB . V1-V5 + no-CoT")
P_M4KB=plot_m4kb()

pr=Presentation(SRC); S=pr.slides
print(f"loaded deck: {len(S)} slides")

def s09():
    sl=S[8]; set_title(sl,"Walkthrough: One example - InjRly weakspot")
    set_subtitle(sl,"InjRly_PwrCtrl (Srv_TrnOffDly delay): the 4 LLM stages -> generated test case -> reference-free checks")
    stages=[("1  Locate",NAVY,"Find weakspot in\nequations.json"),("2  Context",NAVY,"Dependency tree\nto leaf inputs"),
            ("3a Extract",BLUE,"LLM#1 -> context\nanalysis"),("3b TestGen",BLUE,"LLM#2 -> test\ncases (JSON)"),
            ("4 Solve",ORANGE,"LLM#3/#4 + DLL\nverify & retry")]
    x=0.30; w=2.18
    for i,(h,c,d) in enumerate(stages):
        boxtext(sl,x,1.35,w,0.95,c,[(h,{"size":12,"color":WHITE,"bold":True}),(d,{"size":10,"color":WHITE})])
        if i<4: textbox(sl,x+w-0.02,1.6,0.2,0.4,[(">",{"size":16,"color":BLUE,"bold":True})])
        x+=w+0.09
    boxtext(sl,0.30,2.55,5.55,3.30,LGRAY,[("Generated test case (pipeline output)",{"size":13,"color":NAVY,"bold":True})],line=NAVY)
    textbox(sl,0.45,3.02,5.30,2.75,[
        ("Pre-Condition",{"size":11.5,"color":NAVY,"bold":True}),("  input signals pinned to a start state",{"size":10.5}),
        ("Steps = Action + Wait",{"size":11.5,"color":NAVY,"bold":True}),("  set leaf inputs -> wait N x raster dT",{"size":10.5}),
        ("Expected Results",{"size":11.5,"color":NAVY,"bold":True}),("  the weakspot OUTPUT the DLL must reproduce",{"size":10.5}),
        ("Explanation (per test case)",{"size":11.5,"color":NAVY,"bold":True}),("  why this stimulus hits the decision point",{"size":10.5})])
    boxtext(sl,6.05,2.55,5.65,3.30,LBLUE,[("How it is judged (reference-free)",{"size":13,"color":NAVY,"bold":True})],line=BLUE)
    textbox(sl,6.20,3.02,5.40,2.75,[
        ("M1  Expected verifies a REAL output; output never an input",{"size":10.5}),
        ("M2  every signal name resolves (no placeholders)",{"size":10.5}),
        ("M3  numeric values, waits multiple of dT, no overlap",{"size":10.5}),
        ("M4  gpt-5.2 judge: KB guideline scenarios covered?",{"size":10.5}),
        ("M7  DLL back-solve each step: LHS == RHS",{"size":10.5}),
        ("M8  DLL runs whole sequence: reproduces outputs?",{"size":10.5})],sa=4)
    set_note(sl,"extract_mbd_full.py (forward); val example InjRly_PwrCtrl_1_1_0#0; signatures/mbd.py")

def s10():
    sl=S[9]; set_subtitle(sl,"One reference-free composite (0-100%) = 6 weighted checks; no labelled ground truth")
    rows=[["#","What it verifies","On","Wt"],
          ["M1","I/O: Expected is a real output; output never used as input; constants never driven","P2 raw","0.20"],
          ["M2","Signal resolution: names resolve to real signals + required inputs present","final","0.15"],
          ["M3","Structural: numeric values, waits multiple of dT, no precond/expected overlap","P2 raw","0.15"],
          ["M4","Coverage: LLM judge (gpt-5.2) checks KB guideline scenarios covered","judge","0.20"],
          ["M7","DLL pass rate: back-solve each step (LHS==RHS), retry-graded","final","0.15"],
          ["M8","Functional: DLL runs whole step sequence, reproduces outputs","DLL","0.15"]]
    table(sl,rows,0.30,1.45,7.15,3.7,[0.42,5.06,0.72,0.55],fsize=10.2,hsize=10.5)
    add_img(sl,P_W,7.75,1.4,3.9)
    boxtext(sl,7.7,5.0,4.0,0.95,LGRAY,[("composite = .20 M1 + .15 M2 + .15 M3 + .20 M4 + .15 M7 + .15 M8",{"size":10.3,"color":NAVY,"bold":True}),
        ("M6 (boundary) removed; M7/M8 lowered .20 -> .15 (2026-07-09)",{"size":9,"color":MEDGRAY,"italic":True})],line=NAVY,anchor=MSO_ANCHOR.MIDDLE)
    set_note(sl,"evaluation/mbd_metric.py (WEIGHT_M1..M8; _m1_io_classification .. _m8_functional_correctness); M4 judge azure/gpt-5.2")

def s11():
    sl=S[10]; set_subtitle(sl,"10 weakspots across ECU function blocks -> 60/40 stratified split; 5 pinned validation weakspots")
    rows=[["Weakspot","Block type"],["InjRly_PwrCtrl_1_1_0","Srv_TrnOffDly"],["GbxFanCoorn_Sp_1_0_2","coorn/delay"],
          ["oilsepactr_coorr","Srv_TrnOnDly"],["EGSSig_LamSymcp_1_6_0","LowpassT"],["Rail_TempLim_50.8.0","Srv_RampSwitch"],
          ["moftqptdveh_fil","filter"],["AltIO_DD","getBit"],["a2mamg","arithmetic"],["GearLvrPosn_Vd","getBit"],["IT1A_OdoCalc_1.5.0","counter"]]
    table(sl,rows,0.30,1.45,4.55,4.3,[3.05,1.5],fsize=10.2,hsize=10.5)
    boxtext(sl,5.15,1.45,6.55,0.5,NAVY,[("Stratified split   train 60% / val 40% / test 0%",{"size":12.5,"color":WHITE,"bold":True})],anchor=MSO_ANCHOR.MIDDLE)
    boxtext(sl,5.15,2.1,3.15,3.65,LGRAY,[("TRAIN pool",{"size":12,"color":NAVY,"bold":True}),("GEPA reflects on these",{"size":10,"color":MEDGRAY,"italic":True}),
        ("",{"size":4}),("- arithmetic / solver-testable",{"size":10.5}),("- bit-op weakspots",{"size":10.5}),
        ("- duplicated block types spread",{"size":10.5})],line=NAVY)
    boxtext(sl,8.55,2.1,3.15,3.65,LBLUE,[("VALIDATION (5)",{"size":12,"color":NAVY,"bold":True}),("scored every iteration",{"size":10,"color":MEDGRAY,"italic":True}),("",{"size":4})]+
        [(f"- {k.split('#')[0]}",{"size":9.6}) for k,_ in VAL]+[("Rail_TempLim#2 pinned (expert-ok)",{"size":9,"color":MEDGRAY,"italic":True})],line=BLUE)
    set_note(sl,"utils/mbd_data.py (stratified_split_mbd_dataset 0.6/0.4/0.0; _FORCE_VAL_EXAMPLES); extract_mbd_full.py")

def s12():
    sl=S[11]; set_subtitle(sl,"Ablation matrix over 3 axes: inject KB into Prompt-2 . M4-judge KB (old/new) . solver retries")
    rows=[["Variety","Inject KB into Prompt-2","M4 judge KB","Retries","Purpose"],
          ["V1","no","OLD","1","no-inject baseline (old judge)"],["V2","no","NEW","1","no-inject baseline (new judge)"],
          ["V3","yes (OLD guidelines)","OLD","1","inject old KB"],["V4","yes (NEW guidelines)","NEW","1","inject new KB"],
          ["V5","yes (NEW guidelines)","NEW","4","inject new KB + more retries"]]
    table(sl,rows,0.30,1.55,11.4,2.4,[1.0,3.3,1.5,1.1,4.5],fsize=11,hsize=11)
    boxtext(sl,0.30,4.3,3.66,1.5,LBLUE,[("Axis A - Injection",{"size":12,"color":NAVY,"bold":True}),
        ("append KB guideline scenarios into Prompt-2 so the generator can target them",{"size":10.5})],line=BLUE)
    boxtext(sl,4.17,4.3,3.66,1.5,LGREEN,[("Axis B - M4 judge KB",{"size":12,"color":GREEN,"bold":True}),
        ("score coverage vs OLD (ASCET HTML) or NEW (BU Excel) guidelines",{"size":10.5})],line=GREEN)
    boxtext(sl,8.04,4.3,3.66,1.5,LOR,[("Axis C - Retries",{"size":12,"color":ORANGE,"bold":True}),
        ("solver retry budget 1 vs 4; M7 grading pinned at ref=4",{"size":10.5})],line=ORANGE)
    set_note(sl,"_apples_batch_meanstd.py (VLAB; per-variety inject_kb_guidelines / max_retries / _set_kb); M7_REF_PIN=4")

def s13():
    sl=S[12]; set_subtitle(sl,"Same task, two prompt structures: dspy ChainOfThought vs plain Predict")
    boxtext(sl,0.30,1.5,5.6,3.25,LBLUE,[("CoT - dspy.ChainOfThought (default)",{"size":14,"color":NAVY,"bold":True})],line=NAVY)
    textbox(sl,0.5,2.1,5.2,2.6,[("- emits a reasoning scratchpad before output",{"size":11.5}),
        ("- wrapper threads reasoning into Prompt-2 (keeps run.py context)",{"size":11.5}),
        ("- richer intermediate signal for the generator",{"size":11.5}),("- more tokens / higher cost",{"size":11.5}),
        ("- default for all main experiments",{"size":11.5,"color":NAVY,"bold":True})],sa=9)
    boxtext(sl,6.10,1.5,5.6,3.25,LGRAY,[("No-CoT - dspy.Predict (--no-cot)",{"size":14,"color":ORANGE,"bold":True})],line=ORANGE)
    textbox(sl,6.3,2.1,5.2,2.6,[("- direct output, no reasoning field",{"size":11.5}),("- fewer tokens, cheaper & faster",{"size":11.5}),
        ("- tests whether the scratchpad actually helps",{"size":11.5}),("- run as a V1 variant (both models, both KBs)",{"size":11.5}),
        ("- result: close to CoT, slightly lower",{"size":11.5,"color":ORANGE,"bold":True})],sa=9)
    boxtext(sl,0.30,4.9,11.4,0.95,LGREEN,[("Measured (final-KB, V1):   gpt-4.1  CoT 59.1%  vs  no-CoT 58.8%      |      mini  CoT 55.4%  vs  no-CoT 54.8%",{"size":11.5,"color":GREEN,"bold":True,"align":PP_ALIGN.CENTER}),
        ("the reasoning scratchpad helps only marginally -> no-CoT is a cheaper, near-equivalent option",{"size":10,"color":GRAY,"align":PP_ALIGN.CENTER})],line=GREEN,anchor=MSO_ANCHOR.MIDDLE)
    set_note(sl,"extract_mbd_full.py (MBDFullPipelineModule use_cot -> ChainOfThought/Predict); _apples_batch_meanstd.py --no-cot; FINAL_COMPOSITE.csv")

def s14():
    sl=S[13]; set_subtitle(sl,"Which output a test must verify: the weakspot's FINAL output vs its INTERMEDIATE signal")
    chain=[("leaf inputs",LGRAY,GRAY),("weakspot op",AMBER,GRAY),("intermediate\nsignal",BLUE,WHITE),("downstream\nops",LGRAY,GRAY),("FINAL\noutput",NAVY,WHITE)]
    x=0.5; w=1.95
    for i,(t,f,c) in enumerate(chain):
        boxtext(sl,x,1.55,w,0.85,f,[(t,{"size":11,"color":c,"bold":True,"align":PP_ALIGN.CENTER})],anchor=MSO_ANCHOR.MIDDLE)
        if i<4: textbox(sl,x+w-0.03,1.75,0.25,0.4,[(">",{"size":16,"color":GRAY,"bold":True})])
        x+=w+0.13
    boxtext(sl,0.30,2.9,5.6,2.6,LBLUE,[("required_io.json  =  FINAL",{"size":13.5,"color":NAVY,"bold":True}),
        ("test must reach & verify the weakspot's ultimate output",{"size":11}),
        ("harder: often a longer solve chain through downstream ops",{"size":11})],line=NAVY)
    boxtext(sl,6.10,2.9,5.6,2.6,LGRAY,[("required_io_intermediate.json = INTERMEDIATE",{"size":13,"color":BLUE,"bold":True}),
        ("verifies the weakspot's immediate output (e.g. filter value)",{"size":11}),
        ("easier target; steers Prompt-2 via --required-io-path",{"size":11})],line=BLUE)
    set_note(sl,"utils/required_io.json vs utils/required_io_intermediate.json; extract_mbd_full.py --required-io-path / --required-io-outputs")

def s15():
    sl=S[14]; set_subtitle(sl,"Reference guidelines M4 judges against (and optionally injected into Prompt-2)")
    boxtext(sl,0.30,1.5,5.6,2.15,LGRAY,[("OLD KB - ASCET block manual (HTML)",{"size":13,"color":MEDGRAY,"bold":True})],line=MEDGRAY)
    textbox(sl,0.5,2.02,5.2,1.55,[("- historical, validated scenarios per block type",{"size":10.5}),("- source: MBD knowledge-base HTML (docupedia)",{"size":10.5}),
        ("- utils/weakspot_guidelines.json",{"size":9.5,"color":MEDGRAY,"italic":True})],sa=6)
    boxtext(sl,6.10,1.5,5.6,2.15,LGREEN,[("NEW KB - BU Excel (FCI mapping)",{"size":13,"color":GREEN,"bold":True})],line=GREEN)
    textbox(sl,6.3,2.02,5.2,1.55,[("- updated scenarios from 2026 Bosch specs",{"size":10.5}),("- source: FCI test-case Excel template (from PS)",{"size":10.5}),
        ("- utils/new_weakspot_guidelines.json",{"size":9.5,"color":MEDGRAY,"italic":True})],sa=6)
    add_img(sl,P_M4KB,0.30,3.75,6.35)
    boxtext(sl,6.95,3.8,4.75,1.95,LBLUE,[("Why the NEW KB matters",{"size":12.5,"color":NAVY,"bold":True}),
        ("injecting these guidelines into Prompt-2 (V3/V4/V5) is what lifts M4 coverage",{"size":10.5}),
        ("Layered lookup:  NEW KB -> OLD KB -> neutral 0.5",{"size":10.5,"color":NAVY,"bold":True}),
        ("block type resolved by ASCET suffix-stripping (find_block_type)",{"size":9.5,"color":MEDGRAY,"italic":True})],line=BLUE)
    set_note(sl,"utils/kb_parser.py (resolve_guidelines, find_block_type, format_kb_guidelines_block); utils/new_weakspot_guidelines.json; FINAL_COMPOSITE per-metric")

def s16():
    sl=S[15]; set_subtitle(sl,"Reflective prompt optimization: propose -> evaluate on val -> reflect (gpt-5.2) -> keep if better -> repeat")
    steps=[("Seed",NAVY,"4 prompts:\nContext, TestGen,\nSolve, Retry"),("Evaluate",BLUE,"run val;\nM1-M8 score +\ntext feedback"),
           ("Reflect",GREEN,"gpt-5.2 rewrites\nONE predictor\n(round-robin)"),("Accept",ORANGE,"keep candidate\nif val score\nimproves"),
           ("Loop",NAVY,"until budget\nspent\n(150 / 300)")]
    x=0.30; w=2.15
    for i,(h,c,d) in enumerate(steps):
        boxtext(sl,x,1.55,w,1.15,c,[(h,{"size":13,"color":WHITE,"bold":True}),(d,{"size":10,"color":WHITE})])
        if i<4: textbox(sl,x+w-0.04,1.95,0.22,0.4,[(">",{"size":17,"color":GRAY,"bold":True})])
        x+=w+0.12
    textbox(sl,2.5,2.75,7.2,0.35,[("|___________ loop: round-robin over the 4 predictors ___________|",{"size":10,"color":MEDGRAY,"align":PP_ALIGN.CENTER})])
    rows=[["Parameter","Setting"],["Student model","azure/gpt-4.1-mini (temp 0)"],["Reflection model","azure/gpt-5.2 (rewrites prompts)"],
          ["Budget","--gepa-max-metric-calls = 150 or 300 (rollouts, not iters)"],["Predictors","4, one rewritten per iteration (round-robin)"],
          ["Feedback metric","mbd_metric_with_feedback (score + plain-English feedback)"],["Threads / cost","1 when DLL active; hard --max-cost-usd cap"]]
    table(sl,rows,0.30,3.35,11.4,2.45,[2.6,8.8],fsize=10.5,hsize=11)
    set_note(sl,"extract_mbd_full.py (GEPA config, mbd_metric_with_feedback); optimization/opt.py (compile); auto='light' ~1495 calls")

def s17():
    sl=S[16]; set_subtitle(sl,"run.py's OWN prompts + transport, scored with the SAME M1-M8 -> the honest baseline (51.5%)")
    add_img(sl,P_TM,0.30,1.5,6.7)
    rows=[["Val weakspot","comp %"]]+[[k.split("#")[0][:20],f"{sum(v[:3])/3:.1f}"] for k,v in TM["per_example_composite"].items()]
    table(sl,rows,7.35,1.55,4.3,3.0,[3.1,1.2],fsize=10.2,hsize=10.5)
    rep=TM["report"]
    boxtext(sl,7.35,4.75,4.3,1.05,LGRAY,[(f"Composite  {rep['composite']['mean']:.1f}%  +/- {rep['composite']['std']:.1f}",{"size":13,"color":NAVY,"bold":True}),
        ("M8~0: run.py output does not reproduce the DLL functional check",{"size":9,"color":MEDGRAY,"italic":True})],line=NAVY,anchor=MSO_ANCHOR.MIDDLE)
    set_note(sl,"_true_midas_baseline.py; apples_compare/true_midas_baseline/true_midas_baseline_final_20260715_114414.json")

def s18():
    sl=S[17]; set_subtitle(sl,"dspy inference, no GEPA: V1-V5 (CoT) + V1 no-CoT, over 2 models x 2 KBs (final & intermediate)")
    add_img(sl,P_BASE,0.30,1.55,11.35)
    boxtext(sl,0.30,5.35,11.4,0.5,LBLUE,[("dspy prompt structure alone (no optimization) already sits ~5-14 pts above run.py's 51.5%",{"size":11.5,"color":NAVY,"bold":True,"align":PP_ALIGN.CENTER})],line=BLUE,anchor=MSO_ANCHOR.MIDDLE)
    set_note(sl,"apples_compare/FINAL_COMPOSITE.csv (runs 1-6); batch_meanstd/*.json; 3 reps each (2 + top-up)")

def s19():
    sl=S[18]; set_subtitle(sl,"All completed GEPA runs (gpt-4.1-mini): baseline (seed) -> optimized best")
    add_img(sl,P_GEPA,0.30,1.5,6.9)
    rows=[["Run","Base","Opt","D","$"],["150 CoT final","51.4","70.6","+19.2","1.28"],["150 CoT interm","60.4","67.6","+7.2","1.39"],
          ["300 CoT final","61.3","71.9","+10.6","2.70"],["300 NoCoT final","58.9","65.7","+6.8","1.91"]]
    table(sl,rows,7.4,1.7,4.3,2.5,[1.85,0.62,0.62,0.66,0.55],fsize=10,hsize=10)
    boxtext(sl,7.4,4.5,4.3,1.3,LGREEN,[("Best: 300-call CoT -> 71.9%",{"size":12.5,"color":GREEN,"bold":True}),
        ("optimized prompts saved for all 4 runs",{"size":10}),("3 of 4 predictors genuinely rewritten",{"size":10})],line=GREEN,anchor=MSO_ANCHOR.MIDDLE)
    set_note(sl,"log/PRESERVED_GEPA/GEPA_*_run_summary.json (150call final/interm; 300call cot/nocot COMPLETED)")

def s20():
    sl=S[19]; set_subtitle(sl,"Final-KB, gpt-4.1-mini: run.py baseline -> dspy varieties -> best GEPA (single view)")
    add_img(sl,P_C_FIN,0.30,1.55,11.35)
    boxtext(sl,0.30,5.35,11.4,0.5,LGREEN,[("run.py 51.5%  ->  dspy varieties ~55-61%  ->  GEPA optimized 70.6 / 71.9%",{"size":12,"color":GREEN,"bold":True,"align":PP_ALIGN.CENTER})],line=GREEN,anchor=MSO_ANCHOR.MIDDLE)
    set_note(sl,"FINAL_COMPOSITE.csv (run 3 V1-V5, run 5 mini nocot); GEPA_final_cot_300/150 run_summary; true_midas JSON")

def s21():
    sl=S[20]; set_subtitle(sl,"Intermediate-KB, gpt-4.1-mini: dspy varieties -> best GEPA")
    add_img(sl,P_C_INT,0.30,1.55,11.35)
    boxtext(sl,0.30,5.35,11.4,0.5,LBLUE,[("intermediate-KB varieties ~58-63%  ->  GEPA 67.6% (baseline already higher than final-KB)",{"size":11.5,"color":NAVY,"bold":True,"align":PP_ALIGN.CENTER})],line=BLUE,anchor=MSO_ANCHOR.MIDDLE)
    set_note(sl,"FINAL_COMPOSITE.csv (run 1 V1-V5, run 6 mini nocot); GEPA_intermediate_150call run_summary")

def s22():
    sl=S[21]; set_subtitle(sl,"gpt-4.1 (larger model), Final-KB: V1-V5 + V1 no-CoT")
    add_img(sl,P_G41_FIN,0.30,1.6,8.2)
    boxtext(sl,8.7,1.9,3.0,3.2,LGRAY,[("gpt-4.1 vs mini",{"size":12.5,"color":NAVY,"bold":True}),
        ("- larger model lifts every variety ~3-6 pts",{"size":10.5}),("- V4/V5 (inject NEW KB) strongest",{"size":10.5}),
        ("- no-CoT close to CoT",{"size":10.5})],line=NAVY)
    set_note(sl,"FINAL_COMPOSITE.csv (run 4 gpt-4.1 final V1-V5; run 5 gpt-4.1 nocot)")

def s23():
    sl=S[22]; set_subtitle(sl,"gpt-4.1 (larger model), Intermediate-KB: V1-V5 + V1 no-CoT")
    add_img(sl,P_G41_INT,0.30,1.6,8.2)
    boxtext(sl,8.7,1.9,3.0,3.2,LGRAY,[("gpt-4.1 interm",{"size":12.5,"color":NAVY,"bold":True}),
        ("- highest raw scores of any inference set",{"size":10.5}),("- V4/V5 reach ~68%",{"size":10.5}),
        ("- ceiling for un-optimized prompts",{"size":10.5})],line=NAVY)
    set_note(sl,"FINAL_COMPOSITE.csv (run 2 gpt-4.1 interm V1-V5; run 6 gpt-4.1 nocot)")

def per_example_slide(idx, exkey, label):
    sl=S[idx-1]; base=exkey.split("#")[0]; set_subtitle(sl,f"{base}  -  {label}")
    before=after=None
    try:
        png,before,after=plot_perexample(base,f"p_ex_{idx}.png"); add_img(sl,png,0.35,1.7,3.3)
    except Exception as e:
        print("perex fail",base,e)
    boxtext(sl,4.0,1.6,7.7,1.5,LBLUE,[("Weakspot",{"size":13,"color":NAVY,"bold":True}),(f"{base}   -   {label}",{"size":12}),
        ("a critical ECU decision point; the pipeline generates stimulus + expected output that exercises it",{"size":10.5})],line=BLUE)
    boxtext(sl,4.0,3.25,7.7,1.15,LGREEN,[("Per-example improvement (run.py -> dspy V5 injNEW)",{"size":12.5,"color":GREEN,"bold":True}),
        (f"composite {before:.1f}% -> {after:.1f}%   (+{after-before:.1f} pp)" if before is not None else "see aggregate",{"size":12})],line=GREEN,anchor=MSO_ANCHOR.MIDDLE)
    boxtext(sl,4.0,4.55,7.7,1.25,LGRAY,[("Aggregate GEPA effect (all 5 val examples)",{"size":12.5,"color":NAVY,"bold":True}),
        ("GEPA lifts the seed baseline 51.4% -> 70.6% (+19.2 pp, 150-call) / 71.9% (300-call)",{"size":11})],line=NAVY,anchor=MSO_ANCHOR.MIDDLE)
    set_note(sl,"true_midas JSON (run.py per-example) vs batch_meanstd_final_20260714_164247.json (dspy V5); GEPA run_summary")

def issue_slide(idx, items, sub):
    sl=S[idx-1]; set_subtitle(sl,sub); t=1.5
    for head,body,col,lc in items:
        boxtext(sl,0.30,t,11.4,0.92,lc,[(head,{"size":13,"color":col,"bold":True}),(body,{"size":11})],line=col,anchor=MSO_ANCHOR.MIDDLE)
        t+=1.05
    return sl
def s29():
    sl=issue_slide(29,[("Pre / post-process logic is the ceiling","Fixed bridge, mux flattening and dedup constrain what the LLM can express; some agentic-flow logic still limits quality.",NAVY,LBLUE),
        ("M2 seed gap (placeholder Expected)","In several val examples the seed pipeline emits placeholder Expected signals (out/$op21) instead of real ones - GEPA targets this.",ORANGE,LOR),
        ("Function-call weakspots have no equations","M7 is neutral (0.5); correctness rests on M8's DLL run. getBit/branch cases are structurally different.",GREEN,LGREEN)],
        "Known gaps in the pipeline itself (1/4)")
    set_note(sl,"extract_mbd_full.py (forward, post-process); evaluation/mbd_metric.py (_m2, _m7); apples_compare/FINAL_REPORT.txt")
def s30():
    sl=issue_slide(30,[("No ground truth -> reference-free metric","Scores are rule + LLM-judge based; a high composite is a proxy, not verified correctness.",NAVY,LBLUE),
        ("M4 judge (gpt-5.2) non-determinism","Cache-off judge varies run-to-run; mitigated by content-memoization + fixed seed baselines.",ORANGE,LOR),
        ("Guideline injection can teach-to-test","Injecting KB into Prompt-2 lifts M4 cosmetically but can degrade M7/M2 - not a pure functional win.",RED,LOR),
        ("M8 structural-0 vs functional-0","M8=0 can be a structural gate (wrong I/O, unexported fn), not a quality signal - flagged as INFO.",GREEN,LGREEN)],
        "Trusting the numbers: caveats of a no-ground-truth metric (2/4)")
    set_note(sl,"evaluation/mbd_metric.py (_m4 memoization, _m8 gates); apples_compare/FINAL_REPORT.txt; guideline-injection matrix")
def s31():
    sl=issue_slide(31,[("32-bit DLL can hang the run","A frozen is_function_available/solve stalled a run ~2h; fixed with a kill+restart watchdog (solve_with_dll_timeout).",NAVY,LBLUE),
        ("Determinism & caching","Key-auth + cache + temp0 student gives deterministic evals; az-login token provider forces cache off.",BLUE,LBLUE),
        ("Cost control","gpt-5.2 judge + volume drove cost; added a hard --max-cost-usd cap and per-run cost in run_summary.",ORANGE,LOR),
        ("Incomplete repeats standardized","Some apples 3rd repeats were partial; topped up to a uniform, hole-free 3 reps.",GREEN,LGREEN)],
        "Operational robustness fixes shipped this cycle (3/4)")
    set_note(sl,"evaluation/mbd_metric.py (watchdog); config.py / .env (cache, key-auth); utils/cost_guard.py; _final_tables.py")
def s32():
    sl=issue_slide(32,[("Re-define / re-design the use-case logic","Biggest lever: simplify pre/post-process + agentic flow so the metric ceiling rises.",NAVY,LBLUE),
        ("Forward generation (next slide, A)","Let the LLM pick reasonable leaf inputs, run forward through the DLL, and read off the Expected result.",GREEN,LGREEN),
        ("Numeric search when a target is required (B)","Replace LLM inversion with bounded grid/bisection over leaf inputs using the DLL as oracle.",ORANGE,LOR),
        ("Keep GEPA on the optimized flow","Once the flow is cleaner, re-run GEPA - the +19 pp shows the headroom is real.",BLUE,LBLUE)],
        "Priority roadmap (4/4) - detailed on the next slide")
    set_note(sl,"apples_compare/FINAL_REPORT.txt; slide 33 (Alternate Recommendations A/B)")

for f in (s09,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23): f()
for idx,(exk,lab) in zip([24,25,26,27,28],VAL): per_example_slide(idx,exk,lab)
for f in (s29,s30,s31,s32): f()

pr.save(OUT)
print("SAVED:", OUT)

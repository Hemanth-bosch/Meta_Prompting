r"""[wrapper-only] Replace ONLY the plot images in the user-edited MBD-UTA deck (no text/format/position change).
Improvements: no gridlines; bars = mean of 3 reps with min-max whiskers where applicable; cleaner styling;
caption noting mean/min-max. Per-example slides: before = run.py baseline, after = best GEPA-optimized
(from ppt_assets/after_pe.json, produced by the --eval-only run). Each plot is regenerated at its shape's
exact aspect ratio and swapped in place (same L/T/W/H). Run: .meta-prompting\Scripts\python.exe edit_ppt_plots.py
"""
import os, csv, json, re, shutil
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pptx import Presentation
from pptx.util import Emu

ROOT=os.path.dirname(os.path.abspath(__file__))
DECK=r"C:\Users\BHS1RNG\Downloads\Final--2026-07-15_MBD-UTA_Multi_Prompt_Optimization_AIQ212.pptx"
ASSETS=os.path.join(ROOT,"ppt_assets"); os.makedirs(ASSETS,exist_ok=True)
B=os.path.join(ROOT,"apples_compare","batch_meanstd")
mNAVY="#004582"; mBLUE="#007DC3"; mGREEN="#009660"; mOR="#FF6B00"; mAMB="#E0A000"; mRED="#C00000"; mGRAY="#B0BEC5"

FC=list(csv.DictReader(open(os.path.join(ROOT,"apples_compare","FINAL_COMPOSITE.csv"),encoding="utf-8")))
TM=json.load(open(os.path.join(ROOT,"apples_compare","true_midas_baseline","true_midas_baseline_final_20260715_114414.json"),encoding="utf-8"))
B4=json.load(open(os.path.join(B,"batch_meanstd_final_20260715_003950.json"),encoding="utf-8"))["raw_per_repeat"]["azure/gpt-4.1"]
AFTER=None
if os.path.exists(os.path.join(ASSETS,"after_pe.json")):
    AFTER=json.load(open(os.path.join(ASSETS,"after_pe.json"),encoding="utf-8"))
V5=["V1","V2","V3","V4","V5"]
VAL=["GearLvrPosn_Vd","InjRly_PwrCtrl_1_1_0","EGSSig_LamSymcp_1_6_0","oilsepactr_coorr","Rail_TempLim_50.8.0"]
VALLAB={"GearLvrPosn_Vd":"getBit","InjRly_PwrCtrl_1_1_0":"Srv_TrnOffDly","EGSSig_LamSymcp_1_6_0":"LowpassT",
        "oilsepactr_coorr":"Srv_TrnOnDly","Rail_TempLim_50.8.0":"Srv_RampSwitch"}

def vreps(run,var,model):
    for r in FC:
        if r["run"]==run and r["variety"]==var and r["model"]==model:
            return [float(x) for x in r["reps"].split(";") if x]
def vstats(run,var,model):
    rp=vreps(run,var,model); return (sum(rp)/len(rp),min(rp),max(rp))
def runstats(run,model,varis):
    if len(varis)==1: return vstats(run,varis[0],model)
    means=[sum(vreps(run,v,model))/3 for v in varis]; return (sum(means)/len(means),min(means),max(means))
def tm_metric(m):
    v=[r[m]*100 for r in TM["raw_per_repeat"]]; return (sum(v)/len(v),min(v),max(v))
def tm_ex(base):
    for k,l in TM["per_example_composite"].items():
        if k.split("#")[0]==base:
            x=l[:3]; return (sum(x)/len(x),min(x),max(x))
def b4_m4(v):
    x=[r["m4"]*100 for r in B4[v]]; return (sum(x)/len(x),min(x),max(x))

plt.rcParams.update({"font.size":10.5,"font.family":"DejaVu Sans","axes.grid":False,
    "axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#666","axes.linewidth":1.0,
    "xtick.color":"#333","ytick.color":"#333","axes.labelcolor":"#444"})
def newfig(W,H): fig,ax=plt.subplots(figsize=(W,H)); return fig,ax
def savefig(fig,name):
    try: fig.tight_layout(rect=(0,0.02,1,1))
    except Exception: pass
    p=os.path.join(ASSETS,name); fig.savefig(p,dpi=180,facecolor="white"); plt.close(fig); return p
def caption(fig,text): fig.text(0.995,0.008,text,ha="right",va="bottom",fontsize=7.3,color="#8a8a8a",style="italic")
def yerr(means,mins,maxs): return [[m-mn for m,mn in zip(means,mins)],[mx-m for m,mx in zip(means,maxs)]]
def vlabels(ax,means,tops,dy,size=9,fmt="%.1f"):
    for i,(m,t) in enumerate(zip(means,tops)): ax.annotate(fmt%m,(i,t+dy),ha="center",fontsize=size,color="#222",fontweight="bold")
def bars_mm(ax,labels,means,mins,maxs,colors,width=0.66,ymax=None,dy=1.4):
    x=np.arange(len(labels))
    b=ax.bar(x,means,width=width,color=colors,edgecolor="white",linewidth=0.7,
             yerr=yerr(means,mins,maxs),capsize=4,error_kw=dict(elinewidth=1.1,ecolor="#333"))
    ax.set_xticks(x); ax.set_xticklabels(labels)
    vlabels(ax,means,maxs,dy)
    if ymax: ax.set_ylim(0,ymax)
    return b

# ── plots ────────────────────────────────────────────────────────────────────────────────────────
def plot_tm(W,H):
    fig,ax=newfig(W,H); mets=["m1","m2","m3","m4","m7","m8"]; names=["M1\nI/O","M2\nSignal","M3\nStruct","M4\nCoverage","M7\nDLL","M8\nFunc"]
    st=[tm_metric(m) for m in mets]; means=[s[0] for s in st]; mins=[s[1] for s in st]; maxs=[s[2] for s in st]
    cols=[mNAVY,mGREEN,mOR,mBLUE,mAMB,"#8E44AD"]
    bars_mm(ax,names,means,mins,maxs,cols,ymax=108,dy=1.6)
    cm=TM["report"]["composite"]["mean"]; ax.axhline(cm,color=mRED,ls="--",lw=1.5)
    ax.text(5.45,cm+2,f"Composite {cm:.1f}%",color=mRED,fontsize=10,fontweight="bold",ha="right")
    ax.set_ylabel("score (%)"); ax.set_title("true-MIDAS (run.py prompts) - per metric",fontsize=11,color="#222")
    caption(fig,"bars = mean of 3 reps  |  whiskers = min-max across reps")
    return savefig(fig,"e_tm.png")
def plot_base(W,H):
    fig,ax=newfig(W,H)
    specs=[("#3 mini\nfinal","3","azure/gpt-4.1-mini",V5,mNAVY),("#1 mini\ninterm","1","azure/gpt-4.1-mini",V5,mBLUE),
           ("#4 g4.1\nfinal","4","azure/gpt-4.1",V5,mNAVY),("#2 g4.1\ninterm","2","azure/gpt-4.1",V5,mBLUE),
           ("#5 mini\nnocot-fin","5","azure/gpt-4.1-mini",["V1"],mGREEN),("#6 mini\nnocot-int","6","azure/gpt-4.1-mini",["V1"],"#5DADE2"),
           ("#5 g4.1\nnocot-fin","5","azure/gpt-4.1",["V1"],mGREEN),("#6 g4.1\nnocot-int","6","azure/gpt-4.1",["V1"],"#5DADE2")]
    labels=[s[0] for s in specs]; st=[runstats(s[1],s[2],s[3]) for s in specs]
    means=[x[0] for x in st]; mins=[x[1] for x in st]; maxs=[x[2] for x in st]; cols=[s[4] for s in specs]
    bars_mm(ax,labels,means,mins,maxs,cols,width=0.7,ymax=82,dy=1.0)
    cm=TM["report"]["composite"]["mean"]; ax.axhline(cm,color=mRED,ls="--",lw=1.3)
    ax.text(7.4,cm+0.8,"run.py 51.5",color=mRED,fontsize=8.5,ha="right")
    ax.set_ylabel("composite (%)"); ax.set_title("dspy inference baselines (no GEPA) - run-level mean",fontsize=11,color="#222")
    caption(fig,"bars = run-level mean  |  whiskers = min-max across V1-V5 (CoT) or 3 reps (no-CoT)")
    return savefig(fig,"e_base.png")
def plot_gepa(W,H):
    fig,ax=newfig(W,H); runs=["150-call\nCoT final","150-call\nCoT interm","300-call\nCoT final","300-call\nNoCoT final"]
    Bs=[51.42,60.35,61.30,58.93]; Os=[70.59,67.60,71.87,65.73]; x=np.arange(len(runs)); w=0.38
    b1=ax.bar(x-w/2,Bs,w,label="baseline (seed)",color=mGRAY,edgecolor="white",linewidth=0.6)
    b2=ax.bar(x+w/2,Os,w,label="GEPA optimized",color=mGREEN,edgecolor="white",linewidth=0.6)
    for i in range(4):
        ax.annotate(f"{Bs[i]:.1f}",(x[i]-w/2,Bs[i]+1),ha="center",fontsize=8.5,color="#222")
        ax.annotate(f"{Os[i]:.1f}",(x[i]+w/2,Os[i]+1),ha="center",fontsize=8.5,color="#222",fontweight="bold")
        ax.annotate(f"+{Os[i]-Bs[i]:.1f}",(x[i]+w/2,Os[i]+5),ha="center",fontsize=8.5,color=mRED,fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(runs); ax.set_ylim(0,90)
    ax.set_ylabel("composite (%)"); ax.legend(fontsize=9,loc="upper right",frameon=False)
    ax.set_title("GEPA optimized runs (gpt-4.1-mini)",fontsize=11,color="#222")
    caption(fig,"single seed / optimized eval per run (not repeated)")
    return savefig(fig,"e_gepa.png")
def plot_combo(run,model,gp,nocot,fname,title,with_runpy,W,H):
    fig,ax=newfig(W,H); labels=[]; means=[]; mins=[]; maxs=[]; cols=[]
    if with_runpy:
        cm=TM["report"]["composite"]; labels.append("run.py\nbaseline"); means.append(cm["mean"])
        mn=min(r["composite"] for r in TM["raw_per_repeat"]); mx=max(r["composite"] for r in TM["raw_per_repeat"])
        mins.append(mn); maxs.append(mx); cols.append(mRED)
    for v in V5:
        s=vstats(run,v,model); labels.append(v); means.append(s[0]); mins.append(s[1]); maxs.append(s[2]); cols.append(mBLUE)
    s=vstats("5" if "final" in fname else "6",nocot[1],model)
    labels.append("V1\nno-CoT"); means.append(s[0]); mins.append(s[1]); maxs.append(s[2]); cols.append(mGREEN)
    labels+=["GEPA\nbase","GEPA\nopt"]; means+=[gp[0],gp[1]]; mins+=[gp[0],gp[1]]; maxs+=[gp[0],gp[1]]; cols+=[mGRAY,mNAVY]
    bars_mm(ax,labels,means,mins,maxs,cols,width=0.72,ymax=90,dy=1.2)
    ax.set_ylabel("composite (%)"); ax.set_title(title,fontsize=11,color="#222")
    caption(fig,"V1-V5 & no-CoT: mean of 3 reps + min-max whiskers  |  run.py: 3 reps  |  GEPA: single eval")
    return savefig(fig,fname)
def plot_variety(run,model,nocot_run,fname,title,W,H):
    fig,ax=newfig(W,H); labels=list(V5)+["V1\nno-CoT"]
    st=[vstats(run,v,model) for v in V5]+[vstats(nocot_run,"V1",model)]
    means=[x[0] for x in st]; mins=[x[1] for x in st]; maxs=[x[2] for x in st]; cols=[mBLUE]*5+[mGREEN]
    bars_mm(ax,labels,means,mins,maxs,cols,width=0.66,ymax=85,dy=1.1)
    ax.set_ylabel("composite (%)"); ax.set_title(title,fontsize=11,color="#222")
    caption(fig,"bars = mean of 3 reps  |  whiskers = min-max across reps")
    return savefig(fig,fname)
def plot_m4kb(W,H):
    fig,ax=newfig(W,H); labels=["run.py\nbaseline","dspy\nno-inject","inject\nNEW KB (V4)","inject NEW\n+retry (V5)"]
    st=[tm_metric("m4"),b4_m4("V2"),b4_m4("V4"),b4_m4("V5")]
    means=[x[0] for x in st]; mins=[x[1] for x in st]; maxs=[x[2] for x in st]; cols=[mRED,mGRAY,mGREEN,mNAVY]
    bars_mm(ax,labels,means,mins,maxs,cols,width=0.62,ymax=95,dy=1.4)
    ax.set_ylabel("M4 coverage (%)"); ax.set_title("M4 rises with KB guideline injection (gpt-4.1, final-KB)",fontsize=10.5,color="#222")
    caption(fig,"bars = mean of 3 reps  |  whiskers = min-max across reps")
    return savefig(fig,"e_m4kb.png")
def plot_perexample(base,W,H):
    fig,ax=newfig(W,H); bm,bmn,bmx=tm_ex(base)
    a=AFTER.get(base) if AFTER else None
    if a is None: raise RuntimeError("no after data for "+base)
    am,amn,amx=(a["mean"],a.get("min",a["mean"]),a.get("max",a["mean"])) if isinstance(a,dict) else (a,a,a)
    labels=["run.py\nbaseline\n(before)","best GEPA\noptimized\n(after)"]; means=[bm,am]; mins=[bmn,amn]; maxs=[bmx,amx]
    b=ax.bar(np.arange(2),means,width=0.5,color=[mRED,mNAVY],edgecolor="white",linewidth=0.7,
             yerr=yerr(means,mins,maxs),capsize=5,error_kw=dict(elinewidth=1.1,ecolor="#333"))
    ax.set_xticks([0,1]); ax.set_xticklabels(labels); ax.set_ylim(0,max(maxs)*1.32+6)
    vlabels(ax,means,maxs,max(maxs)*0.04+0.5); ax.set_ylabel("composite (%)")
    ax.annotate(f"+{am-bm:.1f} pp",(1,am+max(maxs)*0.12+1),ha="center",color=mRED,fontsize=10,fontweight="bold")
    ax.set_title("per-example composite",fontsize=10.5,color="#222")
    cap = "before = run.py 3 reps (min-max)  |  after = best GEPA eval" if AFTER else ""
    caption(fig,cap)
    return savefig(fig,f"e_ex_{base}.png")

# ── swap into deck ──────────────────────────────────────────────────────────────────────────────
def title_of(sl):
    for sh in sl.shapes:
        if sh.name=="TextBox 6" and sh.has_text_frame: return sh.text_frame.text
    for sh in sl.shapes:
        if sh.has_text_frame and sh.text_frame.text.strip(): return sh.text_frame.text
    return ""
def find_plot(sl):
    best=None
    for sh in sl.shapes:
        if sh.shape_type==13 and Emu(sh.width).inches>2.5: best=sh
    return best
def swap(sl,maker):
    pic=find_plot(sl)
    if pic is None: return "no-plot"
    L,T,Wd,Hd=pic.left,pic.top,pic.width,pic.height; Wi,Hi=Emu(Wd).inches,Emu(Hd).inches
    try: png=maker(Wi,Hi)
    except Exception as e: return f"skip({e})"
    sp=pic._element; sp.getparent().remove(sp); sl.shapes.add_picture(png,L,T,Wd,Hd)
    return "OK"

pr=Presentation(DECK); done=[]
for i,sl in enumerate(pr.slides,1):
    t=title_of(sl); r=None
    if "True midas" in t: r=swap(sl,plot_tm)
    elif "Further Baselines" in t: r=swap(sl,plot_base)
    elif "All GEPA Optimized" in t: r=swap(sl,plot_gepa)
    elif "Final signal desired" in t: r=swap(sl,lambda W,H:plot_combo("3","azure/gpt-4.1-mini",(51.42,71.87),("5","V1"),"e_cfin.png","Final-KB . gpt-4.1-mini . run.py vs varieties vs GEPA",True,W,H))
    elif "Intermediate signal desired" in t: r=swap(sl,lambda W,H:plot_combo("1","azure/gpt-4.1-mini",(60.35,67.60),("6","V1"),"e_cint.png","Intermediate-KB . gpt-4.1-mini . varieties vs GEPA",False,W,H))
    elif "GPT-4.1 Inference" in t and "Final output" in t: r=swap(sl,lambda W,H:plot_variety("4","azure/gpt-4.1","5","e_g4f.png","gpt-4.1 . Final-KB . V1-V5 + no-CoT",W,H))
    elif "GPT-4.1 Inference" in t and "Intermediate output" in t: r=swap(sl,lambda W,H:plot_variety("2","azure/gpt-4.1","6","e_g4i.png","gpt-4.1 . Intermediate-KB . V1-V5 + no-CoT",W,H))
    elif "Old vs New Reference KB" in t: r=swap(sl,plot_m4kb)
    elif "Before vs After: Val example" in t:
        m=re.search(r"Val example (\d)",t)
        if m:
            base=VAL[int(m.group(1))-1]; r=swap(sl,lambda W,H,b=base:plot_perexample(b,W,H))
    if r: done.append(f"slide {i:2}: {r:10} | {t[:44]}")
print("\n".join(done))

backup=DECK.replace(".pptx",".ORIGINAL_backup.pptx")
try:
    if not os.path.exists(backup): shutil.copyfile(DECK,backup)
    pr.save(DECK); print(f"\nSAVED in place (original backup: {os.path.basename(backup)})")
except PermissionError:
    alt=DECK.replace(".pptx","_plotsfixed.pptx"); pr.save(alt); print(f"\nFILE LOCKED (open in PowerPoint?) -> saved to {os.path.basename(alt)}")

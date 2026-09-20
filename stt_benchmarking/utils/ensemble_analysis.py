"""
Per-stage figures for the hybrid ensemble ASR pipeline (thesis).

Every stage is rendered as a matplotlib figure and written to FIG_DIR as
vector PDF (for LaTeX) + PNG (for VS Code markdown preview). Arabic text is
shaped with arabic_reshaper and reordered with python-bidi before drawing,
so tables render correctly instead of as disconnected LTR glyphs.

Requirements: pip install arabic_reshaper python-bidi fonttools
Font: pass font_path=... to render_stages(), or let set_font() auto-pick the
best Arabic-capable font present (Amiri > Noto Naskh > Scheherazade > DejaVu).

Stage order:
  0  reference vs predictions   token table, per-token match colouring
  1  filtration                 agreement lollipop + similarity matrix
  2  reference selection        anchor-validation table
  3  alignment                  token grid coloured KEEP/REPLACE/DELETE/INSERT
  4  voting                     per-position vote table, ties highlighted
  5  final comparison           transcript panel + WER/CER bars
"""
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties, findSystemFonts
from difflib import SequenceMatcher

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    _SHAPE=True
except ImportError:
    _SHAPE=False
    warnings.warn("arabic_reshaper/python-bidi missing: Arabic will render unshaped. pip install arabic_reshaper python-bidi")

# ---- config -------------------------------------------------------
FIG_DIR="figures"
FORMATS=("pdf","png")
DPI=300
FS=9.0
PAD=7.0
_FP=None

C_KEEP="#ffffff";C_REP="#d9d9d9";C_DEL="#a6a6a6";C_INS="#ededed";C_NON="#f7f7f7"
C_ROW="#dde5f0";C_HDR="#f7f7f7";C_TIE="#f7d9d7";C_W="#ffffff"
EMPTY="X"
BLUE="#3b6fb0";GREEN="#3f8b57";RED="#b4413e";GREY="#8a8a8a";INK="#222222";LINE="#b8b8b8"

_PREFER=["amiri","scheherazade","notonaskharabic","kacst","tahoma","arial","dejavusans","freeserif"]
_PRES=[c for c in range(0xFE70,0xFF00) if c not in(0xFE75,0xFEFD,0xFEFE)]
_LIG=list(range(0xFEF5,0xFEFD))
_RESHAPER=None
_FONT_PATH=None
_UI_PATH=None
_UIFP=None
_MISSING=set()
_SYM=[0x2713,0x2717,0x2605,0x2205,0x0394,0x03C4,0x2192]

def _cmap(path):
    try:
        from fontTools.ttLib import TTFont
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f=TTFont(path,fontNumber=0,lazy=True);cm=set(f.getBestCmap());f.close()
        return cm
    except Exception:
        return None

def arabic_capable_fonts(min_coverage=0.98):
    """List system fonts that can draw pre-shaped Arabic, best first.

    Coverage is measured over the Arabic Presentation Forms-B block, because
    that is what arabic_reshaper emits and what matplotlib actually draws.
    Fonts that shape only via OpenType GSUB (e.g. most Noto Sans Arabic builds)
    score 0 here and are unusable with matplotlib, however good they look
    elsewhere.
    """
    out=[]
    for path in findSystemFonts():
        cm=_cmap(path)
        if not cm:continue
        cov=sum(c in cm for c in _PRES)/len(_PRES)
        if cov<min_coverage:continue
        lig=sum(c in cm for c in _LIG)/len(_LIG)
        b=os.path.basename(path).lower().replace("-","").replace("_","")
        rank=next((i for i,n in enumerate(_PREFER) if n in b),len(_PREFER))
        out.append({"path":path,"presentation_coverage":cov,"ligature_coverage":lig,
                    "rank":rank,"bold":"bold" in b})
    out.sort(key=lambda d:(d["rank"],-d["ligature_coverage"],-d["presentation_coverage"],d["bold"],len(d["path"])))
    return out

def download_amiri(dest="fonts",version="1.003"):
    """Fetch the Amiri typeface (OFL) and return the path to Amiri-Regular.ttf."""
    import io,zipfile,urllib.request
    os.makedirs(dest,exist_ok=True)
    target=os.path.join(dest,"Amiri-Regular.ttf")
    if os.path.exists(target):return target
    url=f"https://github.com/aliftype/amiri/releases/download/{version}/Amiri-{version}.zip"
    with urllib.request.urlopen(url) as r:
        z=zipfile.ZipFile(io.BytesIO(r.read()))
        for n in z.namelist():
            if n.endswith("Amiri-Regular.ttf"):
                open(target,"wb").write(z.read(n));break
    return target

def set_font(font_path=None,verbose=True):
    """Resolve the drawing font.

    Auto-selection requires Arabic Presentation Forms-B coverage; a font that
    passes a naive base-Arabic check but lacks that block renders every shaped
    letter as a tofu box.
    """
    global _FP,_RESHAPER,_FONT_PATH,_MISSING
    _MISSING=set()
    cands=arabic_capable_fonts()
    if font_path:
        cm=_cmap(font_path)
        if cm is not None:
            cov=sum(c in cm for c in _PRES)/len(_PRES)
            if cov<0.9 and verbose:
                warnings.warn("%s covers only %.0f%% of Arabic presentation forms; "
                              "text will show tofu boxes. Use download_amiri() or pick from "
                              "arabic_capable_fonts()."%(os.path.basename(font_path),100*cov))
        _FONT_PATH=font_path
    elif cands:
        _FONT_PATH=cands[0]["path"]
    else:
        _FONT_PATH=None
        if verbose:
            warnings.warn("No system font covers Arabic presentation forms. "
                          "Run download_amiri() and pass font_path=... .")
    _FP=FontProperties(fname=_FONT_PATH) if _FONT_PATH else FontProperties(family="DejaVu Sans")
    _resolve_ui()
    if _SHAPE:
        cm=_cmap(_FONT_PATH) if _FONT_PATH else None
        lig=bool(cm) and all(c in cm for c in _LIG)
        _RESHAPER=arabic_reshaper.ArabicReshaper(configuration={
            "delete_harakat":False,"support_ligatures":lig,"use_unshaped_instead_of_isolated":False})
    if verbose and _FONT_PATH:print("font: %s"%_FONT_PATH)
    return _FP

def _resolve_ui():
    """Arabic text fonts (Amiri, Scheherazade) usually lack ✓ ✗ ★ ∅ Δ, so Latin
    and symbol strings are drawn with a separate UI font that covers them."""
    global _UI_PATH,_UIFP
    best,score=None,-1
    for path in findSystemFonts():
        b=os.path.basename(path).lower()
        if not any(k in b for k in("dejavusans","freesans","liberationsans","arial","helvetica")):continue
        if "bold" in b or "oblique" in b or "italic" in b or "mono" in b:continue
        cm=_cmap(path)
        if not cm:continue
        sc=sum(c in cm for c in _SYM)
        if sc>score:best,score=path,sc
    _UI_PATH=best
    _UIFP=FontProperties(fname=best) if best else FontProperties(family="DejaVu Sans")
    return _UIFP

def _has_ar(s):
    return any("\u0600"<=c<="\u06ff" or "\u0750"<=c<="\u077f" or "\ufb50"<=c<="\ufeff" for c in str(s))

def _fp(s=None):
    """Font for a given already-shaped string: Arabic font if it holds Arabic, else UI font."""
    if _FP is None:set_font()
    if s is None:return _UIFP if _UIFP is not None else _FP
    return _FP if _has_ar(s) else(_UIFP if _UIFP is not None else _FP)

def font_coverage_report(strings,font_path=None):
    """Characters in `strings` (after shaping) that the font cannot draw."""
    path=font_path or _FONT_PATH
    cm=_cmap(path) if path else None
    if cm is None:return[]
    return sorted({c for s in strings for c in _ar(s) if ord(c) not in cm})

def set_output(fig_dir=FIG_DIR,formats=FORMATS,dpi=DPI):
    global FIG_DIR,FORMATS,DPI
    FIG_DIR=fig_dir;FORMATS=tuple(formats);DPI=dpi
    plt.rcParams["pdf.fonttype"]=42
    plt.rcParams["ps.fonttype"]=42
    os.makedirs(FIG_DIR,exist_ok=True)

# ---- text helpers -------------------------------------------------
def _ar(s):
    """Shape + bidi-reorder for drawing. Passes Latin/digits through unchanged."""
    if s is None:return EMPTY
    s=str(s)
    if not _SHAPE or not any("\u0600"<=c<="\u06ff" or "\u0750"<=c<="\u077f" for c in s):return s
    out=get_display((_RESHAPER.reshape(s) if _RESHAPER is not None else arabic_reshaper.reshape(s)))
    cm=_cmap(_FONT_PATH) if _FONT_PATH else None
    if cm is not None:_MISSING.update(c for c in out if ord(c) not in cm)
    return out

def _w(s,fs=FS,bold=False):
    """Width of a drawn string, in points."""
    if not s:return 0.0
    fp=_fp(s).copy();fp.set_size(fs)
    if bold:fp.set_weight("bold")
    try:return TextPath((0,0),s,size=fs,prop=fp).get_extents().width
    except Exception:return 0.62*fs*len(s)

def _wrap(s,max_pts,fs=FS):
    """Wrap on logical word order, then shape each resulting line."""
    if s is None:return[EMPTY]
    words=str(s).split()
    if not words:return[""]
    lines,cur=[],[]
    for wd in words:
        if cur and _w(_ar(" ".join(cur+[wd])),fs)>max_pts:
            lines.append(" ".join(cur));cur=[wd]
        else:cur.append(wd)
    if cur:lines.append(" ".join(cur))
    return[_ar(l) for l in lines]

def _save(fig,name):
    os.makedirs(FIG_DIR,exist_ok=True)
    out=[]
    for ext in FORMATS:
        p=os.path.join(FIG_DIR,f"{name}.{ext}")
        fig.savefig(p,dpi=DPI,bbox_inches="tight",pad_inches=0.05,facecolor="white")
        out.append(p)
    plt.close(fig)
    print("saved: "+" | ".join(out))
    return out

def _rc():
    plt.rcParams.update({"figure.dpi":110,"savefig.dpi":DPI,"axes.spines.top":False,
        "axes.spines.right":False,"axes.grid":True,"grid.alpha":0.22,"grid.linewidth":0.6,
        "font.size":FS,"axes.titlesize":FS+2,"axes.titleweight":"600","axes.labelsize":FS+0.5,})

# ---- generic table renderer ---------------------------------------
def _table(cells,row_labels,col_labels,colors=None,align=None,title=None,note=None,
           fs=FS,col_chunk=None,bold_rows=(),extra_bottom=0.0):
    """
    cells       list of rows of already-shaped strings. A cell may contain
                "\\n" to show a second, smaller, grey line underneath the
                main value (used to show "replaced X" beneath a REPLACE
                token, so the table states what changed without a lookup).
    colors      same shape, hex fill per cell (None -> white)
    align       per-column 'c'|'l'|'r'
    col_chunk   split columns into blocks of N, stacked vertically (wide grids)
    """
    nr=len(cells);nc=len(cells[0]) if nr else 0
    colors=colors or[[None]*nc for _ in range(nr)]
    align=align or["c"]*nc
    blocks=[list(range(nc))] if not col_chunk else[list(range(i,min(i+col_chunk,nc))) for i in range(0,nc,col_chunk)]
    split=[[str(cells[i][j]).split("\n") for j in range(nc)] for i in range(nr)]
    nlines=[max(len(split[i][j]) for j in range(nc)) if nc else 1 for i in range(nr)]
    lab_w=max([_w(l,fs,True) for l in row_labels]+[0])+2*PAD
    widths=[max([_w(ln,fs) for i in range(nr) for ln in split[i][j]]+[_w(col_labels[j],fs,True)])+2*PAD
            for j in range(nc)]
    unit=fs*2.05;head=unit*1.15;gap=unit*0.9;sub_fs=fs-1.3
    row_h=[unit+(nlines[i]-1)*sub_fs*1.55 for i in range(nr)]
    total_w=max(lab_w+sum(widths[j] for j in b) for b in blocks)
    tbl_h=len(blocks)*(head+sum(row_h))+(len(blocks)-1)*gap
    nlines_note=(str(note).count("\n")+1) if note else 0
    top=(fs*2.4 if title else 0)+(fs*2.0*nlines_note if note else 0)
    H=top+tbl_h+4+extra_bottom
    fig=plt.figure(figsize=(total_w/72.0,H/72.0))
    ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,total_w);ax.set_ylim(0,H);ax.axis("off")
    y=H-2
    if title:
        ax.text(0,y,title,fontsize=fs+2.2,fontweight="bold",fontproperties=_fp(title),va="top",ha="left",color=INK)
        y-=fs*2.4
    if note:
        ax.text(0,y,note,fontsize=fs-0.6,fontproperties=_fp(note),va="top",ha="left",color="#555")
        y-=fs*2.0*nlines_note
    for b in blocks:
        ax.add_patch(Rectangle((0,y-head),lab_w,head,facecolor=C_HDR,edgecolor=LINE,lw=0.6))
        x=lab_w
        for j in b:
            ax.add_patch(Rectangle((x,y-head),widths[j],head,facecolor=C_HDR,edgecolor=LINE,lw=0.6))
            ax.text(x+widths[j]/2,y-head/2,col_labels[j],fontsize=fs,fontweight="bold",
                    fontproperties=_fp(col_labels[j]),ha="center",va="center",color=INK)
            x+=widths[j]
        yy=y-head
        for i in range(nr):
            bold=row_labels[i] in bold_rows
            rh=row_h[i]
            ax.add_patch(Rectangle((0,yy-rh),lab_w,rh,facecolor=colors[i][b[0]] or C_W,edgecolor=LINE,lw=0.6))
            ax.text(PAD,yy-rh/2,row_labels[i],fontsize=fs,fontproperties=_fp(row_labels[i]),ha="left",va="center",
                    color=INK,fontweight="bold" if bold else"normal")
            x=lab_w
            for j in b:
                ax.add_patch(Rectangle((x,yy-rh),widths[j],rh,facecolor=colors[i][j] or C_W,edgecolor=LINE,lw=0.6))
                a=align[j]
                tx=x+widths[j]/2 if a=="c" else(x+PAD if a=="l" else x+widths[j]-PAD)
                ha="center" if a=="c" else("left" if a=="l" else"right")
                lines=split[i][j]
                if len(lines)==1:
                    ax.text(tx,yy-rh/2,lines[0],fontsize=fs,fontproperties=_fp(lines[0]),ha=ha,va="center",
                            color=INK,fontweight="bold" if bold else"normal")
                else:
                    ty=yy-rh/2+(len(lines)-1)*sub_fs*0.78
                    ax.text(tx,ty,lines[0],fontsize=fs,fontproperties=_fp(lines[0]),ha=ha,va="center",
                            color=INK,fontweight="bold" if bold else"normal")
                    for k,ln in enumerate(lines[1:],1):
                        ty-=sub_fs*1.55
                        ax.text(tx,ty,ln,fontsize=sub_fs,fontproperties=_fp(ln),ha=ha,va="center",color="#888")
                x+=widths[j]
            yy-=rh
        y=yy-gap
    return fig,ax

def _legend(ax,items,x=0.0,y=4.0,fs=FS-0.8):
    for lab,col in items:
        ax.add_patch(Rectangle((x,y),fs*1.5,fs*1.0,facecolor=col,edgecolor=LINE,lw=0.6))
        ax.text(x+fs*1.9,y+fs*0.5,lab,fontsize=fs,fontproperties=_fp(lab),va="center",ha="left",color=INK)
        x+=fs*1.9+_w(lab,fs)+fs*1.6
    return x

# ---- metric accessors ---------------------------------------------
def _none(t):return None if t in(None,"null","None","Null") else t
def _wer(s,p):
    m=s.get(p,{}).get("metrics");return None if not m else m["word_error_rate"]["wer (%)"]
def _cer(s,p):
    m=s.get(p,{}).get("metrics");return None if not m else m["character_error_rate"]["cer (%)"]
def _f(v,d=1):
    return"" if v is None or(isinstance(v,float) and np.isnan(v)) else f"{v:.{d}f}"
def _ratio(a,b):
    a,b=(a or "").split(),(b or "").split()
    if not a and not b:return 100.0
    if not a or not b:return 0.0
    return SequenceMatcher(None,a,b,autojunk=False).ratio()*100.0
def _sortkey(md,aid):
    w=_wer(md,aid);return 1e9 if w is None else w

# ==== stage 0 =======================================================
def stage0_reference(samples,audio_id,model_dicts,display_names,ensemble_name="Ensemble",
                     max_tokens=14,name="stage0_reference"):
    ref=samples[audio_id]["normalized_transcription"] or ""
    order=sorted(model_dicts,key=lambda m:_sortkey(model_dicts[m],audio_id))
    seq=[("Reference (human)",ref,None,None)]
    for m in order:
        seq.append((display_names[m],model_dicts[m][audio_id]["normalized_prediction"] or "",
                    _wer(model_dicts[m],audio_id),_cer(model_dicts[m],audio_id)))
    seq.append((ensemble_name,samples[audio_id]["normalized_prediction"] or "",
                _wer(samples,audio_id),_cer(samples,audio_id)))
    ncol=max(len(t.split()) for _,t,_,_ in seq)
    full=ncol
    if max_tokens:ncol=min(ncol,max_tokens)
    ref_tok=ref.split()
    cells,colors,labels=[],[],[]
    for lab,txt,w,c in seq:
        tok=txt.split()
        if lab=="Reference (human)":
            row=[_ar(tok[j]) if j<len(tok) else"" for j in range(ncol)]+[_f(w),_f(c)]
            col=[C_ROW]*(ncol+2)
        else:
            # Every system, including the ensemble, is graded the same way:
            # match-shading only where its token equals the human reference at
            # that position, differ-shading otherwise with the reference word
            # shown beneath. No row gets a blanket "correct" fill just for
            # being the ensemble.
            row,col=[],[]
            for j in range(ncol):
                if j>=len(tok):
                    row.append("");col.append(None)
                elif j<len(ref_tok) and tok[j]==ref_tok[j]:
                    row.append(_ar(tok[j]));col.append(C_KEEP)
                elif j<len(ref_tok):
                    row.append(_ar(tok[j]));col.append(C_REP)
                else:
                    row.append(_ar(tok[j]));col.append(C_INS)
            row+=[_f(w),_f(c)];col+=[None,None]
        cells.append(row);colors.append(col);labels.append(_ar(lab))
    cols=[str(j+1) for j in range(ncol)]+["WER","CER"]
    note=("%s%s   |   shading: matches reference / differs / extra word not in reference"
          "   |   systems sorted by WER")%(
        os.path.basename(audio_id)," (first %d of %d tokens)"%(ncol,full) if ncol<full else"")
    fig,_=_table(cells,labels,cols,colors,["c"]*ncol+["r","r"],
                 title="Stage 0 - human reference vs individual systems",note=note,
                 bold_rows=(_ar("Reference (human)"),_ar(ensemble_name)))
    return _save(fig,name)

# ==== stage 1 =======================================================
def stage1_filtration(samples,audio_id,model_dicts,voter_order,display_names,name="stage1_filtration"):
    _rc()
    md=samples[audio_id]["metadata"]["records_filtration"]
    labels=[_ar(display_names[v]) for v in voter_order]
    preds=[(model_dicts[v][audio_id]["normalized_prediction"] or "") for v in voter_order]
    n=len(voter_order)
    kept=set(md.get("kept_indices",list(range(n))))
    scores=md.get("agreement_scores");thr=md.get("dynamic_threshold")
    fig,axes=plt.subplots(1,2,figsize=(12.4,0.52*n+2.8),gridspec_kw={"width_ratios":[1,1.2]})
    ax=axes[0]
    if scores is None:
        ax.text(.5,.5,"filtration skipped\n%s"%md.get("reason",""),ha="center",va="center",
                transform=ax.transAxes,color=GREY,fontproperties=_fp(None))
        ax.axis("off")
    else:
        s=np.array(scores,dtype=float);o=np.argsort(s);y=np.arange(n)
        cols=[GREEN if o[i] in kept else RED for i in range(n)]
        ax.hlines(y,s.min()-4,s[o],color="#d6d6d6",lw=1.1,zorder=1)
        ax.scatter(s[o],y,c=cols,s=64,zorder=3,edgecolor="white",lw=.9)
        ax.set_yticks(y)
        ax.set_yticklabels([labels[o[i]] for i in range(n)],fontsize=FS-0.5)
        for i,t in enumerate(ax.get_yticklabels()):
            t.set_fontproperties(_fp(labels[o[i]]));t.set_color(GREEN if o[i] in kept else RED)
        for i in range(n):
            ax.text(s[o][i]+0.8,i,f"{s[o][i]:.1f}",va="center",fontsize=FS-2,color="#555")
        if thr is not None:
            ax.axvline(thr,color=BLUE,ls="--",lw=1.3)
            ax.axvspan(ax.get_xlim()[0],thr,color=RED,alpha=.05)
            ax.text(0.98,0.045,"threshold = %.1f"%thr,transform=ax.transAxes,color=BLUE,
                    fontsize=FS-0.5,ha="right",va="bottom",
                    bbox=dict(facecolor="white",edgecolor="none",alpha=.75,pad=1.5))
        ax.set_xlabel("mean pairwise agreement (%)")
        ax.set_title("(a) dynamic-threshold filtration   [kept above threshold, dropped below]",loc="left",pad=10)
    ax=axes[1]
    M=np.array([[_ratio(preds[i],preds[j]) if i!=j else 100. for j in range(n)] for i in range(n)])
    cmap=plt.get_cmap("viridis")
    im=ax.imshow(M,cmap=cmap,vmin=0,vmax=100)
    ax.set_xticks(range(n));ax.set_yticks(range(n))
    ax.set_xticklabels(labels,rotation=42,ha="right",fontsize=FS-2)
    ax.set_yticklabels(labels,fontsize=FS-2)
    for i,t in enumerate(ax.get_xticklabels()):t.set_fontproperties(_fp(labels[i]))
    for i,t in enumerate(ax.get_yticklabels()):
        t.set_fontproperties(_fp(labels[i]));t.set_color(GREEN if i in kept else RED)
    for i in range(n):
        for j in range(n):
            r,g,bl,_=cmap(M[i,j]/100.0)
            lum=0.299*r+0.587*g+0.114*bl   # relative luminance of the actual cell colour
            ax.text(j,i,f"{M[i,j]:.0f}",ha="center",va="center",fontsize=FS-2,
                    color=INK if lum>0.6 else"white")
        if i not in kept:
            ax.add_patch(Rectangle((-.5,i-.5),n,1,fill=False,edgecolor=RED,lw=2.1))
    ax.grid(False)
    fig.colorbar(im,ax=ax,fraction=.046,pad=.04,label="word-level similarity (%)")
    ax.set_title("(b) pairwise similarity matrix",loc="left")
    fig.suptitle("Stage 1 - agreement filtration   |   kept %s/%s"%(md.get("num_kept","?"),md.get("num_models",n)),
                 y=1.03,fontsize=FS+3,fontweight="bold",x=0.005,ha="left")
    fig.tight_layout()
    return _save(fig,name)

# ==== stage 2 =======================================================
def stage2_reference_selection(samples,audio_id,kept_names,kept_preds,name="stage2_reference_selection"):
    md=samples[audio_id]["metadata"]["reference_selection"]
    strat=md.get("strategy_metric");ridx=md.get("reference_index");score=md.get("score")
    anch={a["transcription_index"]:a for a in(md.get("anchor_metadata") or[])}
    has=bool(anch)
    cols=["words","longest match","min required","length class","valid","selected"] if has else["words","selected"]
    al=["r","r","r","c","c","c"] if has else["r","c"]
    cells,colors,labels=[],[],[]
    for k,(nm,pr) in enumerate(zip(kept_names,kept_preds)):
        sel="*" if k==ridx else""
        if has:
            d=(anch.get(k) or{}).get("details") or{}
            row=[str(len((pr or "").split())),str(d.get("additional_matches","-")),
                 str(d.get("min_required","-")),str(d.get("length_category","-")),
                 ("yes" if anch.get(k,{}).get("is_valid") else"no") if k in anch else"-",sel]
        else:
            row=[str(len((pr or "").split())),sel]
        cells.append(row);colors.append([C_ROW]*len(row) if k==ridx else[None]*len(row));labels.append(_ar(nm))
    note="strategy = %s"%strat
    if score is not None:note+="   |   score = %.3f"%score
    ls,ms=md.get("length_stats"),md.get("match_stats")
    if ls:note+="   |   length mode = %s"%ls.get("mode")
    if ms:note+="   |   match mode = %s"%ms.get("mode")
    note+="   |   * = selected reference"
    fig,_=_table(cells,labels,cols,colors,al,title="Stage 2 - reference selection",note=note)
    return _save(fig,name)

# ==== stage 3 =======================================================
def _op(rt,t):
    if rt is not None:return"DEL" if t is None else("KEEP" if t==rt else"REP")
    return"INS" if t is not None else"NON"
_OPC={"KEEP":C_KEEP,"REP":C_REP,"DEL":C_DEL,"INS":C_INS,"NON":C_NON}

def _grade_vs_truth(true_ref_words,hyp_words):
    """
    Per-token correctness of hyp_words against the real human reference,
    via SequenceMatcher opcodes. Returns (is_correct[i], true_word_or_None[i])
    aligned to hyp_words. Used to colour the Fusion row honestly: matching
    the pipeline's own structural reference is not the same claim as
    matching the human transcript, and only the latter means "correct".
    """
    sm=SequenceMatcher(None,true_ref_words,hyp_words,autojunk=False)
    ok=[False]*len(hyp_words);paired=[None]*len(hyp_words)
    for tag,i1,i2,j1,j2 in sm.get_opcodes():
        if tag=="equal":
            for j in range(j1,j2):ok[j]=True
        elif tag=="replace" and(i2-i1)==(j2-j1):
            for k in range(j2-j1):paired[j1+k]=true_ref_words[i1+k]
    return ok,paired

def _align_to_grid(padded_ref,words):
    """Lay `words` onto the padded reference grid using the same opcode mapping
    the pipeline applies to candidate hypotheses."""
    L=len(padded_ref)
    out=[None]*L
    sm=SequenceMatcher(None,padded_ref,words,autojunk=False)
    for tag,i1,i2,j1,j2 in sm.get_opcodes():
        if tag=="equal":
            for i in range(i1,i2):out[i]=words[j1+(i-i1)]
        elif tag=="replace":
            for i in range(i1,i2):
                k=j1+(i-i1)
                if k<j2:out[i]=words[k]
    return out

def stage3_alignment(samples,audio_id,kept_names,col_chunk=12,name="stage3_alignment"):
    md=samples[audio_id]["metadata"]
    ref=[_none(x) for x in md["alignment"]["adjusted_reference_tokens"]]
    cands=[[_none(x) for x in c] for c in md["voting"]["candidates_tokens"]]
    fusion=[_none(x) for x in md["voting"]["fusion_tokens"]]
    L=len(ref)
    true_words=(samples[audio_id].get("normalized_transcription") or"").split()
    true_grid=_align_to_grid(ref,true_words)
    rows=[("Reference (human)",None,[C_ROW]*L,
           [_ar(t) if t is not None else EMPTY for t in true_grid]),
          ("Padded reference",None,[C_ROW]*L,
           [_ar(t) if t is not None else EMPTY for t in ref])]    
    for nm,c in zip(kept_names,cands):
        col,txt=[],[]
        for i in range(L):
            ci=c[i] if i<len(c) else None
            op=_op(ref[i],ci)
            col.append(_OPC[op])
            if op in("KEEP","REP","INS"):txt.append(_ar(ci))
            else:txt.append(EMPTY)
        rows.append((nm,None,col,txt))
    # Fusion is graded against the human reference transcript, not the
    # padded structural reference above: matching that backbone by
    # construction is not the same as being correct.
    hyp_idx=[i for i,t in enumerate(fusion) if t is not None]
    hyp_words=[fusion[i] for i in hyp_idx]
    ok,_=_grade_vs_truth(true_words,hyp_words)
    fus_text=[EMPTY]*L;fus_color=[C_DEL]*L
    for k,i in enumerate(hyp_idx):
        fus_text[i]=_ar(fusion[i])
        fus_color[i]=C_KEEP if ok[k] else C_REP
    rows.append(("Fusion",None,fus_color,fus_text))
    cells=[list(r[3]) for r in rows]
    colors=[list(r[2]) for r in rows]
    labels=[_ar(r[0]) for r in rows]
    ins=sum(1 for t in ref if t is None)
    note=("%s = no token in this slot   |   %d padded insertion slot(s)\n"
          "cell shading gives the edit operation against the padded reference; "
          "the Fusion row is graded against the human transcript instead")%(EMPTY,ins)
    fig,ax=_table(cells,labels,[str(j) for j in range(L)],colors,title="Stage 3 - alignment to padded reference",
                  note=note,col_chunk=col_chunk,bold_rows=(_ar("Reference (human)"),_ar("Padded reference"),_ar("Fusion")),
                  extra_bottom=FS*2.2)
    _legend(ax,[("KEEP",C_KEEP),("REPLACE",C_REP),("DELETE",C_DEL),("INSERT",C_INS)],x=0,y=4)
    return _save(fig,name)

# ==== stage 4 =======================================================
def stage4_voting(samples,audio_id,name="stage4_voting"):
    v=samples[audio_id]["metadata"]["voting"]
    cells,colors,labels=[],[],[]
    nties=0
    for vd in v["voting_details"]:
        votes={(EMPTY if _none(k) is None else k):c for k,c in vd["token_votes"].items()}
        cnt=sorted(votes.values(),reverse=True)
        tie=len(cnt)>1 and cnt[0]==cnt[1]
        nties+=int(tie)
        rule="tie -> cascade" if tie else("unanimous" if len(votes)==1 else"plurality")
        dist="    ".join("%s:%d"%(_ar(k),c) for k,c in sorted(votes.items(),key=lambda kv:-kv[1]))
        win=vd["final_token"]
        cells.append([dist,_ar(win) if _none(win) is not None else EMPTY,rule])
        colors.append([C_TIE]*3 if tie else[None]*3)
        labels.append(str(vd["position"]))
    note="%d position(s)   |   %d tie(s) resolved by the edit-distance / character-overlap cascade"%(len(cells),nties)
    conf=v.get("confidence_score")
    if conf is not None:note+="   |   mean agreement on winner = %.3f"%conf
    fig,_=_table(cells,labels,["vote distribution","winner","rule"],colors,["l","c","c"],
                 title="Stage 4 - token-level plurality voting",note=note)
    return _save(fig,name)

# ==== stage 5 =======================================================
def stage5_comparison(samples,audio_id,model_dicts,display_names,ensemble_name="Ensemble",
                      wrap_pts=430,name="stage5_comparison"):
    _rc()
    fs=FS
    ref=samples[audio_id]["normalized_transcription"] or ""
    order=sorted(model_dicts,key=lambda m:_sortkey(model_dicts[m],audio_id))
    ew,ec=_wer(samples,audio_id),_cer(samples,audio_id)
    seq=[("Reference (human)",ref,None,None,C_ROW)]
    for m in order:
        seq.append((display_names[m],model_dicts[m][audio_id]["normalized_prediction"] or "",
                    _wer(model_dicts[m],audio_id),_cer(model_dicts[m],audio_id),None))
    seq.append((ensemble_name,samples[audio_id]["normalized_prediction"] or "",ew,ec,C_ROW))
    lab_w=max(_w(_ar(l),fs,True) for l,_,_,_,_ in seq)+2*PAD
    num_w=_w("WER (%)",fs,True)+2*PAD
    wrapped=[_wrap(t,wrap_pts,fs) for _,t,_,_,_ in seq]
    rh=fs*1.9;head=rh*1.2
    heights=[max(1,len(x))*rh+PAD for x in wrapped]
    total_w=lab_w+2*num_w+wrap_pts+2*PAD
    tbl_h=head+sum(heights)
    top=fs*2.4+fs*2.0
    bar_h=210.0
    H=top+tbl_h+bar_h+26
    fig=plt.figure(figsize=(total_w/72.0,H/72.0))
    ax=fig.add_axes([0,(bar_h+26)/H,1,(H-bar_h-26)/H])
    ax.set_xlim(0,total_w);ax.set_ylim(0,top+tbl_h);ax.axis("off")
    y=top+tbl_h
    singles=[w for _,_,w,_,_ in seq[1:-1] if w is not None]
    best=min(singles) if singles else None
    ax.text(0,y,"Stage 5 - final comparison",fontsize=fs+2.2,fontweight="bold",fontproperties=_fp(None),va="top",color=INK)
    y-=fs*2.4
    sub="" if(ew is None or best is None) else"ensemble WER %.1f   vs   best single system %.1f   (diff %+.1f)"%(ew,best,best-ew)
    ax.text(0,y,sub,fontsize=fs-0.6,fontproperties=_fp(sub),va="top",color="#555")
    y-=fs*2.0
    hdr=[("system",0,lab_w),("WER (%)",lab_w,num_w),("CER (%)",lab_w+num_w,num_w),
         ("transcript",lab_w+2*num_w,wrap_pts+2*PAD)]
    for lab,x0,wd in hdr:
        ax.add_patch(Rectangle((x0,y-head),wd,head,facecolor=C_HDR,edgecolor=LINE,lw=.6))
        ax.text(x0+wd/2,y-head/2,lab,fontsize=fs,fontweight="bold",fontproperties=_fp(lab),
                ha="center",va="center",color=INK)
    y-=head
    for(lab,txt,w,c,fill),lines,h in zip(seq,wrapped,heights):
        bold=fill is not None
        for _,x0,wd in hdr:
            ax.add_patch(Rectangle((x0,y-h),wd,h,facecolor=fill or C_W,edgecolor=LINE,lw=.6))
        ax.text(PAD,y-h/2,_ar(lab),fontsize=fs,fontproperties=_fp(_ar(lab)),ha="left",va="center",
                color=INK,fontweight="bold" if bold else"normal")
        ax.text(lab_w+num_w-PAD,y-h/2,_f(w),fontsize=fs,fontproperties=_fp(None),ha="right",va="center",color=INK)
        ax.text(lab_w+2*num_w-PAD,y-h/2,_f(c),fontsize=fs,fontproperties=_fp(None),ha="right",va="center",color=INK)
        ty=y-PAD/2-rh*0.72
        for ln in lines:
            ax.text(total_w-PAD,ty,ln,fontsize=fs,fontproperties=_fp(ln),ha="right",va="center",
                    color=INK,fontweight="bold" if bold else"normal")
            ty-=rh
        y-=h
    axb=fig.add_axes([0.075,0.055,0.90,bar_h/H*0.68])
    names=[_ar(display_names[m]) for m in order]+[_ar(ensemble_name)]
    wv=[_wer(model_dicts[m],audio_id) for m in order]+[ew]
    cv=[_cer(model_dicts[m],audio_id) for m in order]+[ec]
    x=np.arange(len(names));bw=0.38
    axb.bar(x-bw/2,wv,bw,color=[GREY]*len(order)+[GREEN],edgecolor="white",label="WER")
    axb.bar(x+bw/2,cv,bw,color=["#cccccc"]*len(order)+["#8fc6a1"],edgecolor="white",label="CER")
    if ew is not None:axb.axhline(ew,color=GREEN,ls="--",lw=1.1)
    axb.set_xticks(x);axb.set_xticklabels(names,rotation=38,ha="right",fontsize=fs-2)
    for i,t in enumerate(axb.get_xticklabels()):t.set_fontproperties(_fp(names[i]))
    axb.set_ylabel("error rate (%)")
    axb.legend(frameon=False,fontsize=fs-2,ncol=2,loc="upper right")
    for i,val in enumerate(wv):
        if val is not None:axb.text(i-bw/2,val,f"{val:.0f}",ha="center",va="bottom",fontsize=fs-2.5)
    axb.spines["top"].set_visible(False);axb.spines["right"].set_visible(False)
    axb.grid(axis="y",alpha=.22,lw=.6);axb.set_axisbelow(True)
    axb.set_title("per-system error rates on this utterance",loc="left",fontsize=fs+0.5)
    return _save(fig,name)

# ==== driver ========================================================
def render_stages(audio_id,samples,model_dicts,voter_order,display_names,ensemble_name="Ensemble",
                  fig_dir=FIG_DIR,formats=FORMATS,dpi=DPI,font_path=None,prefix="",
                  max_tokens=14,col_chunk=12,wrap_pts=430):
    """Render every stage and write it to fig_dir as <prefix>stageN_*.{pdf,png}."""
    set_output(fig_dir,formats,dpi);set_font(font_path)
    md=samples[audio_id]["metadata"]
    kept_idx=md["records_filtration"].get("kept_indices",list(range(len(voter_order))))
    kept_voters=[voter_order[i] for i in kept_idx]
    kept_names=[display_names[v] for v in kept_voters]
    kept_preds=[model_dicts[v][audio_id]["normalized_prediction"] for v in kept_voters]
    p=[]
    p+=stage0_reference(samples,audio_id,model_dicts,display_names,ensemble_name,max_tokens,prefix+"stage0_reference")
    p+=stage1_filtration(samples,audio_id,model_dicts,voter_order,display_names,prefix+"stage1_filtration")
    p+=stage2_reference_selection(samples,audio_id,kept_names,kept_preds,prefix+"stage2_reference_selection")
    p+=stage3_alignment(samples,audio_id,kept_names,col_chunk,prefix+"stage3_alignment")
    p+=stage4_voting(samples,audio_id,prefix+"stage4_voting")
    p+=stage5_comparison(samples,audio_id,model_dicts,display_names,ensemble_name,wrap_pts,prefix+"stage5_comparison")
    if _MISSING:
        alt=arabic_capable_fonts()
        warnings.warn("font %s cannot draw %d character(s): %s. %s"%(
            os.path.basename(_FONT_PATH or "?"),len(_MISSING),"".join(sorted(_MISSING)),
            ("Try font_path=%r."%alt[1]["path"]) if len(alt)>1 else "Run download_amiri() and pass font_path=... ."))
    return p
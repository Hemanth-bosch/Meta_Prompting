"""
Build: AIQ-212-Overview-For-MIDAS.pptx
4 slides for MIDAS lead (DCA + MBD-UTA use case owners):
  Slide 1 — AIQ-212 Vision & Agenda
  Slide 2 — MIDAS Use Case Selection
  Slide 3 — DCA: What was done, how, results, future
  Slide 4 — MBD-UTA: What was done, how, results, future

Style: matches Brief-AIQ-212-DCA_MBD-Overview_Q2-20-5-26-2.pptx
  - Bosch Blue: 007BC0
  - Dark navy: 203864
  - Green accent: 009660 / 00B050
  - Orange accent: FF6B00
  - Red: E2001A
  - Light gray bg: F2F2F2
  - White: FFFFFF
  - Dark text: 333333

Usage:
  /c/Users/BHS1RNG/.conda/envs/new2/python.exe build_overview_ppt.py
"""

import sys
sys.path.insert(0, r'C:\Users\BHS1RNG\.conda\envs\new2\Lib\site-packages')

from pptx import Presentation
from pptx.util import Emu, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu as E
import copy

# ── Output path ──────────────────────────────────────────────────────────────
DST = r"C:\Users\BHS1RNG\Downloads\AIQ-212-Overview-For-MIDAS.pptx"
SRC = r"C:\Users\BHS1RNG\Downloads\Brief-AIQ-212-DCA_MBD-Overview_Q2-20-5-26-2.pptx"

# ── Dimensions (12 × 6.75 inches widescreen) ─────────────────────────────────
W  = 10_969_625   # slide width  EMU
H  =  6_170_613   # slide height EMU

# ── Color palette ─────────────────────────────────────────────────────────────
BOSCH_BLUE   = RGBColor(0x00, 0x7B, 0xC0)
DARK_NAVY    = RGBColor(0x20, 0x38, 0x64)
DARK_BLUE2   = RGBColor(0x00, 0x45, 0x82)
GREEN        = RGBColor(0x00, 0x96, 0x60)
GREEN_LIGHT  = RGBColor(0x00, 0xB0, 0x50)
ORANGE       = RGBColor(0xFF, 0x6B, 0x00)
RED          = RGBColor(0xE2, 0x00, 0x1A)
WHITE        = RGBColor(0xFF, 0xFF, 0xFF)
DARK_GRAY    = RGBColor(0x33, 0x33, 0x33)
MID_GRAY     = RGBColor(0x59, 0x59, 0x59)
LIGHT_GRAY   = RGBColor(0xF2, 0xF2, 0xF2)
YELLOW       = RGBColor(0xFF, 0xBF, 0x00)
TEAL         = RGBColor(0x00, 0xB0, 0xA0)
BLACK        = RGBColor(0x00, 0x00, 0x00)
PURPLE       = RGBColor(0x55, 0x11, 0x51)

# ── Build from blank template using source file for theme ─────────────────────
prs = Presentation(SRC)
# Remove all existing slides (keep theme/layouts)
for _ in range(len(prs.slides)):
    rId = prs.slides._sldIdLst[0].rId
    prs.part.drop_rel(rId)
    del prs.slides._sldIdLst[0]

# ── Layout helpers ────────────────────────────────────────────────────────────
def layout(name):
    for l in prs.slide_layouts:
        if l.name.strip().lower() == name.strip().lower():
            return l
    return prs.slide_layouts[17]  # blank fallback

def add_slide(layout_name="Blank slide"):
    sl = prs.slides.add_slide(layout(layout_name))
    # Remove all placeholder shapes from layout (start clean)
    for ph in sl.placeholders:
        sp = ph._element
        sp.getparent().remove(sp)
    return sl

# ── Shape helpers ─────────────────────────────────────────────────────────────
from pptx.util import Emu
from pptx.oxml.ns import qn
from lxml import etree

def add_rect(slide, x, y, w, h, fill_color, radius=0):
    """Add a solid-filled rectangle, optionally rounded."""
    from pptx.util import Emu
    txBox = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        Emu(x), Emu(y), Emu(w), Emu(h)
    )
    txBox.fill.solid()
    txBox.fill.fore_color.rgb = fill_color
    txBox.line.color.rgb = fill_color  # no border
    if radius > 0:
        # apply rounded corners via XML
        sp = txBox._element
        prstGeom = sp.find('.//' + qn('a:prstGeom'))
        if prstGeom is not None:
            prstGeom.set('prst', 'roundRect')
            avLst = prstGeom.find(qn('a:avLst'))
            if avLst is None:
                avLst = etree.SubElement(prstGeom, qn('a:avLst'))
            avLst.clear()
            gd = etree.SubElement(avLst, qn('a:gd'))
            gd.set('name', 'adj')
            gd.set('fmla', f'val {radius}')
    return txBox

def add_textbox(slide, x, y, w, h, text, font_size_pt, bold=False,
                color=WHITE, align=PP_ALIGN.LEFT, wrap=True, italic=False):
    """Add a text box with a single paragraph."""
    txBox = slide.shapes.add_textbox(Emu(x), Emu(y), Emu(w), Emu(h))
    tf = txBox.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size_pt)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return txBox

def add_textbox_lines(slide, x, y, w, h, lines, font_size_pt, bold=False,
                      color=WHITE, align=PP_ALIGN.LEFT, line_bold=None, line_color=None,
                      line_spacing_pt=None):
    """Add multi-line text box. lines = list of strings."""
    txBox = slide.shapes.add_textbox(Emu(x), Emu(y), Emu(w), Emu(h))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.alignment = align
        if line_spacing_pt:
            from pptx.util import Pt as _Pt
            p.line_spacing = _Pt(line_spacing_pt)
        run = p.add_run()
        run.text = line
        run.font.size = Pt(font_size_pt)
        is_bold = (line_bold[i] if line_bold and i < len(line_bold) else bold)
        run.font.bold = is_bold
        col = (line_color[i] if line_color and i < len(line_color) else color)
        run.font.color.rgb = col
    return txBox

def add_label_value(slide, x, y, w, h, label, value, label_pt, value_pt,
                    label_color=BOSCH_BLUE, value_color=DARK_GRAY):
    """Small label + value pair."""
    txBox = slide.shapes.add_textbox(Emu(x), Emu(y), Emu(w), Emu(h))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r1 = p.add_run()
    r1.text = label + "  "
    r1.font.size = Pt(label_pt)
    r1.font.bold = True
    r1.font.color.rgb = label_color
    r2 = p.add_run()
    r2.text = value
    r2.font.size = Pt(value_pt)
    r2.font.bold = False
    r2.font.color.rgb = value_color
    return txBox

def header_bar(slide, title, subtitle=None, title_color=BLACK, sub_color=MID_GRAY):
    """Standard page header used on content slides."""
    # thin top accent line
    add_rect(slide, 0, 0, W, 16_000, BOSCH_BLUE)
    # title
    add_textbox(slide, 205_200, 30_000, W - 400_000, 380_000,
                title, 20, bold=True, color=title_color)
    if subtitle:
        add_textbox(slide, 205_200, 390_000, W - 400_000, 250_000,
                    subtitle, 11, bold=False, color=sub_color)
    # page bottom line
    add_rect(slide, 0, H - 80_000, W, 18_000, BOSCH_BLUE)
    # Bosch logo area (right top corner) - small colored square placeholder
    add_rect(slide, W - 1_100_000, 0, 1_100_000, 16_000, DARK_NAVY)

def card(slide, x, y, w, h, header_text, bullets, header_color=DARK_NAVY,
         bg_color=None, header_txt_color=WHITE, bullet_pt=10.5, header_pt=12):
    """Colored card with header bar + bullet list below."""
    if bg_color is None:
        bg_color = LIGHT_GRAY
    # card background
    add_rect(slide, x, y, w, h, bg_color, radius=20000)
    # header strip
    add_rect(slide, x, y, w, int(h * 0.22), header_color, radius=20000)
    # fix bottom corners of header
    add_rect(slide, x, y + int(h * 0.11), w, int(h * 0.11), header_color)
    # header text
    add_textbox(slide, x + 60_000, y + 15_000, w - 120_000, int(h * 0.20),
                header_text, header_pt, bold=True, color=header_txt_color)
    # bullets
    if bullets:
        add_textbox_lines(slide, x + 60_000, y + int(h * 0.24), w - 120_000,
                          int(h * 0.72), bullets, bullet_pt,
                          bold=False, color=DARK_GRAY)

def stat_box(slide, x, y, w, h, big_number, label, bg_color=DARK_NAVY,
             number_pt=32, label_pt=10):
    """Large number with label below — for result callouts."""
    add_rect(slide, x, y, w, h, bg_color, radius=25000)
    add_textbox(slide, x, y + int(h * 0.08), w, int(h * 0.55),
                big_number, number_pt, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_textbox(slide, x, y + int(h * 0.62), w, int(h * 0.34),
                label, label_pt, bold=False, color=WHITE, align=PP_ALIGN.CENTER)

def pill(slide, x, y, w, h, text, bg_color, text_color=WHITE, font_pt=10, bold=True):
    """Small pill/badge shape."""
    add_rect(slide, x, y, w, h, bg_color, radius=30000)
    add_textbox(slide, x, y, w, h, text, font_pt, bold=bold,
                color=text_color, align=PP_ALIGN.CENTER)

def arrow_right(slide, x, y, w=200_000, h=120_000, color=BOSCH_BLUE):
    """Simple right arrow using a shape."""
    shp = slide.shapes.add_shape(13, Emu(x), Emu(y), Emu(w), Emu(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.color.rgb = color


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — AIQ-212: Vision & Agenda
# ═══════════════════════════════════════════════════════════════════════════════
def slide_1_aiq212():
    sl = add_slide()

    # Full-width dark header band
    add_rect(sl, 0, 0, W, 1_500_000, DARK_NAVY)
    # Bosch blue accent strip inside header
    add_rect(sl, 0, 1_460_000, W, 60_000, BOSCH_BLUE)

    # Title in header
    add_textbox(sl, 300_000, 180_000, W - 600_000, 700_000,
                "AIQ-212: Automated LLM Agent Optimization",
                32, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
    add_textbox(sl, 300_000, 900_000, W - 600_000, 430_000,
                "CR / Bosch Research  ·  MIDAS Collaboration  ·  Q2 2026",
                14, bold=False, color=RGBColor(0xBB, 0xCC, 0xDD), align=PP_ALIGN.LEFT)

    # Phase badge
    add_rect(sl, 300_000, 1_620_000, 820_000, 280_000, BOSCH_BLUE, radius=20000)
    add_textbox(sl, 300_000, 1_620_000, 820_000, 280_000,
                "Phase 2", 14, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    # ── Vision box ──────────────────────────────────────────────────────────────
    add_rect(sl, 300_000, 1_980_000, W - 600_000, 520_000, LIGHT_GRAY, radius=15000)
    add_textbox(sl, 400_000, 2_010_000, 500_000, 240_000,
                "VISION", 10, bold=True, color=BOSCH_BLUE)
    add_textbox(sl, 400_000, 2_220_000, W - 800_000, 250_000,
                "Systematically optimize LLM agents for engineering tasks "
                "— without manual prompt tuning — using DSPy + reflection-based optimizers.",
                12, bold=False, color=DARK_GRAY)

    # ── Three column cards for Agenda ──────────────────────────────────────────
    col_y   = 2_650_000
    col_h   = 3_100_000
    col_w   = 3_200_000
    gap     = 136_000
    x_start = 300_000

    # Card 1: Use Cases
    card(sl, x_start, col_y, col_w, col_h,
         "Use Cases", [
             "Document Comparison Assist (DCA)",
             "  — Bosch CR × MiDAS",
             "",
             "MBD Unit Test Assist (MBD-UTA)",
             "  — Bosch CR × MiDAS",
             "",
             "RSV Agentic Design (bonus)",
             "  — Bosch CR × CMU",
         ],
         header_color=BOSCH_BLUE, header_pt=13, bullet_pt=10.5)

    # Card 2: Optimization Stack
    card(sl, x_start + col_w + gap, col_y, col_w, col_h,
         "Optimization Stack", [
             "Framework: DSPy 3.0",
             "",
             "Optimizers used:",
             "  Bootstrap  — few-shot synthesis",
             "  MIPROv2   — Bayesian search",
             "  GEPA        — reflection rewriting",
             "  Rewriter   — seed diversification",
             "",
             "Metrics: task-specific, reference-free",
         ],
         header_color=DARK_BLUE2, header_pt=13, bullet_pt=10.5)

    # Card 3: Key Outcomes
    card(sl, x_start + 2 * (col_w + gap), col_y, col_w, col_h,
         "Key Outcomes", [
             "DCA:  +5.7 pp over human prompts",
             "  86.6% → 92.3% (gpt-4.1)",
             "  Pareto front: -46% token cost",
             "",
             "MBD-UTA:  +17 pp (partial run)",
             "  73.9% → 91.0% (iter 7 / 742)",
             "  Run stopped at 27%",
             "",
             "Next: full pipeline + 3-prompt opt.",
         ],
         header_color=RGBColor(0x00, 0x70, 0x50), header_pt=13, bullet_pt=10.5)

    # bottom line
    add_rect(sl, 0, H - 80_000, W, 18_000, BOSCH_BLUE)
    add_textbox(sl, W - 500_000, H - 75_000, 450_000, 70_000,
                "1", 9, color=MID_GRAY, align=PP_ALIGN.RIGHT)
    return sl


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — MIDAS Use Case Selection
# ═══════════════════════════════════════════════════════════════════════════════
def slide_2_midas():
    sl = add_slide()
    header_bar(sl,
               "MIDAS Use Case Selection",
               "Why DCA and MBD-UTA  ·  What AIQ-212 brings to MIDAS")

    content_y = 680_000

    # ── Left: Selection criteria ────────────────────────────────────────────────
    add_rect(sl, 200_000, content_y, 3_100_000, 330_000, BOSCH_BLUE, radius=15000)
    add_textbox(sl, 260_000, content_y + 50_000, 3_000_000, 250_000,
                "Selection Criteria", 13, bold=True, color=WHITE)

    criteria = [
        ("Multi-LLM call pipeline",   "Both use multiple sequential LLM steps — rich optimization surface"),
        ("Reference-free evaluation", "No ground-truth needed — metrics derived from task structure"),
        ("Real production value",      "Both are active MIDAS tools used by engineers"),
        ("Complementary complexity",   "DCA: multimodal classification · MBD-UTA: structured equation solving"),
    ]
    for i, (label, desc) in enumerate(criteria):
        y = content_y + 380_000 + i * 500_000
        add_rect(sl, 200_000, y, 90_000, 90_000, BOSCH_BLUE, radius=30000)
        add_textbox(sl, 200_000, y, 90_000, 90_000,
                    str(i+1), 11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        add_textbox(sl, 340_000, y - 10_000, 2_850_000, 200_000,
                    label, 11, bold=True, color=DARK_NAVY)
        add_textbox(sl, 340_000, y + 175_000, 2_850_000, 230_000,
                    desc, 10, bold=False, color=DARK_GRAY)

    # ── Center divider ──────────────────────────────────────────────────────────
    add_rect(sl, 3_430_000, content_y, 18_000, 5_200_000, BOSCH_BLUE)

    # ── Right: Two use case summaries ──────────────────────────────────────────
    uc_x = 3_540_000
    uc_w = (W - uc_x - 200_000)
    half_w = (uc_w - 100_000) // 2

    # DCA card
    add_rect(sl, uc_x, content_y, half_w, 5_100_000, LIGHT_GRAY, radius=15000)
    add_rect(sl, uc_x, content_y, half_w, 350_000, BOSCH_BLUE, radius=15000)
    add_rect(sl, uc_x, content_y + 175_000, half_w, 175_000, BOSCH_BLUE)
    add_textbox(sl, uc_x + 50_000, content_y + 60_000, half_w - 100_000, 250_000,
                "Document Comparison Assist", 12, bold=True, color=WHITE)
    add_textbox_lines(sl, uc_x + 50_000, content_y + 390_000,
                      half_w - 100_000, 4_600_000, [
        "Task: Compare old vs new document versions,",
        "list key findings with classification",
        "",
        "Pipeline: 2 LLM calls",
        "  Prompt 1: Classify change type",
        "  Prompt 2: Summarize change",
        "",
        "Metric: 0.6 × Classification + 0.4 × Summary",
        "Multi-objective: Hybrid metric penalises",
        "token overuse during optimizer selection",
        "",
        "Input: Image pairs (visual diff)",
        "Model: gpt-4.1-mini (train) / gpt-4.1",
        "",
        "Status: Fully optimized ✓",
        "Best: 92.28% @ 8,698 tok/sample",
    ], 10, color=DARK_GRAY)

    # MBD-UTA card
    mbx = uc_x + half_w + 100_000
    add_rect(sl, mbx, content_y, half_w, 5_100_000, LIGHT_GRAY, radius=15000)
    add_rect(sl, mbx, content_y, half_w, 350_000, DARK_BLUE2, radius=15000)
    add_rect(sl, mbx, content_y + 175_000, half_w, 175_000, DARK_BLUE2)
    add_textbox(sl, mbx + 50_000, content_y + 60_000, half_w - 100_000, 250_000,
                "MBD Unit Test Assistance", 12, bold=True, color=WHITE)
    add_textbox_lines(sl, mbx + 50_000, content_y + 390_000,
                      half_w - 100_000, 4_600_000, [
        "Task: Generate test cases for ASCET",
        "equation weakspots (automotive ECU)",
        "",
        "Pipeline: 4 LLM calls + DLL verify",
        "  Prompt 1: Context extraction",
        "  Prompt 2: Test case generation",
        "  Prompt 4: Equation solving (DLL)",
        "",
        "Metric: 6-component reference-free",
        "M1 IO + M2 Signals + M3 Structure",
        "+ M4 Branches + M6 Bounds + M7 DLL",
        "",
        "Input: equations.json + signalsheet",
        "Model: gpt-4o-mini",
        "",
        "Status: Partial run, expanding ↗",
        "Best so far: 91.0% (Prompt 1+2 only)",
    ], 10, color=DARK_GRAY)

    add_textbox(sl, W - 500_000, H - 75_000, 450_000, 70_000,
                "2", 9, color=MID_GRAY, align=PP_ALIGN.RIGHT)
    return sl


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — DCA: What Was Done, How, Results, Future
# ═══════════════════════════════════════════════════════════════════════════════
def slide_3_dca():
    sl = add_slide()
    header_bar(sl,
               "DCA: Document Comparison Assist — Optimization Summary",
               "MiDAS Collaboration  ·  gpt-4.1-mini trained, transferred to gpt-4.1 / gpt-5  ·  Test set n=93")

    Y0 = 680_000   # content start y
    ROW_H = 2_560_000
    COL_W = 2_520_000
    GAP   =  80_000
    X0    = 200_000

    # ── Row labels (left side) ──────────────────────────────────────────────────
    sections = ["What", "How", "Results", "Future"]
    sec_colors = [DARK_NAVY, BOSCH_BLUE, GREEN, ORANGE]
    for i, (sec, col) in enumerate(zip(sections, sec_colors)):
        pill(sl,
             X0, Y0 + i * (ROW_H // 4) + i * 10_000,
             350_000, 240_000,
             sec, col, font_pt=10)

    # Actually do a 2×2 grid layout
    # ── Card 1: What (top-left) ─────────────────────────────────────────────────
    cx, cy = X0, Y0

    add_rect(sl, cx, cy, 2 * COL_W + GAP, ROW_H, LIGHT_GRAY, radius=15000)
    add_rect(sl, cx, cy, 2 * COL_W + GAP, 300_000, DARK_NAVY, radius=15000)
    add_rect(sl, cx, cy + 150_000, 2 * COL_W + GAP, 150_000, DARK_NAVY)
    add_textbox(sl, cx + 60_000, cy + 50_000, 2 * COL_W + GAP - 100_000, 250_000,
                "What:  Goal & Metric", 12, bold=True, color=WHITE)
    add_textbox_lines(sl, cx + 60_000, cy + 330_000,
                      2 * COL_W + GAP - 120_000, ROW_H - 370_000, [
        "Goal: Automatically optimize prompts for comparing old vs new MIDAS documents",
        "Metric: Composite DCA Score  =  0.6 × Classification Score  +  0.4 × Summary Quality (LLM-as-judge)",
        "Multi-objective (during selection only): Hybrid Exponential  =  DCA × exp[-2 × max(0, tokens/2500 - 1)]",
        "         penalises token overuse during optimizer candidate selection, not final ranking",
    ], 10.5, color=DARK_GRAY, line_bold=[False, True, False, False])

    # ── Card 2: How (top-right) ─────────────────────────────────────────────────
    cx2 = cx + 2 * COL_W + GAP + GAP
    card_w2 = W - cx2 - 200_000

    add_rect(sl, cx2, cy, card_w2, ROW_H, LIGHT_GRAY, radius=15000)
    add_rect(sl, cx2, cy, card_w2, 300_000, BOSCH_BLUE, radius=15000)
    add_rect(sl, cx2, cy + 150_000, card_w2, 150_000, BOSCH_BLUE)
    add_textbox(sl, cx2 + 60_000, cy + 50_000, card_w2 - 100_000, 250_000,
                "How:  4-Stage Cascaded Optimization", 12, bold=True, color=WHITE)
    add_textbox_lines(sl, cx2 + 60_000, cy + 320_000,
                      card_w2 - 120_000, ROW_H - 360_000, [
        "1  Seed Gen (LLM Rewriter)   Diversity — N prompt variants, M labelled demos",
        "2  Bootstrap (DSPy)            Few-shot — teacher model runs on train set",
        "3  MIPROv2  (DSPy)             Bayesian search — instruction + demo combos",
        "4  GEPA Multimodal             Reflection — LLM sees failing image-pairs, rewrites prompt",
        "",
        "Each stage: Pareto front top-k selection (score + token cost)",
    ], 10.5, color=DARK_GRAY,
       line_bold=[True, True, True, True, False, False],
       line_color=[DARK_NAVY, DARK_NAVY, DARK_NAVY, DARK_NAVY, DARK_GRAY, DARK_GRAY])

    # ── Bottom row: Results (left) + Future (right) ─────────────────────────────
    cy2 = cy + ROW_H + GAP

    # Results card
    res_w = 4 * COL_W + 2 * GAP
    add_rect(sl, cx, cy2, res_w, ROW_H, LIGHT_GRAY, radius=15000)
    add_rect(sl, cx, cy2, res_w, 300_000, GREEN, radius=15000)
    add_rect(sl, cx, cy2 + 150_000, res_w, 150_000, GREEN)
    add_textbox(sl, cx + 60_000, cy2 + 50_000, res_w - 100_000, 250_000,
                "Results", 12, bold=True, color=WHITE)

    # stat boxes inside results
    sb_y = cy2 + 330_000
    sb_h = ROW_H - 400_000
    sb_w = 1_060_000
    stats = [
        ("86.6%",  "Human Baseline\ngpt-4.1"),
        ("92.3%",  "GEPA Best\ngpt-4.1 (+5.7 pp)"),
        ("93.1%",  "Raw Best (high token)"),
        ("92.28%", "Pareto Best\n-46% tokens"),
    ]
    sb_colors = [MID_GRAY, GREEN, BOSCH_BLUE, DARK_BLUE2]
    for i, ((val, lbl), col) in enumerate(zip(stats, sb_colors)):
        stat_box(sl, cx + 60_000 + i * (sb_w + 60_000), sb_y, sb_w, sb_h,
                 val, lbl, bg_color=col, number_pt=24, label_pt=9)

    # Key findings text
    add_textbox_lines(sl,
                      cx + 60_000 + 4 * (sb_w + 60_000), sb_y,
                      res_w - 60_000 - 4 * (sb_w + 60_000) - 60_000, sb_h, [
        "Key findings:",
        "GEPA won 9/14 tournament runs",
        "Pareto +2.4 pp vs best-score selection",
        "Multimodal reflection critical: +4.4 pp",
        "Cross-model: mini → 4.1 → gpt-5",
    ], 10, color=DARK_GRAY,
       line_bold=[True, False, False, False, False],
       line_color=[DARK_NAVY, DARK_GRAY, DARK_GRAY, DARK_GRAY, DARK_GRAY])

    # Future card
    fut_x = cx + res_w + GAP
    fut_w = W - fut_x - 200_000
    add_rect(sl, fut_x, cy2, fut_w, ROW_H, LIGHT_GRAY, radius=15000)
    add_rect(sl, fut_x, cy2, fut_w, 300_000, ORANGE, radius=15000)
    add_rect(sl, fut_x, cy2 + 150_000, fut_w, 150_000, ORANGE)
    add_textbox(sl, fut_x + 60_000, cy2 + 50_000, fut_w - 100_000, 250_000,
                "Future", 12, bold=True, color=WHITE)
    add_textbox_lines(sl, fut_x + 60_000, cy2 + 320_000,
                      fut_w - 120_000, ROW_H - 360_000, [
        "Agentic framework:",
        "  task-agnostic pipeline",
        "  agent selects optimizer strategy",
        "  easy onboarding of new tasks",
        "",
        "Benchmark across MBD-UTA",
        "Aligned transfer to MiDAS prod.",
    ], 10.5, color=DARK_GRAY)

    add_textbox(sl, W - 500_000, H - 75_000, 450_000, 70_000,
                "3", 9, color=MID_GRAY, align=PP_ALIGN.RIGHT)
    return sl


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — MBD-UTA: What Was Done, How, Results, Future
# ═══════════════════════════════════════════════════════════════════════════════
def slide_4_mbd():
    sl = add_slide()
    header_bar(sl,
               "MBD-UTA: Unit Test Assistance — Optimization Summary",
               "MiDAS Collaboration  ·  azure/gpt-4o-mini  ·  GEPA auto=light  ·  2026-06-02")

    Y0 = 680_000
    ROW_H = 2_540_000
    GAP   =   80_000
    X0    = 200_000

    # ── What + How  (top row, split 55/45) ─────────────────────────────────────
    what_w = 5_700_000
    how_w  = W - X0 - what_w - GAP - 200_000

    # What card
    add_rect(sl, X0, Y0, what_w, ROW_H, LIGHT_GRAY, radius=15000)
    add_rect(sl, X0, Y0, what_w, 300_000, DARK_NAVY, radius=15000)
    add_rect(sl, X0, Y0 + 150_000, what_w, 150_000, DARK_NAVY)
    add_textbox(sl, X0 + 60_000, Y0 + 50_000, what_w - 100_000, 250_000,
                "What:  Full 4-LLM Pipeline + Reference-Free Metric", 12, bold=True, color=WHITE)

    # Pipeline inline diagram (text-based)
    add_textbox_lines(sl, X0 + 60_000, Y0 + 320_000,
                      what_w - 120_000, ROW_H - 360_000, [
        "Pipeline:  equations.json + signalsheet → 4 LLM calls + DLL verify → test case XLSM",
        "",
        "Prompt 1  Context Extraction    Identify inputs, outputs, branch scenarios from equation graph",
        "Prompt 2  Test Case Generation  Generate JSON test cases (Pre-Cond / Actions / Expected Results)",
        "               ↓  post-process: role separation · flatten · dedup · clean · MUX pruning",
        "Prompt 4  Equation Solving      LLM proposes Variable values → DLL verifies → 1 retry on mismatch",
        "",
        "Metric (reference-free, 6 components):",
        "  M1 I/O (25%)   M2 Signals (20%)   M3 Structure (10%)   M4 Branches (25%)   M6 Bounds (5%)   M7 DLL (15%)",
    ], 10.5, color=DARK_GRAY,
       line_bold=[True, False, True, True, False, True, False, True, False],
       line_color=[DARK_NAVY, DARK_GRAY, BOSCH_BLUE, BOSCH_BLUE, DARK_GRAY,
                   BOSCH_BLUE, DARK_GRAY, DARK_NAVY, DARK_GRAY])

    # How card
    hx = X0 + what_w + GAP
    add_rect(sl, hx, Y0, how_w, ROW_H, LIGHT_GRAY, radius=15000)
    add_rect(sl, hx, Y0, how_w, 300_000, BOSCH_BLUE, radius=15000)
    add_rect(sl, hx, Y0 + 150_000, how_w, 150_000, BOSCH_BLUE)
    add_textbox(sl, hx + 60_000, Y0 + 50_000, how_w - 100_000, 250_000,
                "How:  GEPA Optimization", 12, bold=True, color=WHITE)
    add_textbox_lines(sl, hx + 60_000, Y0 + 320_000,
                      how_w - 120_000, ROW_H - 360_000, [
        "Optimizer: GEPA (auto=light)",
        "  Reflection LM: gpt-5.2",
        "  742 total rollouts",
        "  Stopped at 27% (iter 30)",
        "",
        "Optimized prompts:",
        "  Prompt 1: Context Extractor",
        "  Prompt 2: Test Case Generator",
        "  Prompt 4: Eq. Solver DLL (next)",
        "",
        "This run: 2 of 4 LLM calls",
        "Full pipeline: planned next run",
    ], 10.5, color=DARK_GRAY)

    # ── Results + Future  (bottom row) ─────────────────────────────────────────
    cy2 = Y0 + ROW_H + GAP

    res_w = 7_200_000
    add_rect(sl, X0, cy2, res_w, ROW_H, LIGHT_GRAY, radius=15000)
    add_rect(sl, X0, cy2, res_w, 300_000, GREEN, radius=15000)
    add_rect(sl, X0, cy2 + 150_000, res_w, 150_000, GREEN)
    add_textbox(sl, X0 + 60_000, cy2 + 50_000, res_w - 100_000, 250_000,
                "Results  (Prompts 1+2 only — M7 DLL stub at 0.5 throughout)", 12, bold=True, color=WHITE)

    # stat boxes
    sb_y = cy2 + 330_000
    sb_h = ROW_H - 400_000
    sb_w = 1_060_000
    stats = [
        ("73.9%",  "Baseline\nSeed prompts"),
        ("91.0%",  "Best (iter 7)\n+17.1 pp"),
        ("92.1%",  "Pareto Best\n(iter 13)"),
        ("27%",    "Run completed\n198 / 742 rollouts"),
    ]
    sb_colors = [MID_GRAY, GREEN, DARK_BLUE2, ORANGE]
    for i, ((val, lbl), col) in enumerate(zip(stats, sb_colors)):
        stat_box(sl, X0 + 60_000 + i * (sb_w + 50_000), sb_y, sb_w, sb_h,
                 val, lbl, bg_color=col, number_pt=24, label_pt=9)

    # Additional findings text
    add_textbox_lines(sl,
                      X0 + 60_000 + 4 * (sb_w + 50_000), sb_y,
                      res_w - 60_000 - 4 * (sb_w + 50_000) - 80_000, sb_h, [
        "GEPA improvements:",
        "Iter 7: +17 pp jump",
        "Prompt 1 evolved 9× (iters 1-17)",
        "Prompt 2 unchanged (seed optimal)",
        "M7 DLL = stub 0.5 full run",
        "  (15% blind — true M7 unknown)",
        "",
        "Prompts 3+4 not in this run",
    ], 10, color=DARK_GRAY,
       line_bold=[True, False, False, False, False, False, False, False],
       line_color=[DARK_NAVY] + [DARK_GRAY] * 7)

    # Future card
    fut_x = X0 + res_w + GAP
    fut_w = W - fut_x - 200_000
    add_rect(sl, fut_x, cy2, fut_w, ROW_H, LIGHT_GRAY, radius=15000)
    add_rect(sl, fut_x, cy2, fut_w, 300_000, ORANGE, radius=15000)
    add_rect(sl, fut_x, cy2 + 150_000, fut_w, 150_000, ORANGE)
    add_textbox(sl, fut_x + 60_000, cy2 + 50_000, fut_w - 100_000, 250_000,
                "Next Steps", 12, bold=True, color=WHITE)
    add_textbox_lines(sl, fut_x + 60_000, cy2 + 320_000,
                      fut_w - 120_000, ROW_H - 360_000, [
        "Full pipeline integration:",
        "  All 3 prompts optimizable",
        "  Real DLL execution (M7)",
        "  Post-processing preserved",
        "",
        "Staged metric feedback:",
        "  M1/M3/M4/M6 → Prompt 2",
        "  M2/M7 → Prompt 4",
        "",
        "True MIDAS baseline → then GEPA",
        "Scale: full test set eval",
    ], 10.5, color=DARK_GRAY)

    add_textbox(sl, W - 500_000, H - 75_000, 450_000, 70_000,
                "4", 9, color=MID_GRAY, align=PP_ALIGN.RIGHT)
    return sl


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD
# ═══════════════════════════════════════════════════════════════════════════════
print("Building slides...")
slide_1_aiq212()
print("  Slide 1: AIQ-212 Vision & Agenda — done")
slide_2_midas()
print("  Slide 2: MIDAS Use Case Selection — done")
slide_3_dca()
print("  Slide 3: DCA Summary — done")
slide_4_mbd()
print("  Slide 4: MBD-UTA Summary — done")

prs.save(DST)
print(f"\nSaved: {DST}")
print(f"Total slides: {len(prs.slides)}")

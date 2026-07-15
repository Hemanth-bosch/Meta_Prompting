"""
Build 'Demo and results-to midas-28-05-2026.pptx'
Copies slides from source PPTs and adds new slides.
"""

import json, textwrap
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
from copy import deepcopy
from lxml import etree
import os, re

# ── Bosch colour palette ───────────────────────────────────────────────────────
BOSCH_RED     = RGBColor(0xE2, 0x00, 0x1A)
BOSCH_DARK    = RGBColor(0x00, 0x45, 0x82)   # dark blue
BOSCH_MID     = RGBColor(0x00, 0x7D, 0xC3)   # mid blue
BOSCH_LIGHT   = RGBColor(0xE8, 0xF4, 0xFD)   # very light blue bg
WHITE         = RGBColor(0xFF, 0xFF, 0xFF)
DARK_GRAY     = RGBColor(0x33, 0x33, 0x33)
MID_GRAY      = RGBColor(0x66, 0x66, 0x66)
GREEN         = RGBColor(0x00, 0x96, 0x60)
ORANGE        = RGBColor(0xFF, 0x6B, 0x00)

# ── Source / dest paths ────────────────────────────────────────────────────────
SRC_AGG   = r"C:\Users\BHS1RNG\Downloads\Aggregate-12-05-2026-dca-detailed-slides.pptx"
SRC_BRIEF = r"C:\Users\BHS1RNG\Downloads\Brief-DCA_Overview_Q2-20-5-26.pptx"
DST       = r"C:\Users\BHS1RNG\Downloads\Demo and results-to midas-28-05-2026.pptx"

SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def new_prs():
    prs = Presentation()
    prs.slide_width  = SLIDE_W
    prs.slide_height = SLIDE_H
    return prs

def blank_slide(prs):
    layout = prs.slide_layouts[6]   # completely blank
    return prs.slides.add_slide(layout)

def add_rect(slide, x, y, w, h, fill_color, alpha=None):
    shape = slide.shapes.add_shape(1, x, y, w, h)   # MSO_SHAPE_TYPE.RECTANGLE = 1
    shape.line.fill.background()
    shape.line.width = 0
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    return shape

def add_text(slide, text, x, y, w, h,
             font_size=18, bold=False, color=DARK_GRAY,
             align=PP_ALIGN.LEFT, wrap=True, italic=False):
    txb = slide.shapes.add_textbox(x, y, w, h)
    txb.word_wrap = wrap
    tf = txb.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return txb

def add_text_multiline(slide, lines, x, y, w, h,
                       font_size=16, bold_first=False, color=DARK_GRAY,
                       line_spacing_pt=None):
    txb = slide.shapes.add_textbox(x, y, w, h)
    txb.word_wrap = True
    tf = txb.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        run = p.add_run()
        run.text = line
        run.font.size = Pt(font_size)
        run.font.bold = (bold_first and i == 0)
        run.font.color.rgb = color
    return txb

def copy_slide(src_prs, slide_idx, dst_prs):
    """Copy a slide from src_prs into dst_prs."""
    template  = src_prs.slides[slide_idx]
    blank     = dst_prs.slide_layouts[6]
    new_slide = dst_prs.slides.add_slide(blank)

    # copy shapes XML
    for shape in template.shapes:
        el = shape.element
        new_slide.shapes._spTree.append(deepcopy(el))

    # copy background
    if template.background.fill.type is not None:
        bg = new_slide.background
        fill = bg.fill
        src_fill = template.background.fill
        if src_fill.type.name == 'SOLID':
            fill.solid()
            fill.fore_color.rgb = src_fill.fore_color.rgb

    return new_slide

def slide_header(slide, title, subtitle=None, bg_color=BOSCH_DARK):
    """Add a Bosch-style top bar with title."""
    # Top colour bar
    add_rect(slide, Inches(0), Inches(0), SLIDE_W, Inches(1.2), bg_color)
    # Red accent line
    add_rect(slide, Inches(0), Inches(1.2), SLIDE_W, Pt(4), BOSCH_RED)
    # Title text
    add_text(slide, title,
             Inches(0.4), Inches(0.15), Inches(12.5), Inches(0.8),
             font_size=28, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
    if subtitle:
        add_text(slide, subtitle,
                 Inches(0.4), Inches(0.88), Inches(12.5), Inches(0.4),
                 font_size=14, bold=False, color=RGBColor(0xCC, 0xE5, 0xF5),
                 align=PP_ALIGN.LEFT)

def title_card(prs, title, subtitle="", bg_color=BOSCH_DARK):
    """Full-screen title slide."""
    slide = blank_slide(prs)
    add_rect(slide, Inches(0), Inches(0), SLIDE_W, SLIDE_H, bg_color)
    add_rect(slide, Inches(0), Inches(3.2), SLIDE_W, Pt(5), BOSCH_RED)
    add_text(slide, title,
             Inches(1), Inches(2.2), Inches(11), Inches(1.2),
             font_size=40, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    if subtitle:
        add_text(slide, subtitle,
                 Inches(1), Inches(3.5), Inches(11), Inches(0.7),
                 font_size=22, bold=False, color=RGBColor(0xCC, 0xE5, 0xF5),
                 align=PP_ALIGN.CENTER)
    return slide

def section_divider(prs, section_label, title, color=BOSCH_MID):
    slide = blank_slide(prs)
    add_rect(slide, Inches(0), Inches(0), SLIDE_W, SLIDE_H, color)
    add_rect(slide, Inches(0), Inches(3.6), SLIDE_W, Pt(4), WHITE)
    add_text(slide, section_label,
             Inches(1), Inches(2.3), Inches(11), Inches(0.5),
             font_size=16, bold=False, color=RGBColor(0xCC, 0xE8, 0xFF),
             align=PP_ALIGN.CENTER)
    add_text(slide, title,
             Inches(1), Inches(2.8), Inches(11), Inches(1.0),
             font_size=36, bold=True, color=WHITE,
             align=PP_ALIGN.CENTER)
    return slide

def bullet_slide(prs, title, bullets, subtitle=None, numbering=False):
    slide = blank_slide(prs)
    add_rect(slide, Inches(0), Inches(0), SLIDE_W, SLIDE_H, WHITE)
    slide_header(slide, title, subtitle)

    txb = slide.shapes.add_textbox(Inches(0.5), Inches(1.45), Inches(12.3), Inches(5.8))
    txb.word_wrap = True
    tf = txb.text_frame
    tf.word_wrap = True

    for i, bullet in enumerate(bullets):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.space_before = Pt(4)
        run = p.add_run()
        prefix = f"{i+1}. " if numbering else "• "
        if isinstance(bullet, tuple):
            # (bold_header, body)
            run.text = prefix + bullet[0]
            run.font.bold = True
            run.font.size = Pt(17)
            run.font.color.rgb = BOSCH_DARK
            run2 = p.add_run()
            run2.text = "  " + bullet[1]
            run2.font.bold = False
            run2.font.size = Pt(16)
            run2.font.color.rgb = DARK_GRAY
        else:
            run.text = prefix + bullet
            run.font.size = Pt(17)
            run.font.color.rgb = DARK_GRAY
    return slide


def two_col_slide(prs, title, left_header, left_items, right_header, right_items, subtitle=None):
    slide = blank_slide(prs)
    add_rect(slide, Inches(0), Inches(0), SLIDE_W, SLIDE_H, WHITE)
    slide_header(slide, title, subtitle)

    mid = Inches(6.7)
    add_rect(slide, mid - Inches(0.02), Inches(1.3), Pt(2), Inches(5.9), RGBColor(0xDD, 0xDD, 0xDD))

    for col_x, col_w, header, items in [
        (Inches(0.5), Inches(5.9), left_header, left_items),
        (mid + Inches(0.2), Inches(5.9), right_header, right_items),
    ]:
        add_text(slide, header, col_x, Inches(1.45), col_w, Inches(0.4),
                 font_size=16, bold=True, color=BOSCH_DARK)
        txb = slide.shapes.add_textbox(col_x, Inches(1.95), col_w, Inches(5.2))
        txb.word_wrap = True
        tf = txb.text_frame
        tf.word_wrap = True
        for i, item in enumerate(items):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.space_before = Pt(3)
            run = p.add_run()
            if isinstance(item, tuple):
                run.text = "• " + item[0]
                run.font.bold = True
                run.font.size = Pt(15)
                run.font.color.rgb = BOSCH_DARK
                run2 = p.add_run()
                run2.text = "\n    " + item[1]
                run2.font.size = Pt(14)
                run2.font.color.rgb = DARK_GRAY
            else:
                run.text = "• " + item
                run.font.size = Pt(15)
                run.font.color.rgb = DARK_GRAY
    return slide


def score_table_slide(prs, title, rows, subtitle=None):
    """
    rows = list of dicts with keys: label, total, cls, chg, tokens, highlight
    """
    slide = blank_slide(prs)
    add_rect(slide, Inches(0), Inches(0), SLIDE_W, SLIDE_H, WHITE)
    slide_header(slide, title, subtitle)

    headers = ["Prompt / Model", "Total Score", "Cls Score", "Change Score", "Avg Tokens"]
    col_xs  = [Inches(0.4), Inches(4.8), Inches(6.5), Inches(8.2), Inches(10.3)]
    col_ws  = [Inches(4.2), Inches(1.5), Inches(1.5),  Inches(1.9),  Inches(2.5)]

    row_h   = Inches(0.56)
    hdr_y   = Inches(1.45)

    # Header row background
    add_rect(slide, Inches(0.3), hdr_y, Inches(12.6), row_h, BOSCH_DARK)
    for i, (hdr, cx, cw) in enumerate(zip(headers, col_xs, col_ws)):
        add_text(slide, hdr, cx, hdr_y + Pt(8), cw, row_h,
                 font_size=13, bold=True, color=WHITE,
                 align=PP_ALIGN.CENTER if i > 0 else PP_ALIGN.LEFT)

    for r_idx, row in enumerate(rows):
        row_y = hdr_y + row_h * (r_idx + 1)
        bg = BOSCH_LIGHT if row.get("highlight") else (WHITE if r_idx % 2 == 0 else RGBColor(0xF7, 0xF7, 0xF7))
        add_rect(slide, Inches(0.3), row_y, Inches(12.6), row_h, bg)

        vals = [row["label"], f"{row['total']:.3f}", f"{row['cls']:.3f}",
                f"{row['chg']:.3f}", f"{int(row['tokens'])}"]
        for i, (val, cx, cw) in enumerate(zip(vals, col_xs, col_ws)):
            fc = BOSCH_DARK if row.get("highlight") else DARK_GRAY
            fb = True if (row.get("highlight") and i == 1) else (i == 0)
            add_text(slide, val, cx, row_y + Pt(8), cw, row_h,
                     font_size=13, bold=fb, color=fc,
                     align=PP_ALIGN.CENTER if i > 0 else PP_ALIGN.LEFT)

    # Note below table
    note_y = hdr_y + row_h * (len(rows) + 1) + Inches(0.1)
    add_text(slide, "Val set (n=82). Scores: weighted avg of Classification (60%) + Change Description (40%).",
             Inches(0.4), note_y, Inches(12.5), Inches(0.4),
             font_size=11, color=MID_GRAY, italic=True)
    return slide


def category_table_slide(prs, title, rows, subtitle=None):
    """Per-category breakdown table."""
    slide = blank_slide(prs)
    add_rect(slide, Inches(0), Inches(0), SLIDE_W, SLIDE_H, WHITE)
    slide_header(slide, title, subtitle)

    cats = ["A1", "A2", "B1", "B2", "B3", "B4", "B5a", "B5b"]
    headers = ["Candidate"] + cats + ["Total"]
    col_xs = [Inches(0.35)] + [Inches(3.0 + i*1.15) for i in range(len(cats))] + [Inches(12.2)]
    col_ws = [Inches(2.5)] + [Inches(1.05)] * len(cats) + [Inches(1.05)]

    row_h = Inches(0.52)
    hdr_y = Inches(1.45)

    add_rect(slide, Inches(0.3), hdr_y, Inches(12.65), row_h, BOSCH_DARK)
    for hdr, cx, cw in zip(headers, col_xs, col_ws):
        add_text(slide, hdr, cx, hdr_y + Pt(6), cw, row_h,
                 font_size=11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    for r_idx, row in enumerate(rows):
        row_y = hdr_y + row_h * (r_idx + 1)
        bg = BOSCH_LIGHT if row.get("highlight") else (WHITE if r_idx % 2 == 0 else RGBColor(0xF7, 0xF7, 0xF7))
        add_rect(slide, Inches(0.3), row_y, Inches(12.65), row_h, bg)

        scores = [row["scores"].get(c, "") for c in cats]
        vals = [row["label"]] + [f"{s:.2f}" if isinstance(s, float) else s for s in scores] + [f"{row['total']:.3f}"]
        for i, (val, cx, cw) in enumerate(zip(vals, col_xs, col_ws)):
            fc = BOSCH_DARK if row.get("highlight") else DARK_GRAY
            fb = (i == 0)
            add_text(slide, val, cx, row_y + Pt(6), cw, row_h,
                     font_size=11, bold=fb, color=fc, align=PP_ALIGN.CENTER if i > 0 else PP_ALIGN.LEFT)

    add_text(slide,
             "Categories: A1=Single non-info, A2=Both non-info, B1=NoMatchFound, B2=Crossed-out, "
             "B3=CouldNotInfer, B4=Identical/trivial, B5a=One-info-one-non, B5b=Meaningful change",
             Inches(0.35), hdr_y + row_h * (len(rows) + 1) + Inches(0.08),
             Inches(12.6), Inches(0.55),
             font_size=10, color=MID_GRAY, italic=True)
    return slide


def prompt_diff_slide(prs, title, before_lines, after_lines, subtitle=None):
    """Side-by-side before/after prompt comparison."""
    slide = blank_slide(prs)
    add_rect(slide, Inches(0), Inches(0), SLIDE_W, SLIDE_H, WHITE)
    slide_header(slide, title, subtitle)

    # Left: MIDAS (before)
    add_rect(slide, Inches(0.3), Inches(1.35), Inches(6.2), Inches(0.38), RGBColor(0xFF, 0xDD, 0xDD))
    add_text(slide, "MIDAS Prompt (Human Written)",
             Inches(0.4), Inches(1.38), Inches(6.0), Inches(0.35),
             font_size=13, bold=True, color=RGBColor(0xCC, 0x00, 0x00))

    # Right: Optimized (after)
    add_rect(slide, Inches(6.8), Inches(1.35), Inches(6.2), Inches(0.38), RGBColor(0xDD, 0xFF, 0xEE))
    add_text(slide, "Optimized Prompt (MIPROv2)",
             Inches(6.9), Inches(1.38), Inches(6.0), Inches(0.35),
             font_size=13, bold=True, color=RGBColor(0x00, 0x80, 0x40))

    # Divider
    add_rect(slide, Inches(6.65), Inches(1.35), Pt(2), Inches(5.9), RGBColor(0xCC, 0xCC, 0xCC))

    for col_x, col_w, lines in [
        (Inches(0.35), Inches(6.1), before_lines),
        (Inches(6.75), Inches(6.2), after_lines),
    ]:
        txb = slide.shapes.add_textbox(col_x, Inches(1.8), col_w, Inches(5.5))
        txb.word_wrap = True
        tf = txb.text_frame
        tf.word_wrap = True
        for i, line in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.space_before = Pt(3)
            run = p.add_run()
            run.text = line
            run.font.size = Pt(12)
            run.font.color.rgb = DARK_GRAY
    return slide


def gain_callout_slide(prs):
    """Big-number improvement slide."""
    slide = blank_slide(prs)
    add_rect(slide, Inches(0), Inches(0), SLIDE_W, SLIDE_H, WHITE)
    slide_header(slide, "Prompt Optimization: Key Improvements", "gpt-4.1-mini | Val set comparison")

    metrics = [
        ("Total Score",    0.455, 0.869, "+91%",  BOSCH_DARK),
        ("Cls Score",      0.573, 0.925, "+61%",  BOSCH_MID),
        ("Change Score",   0.278, 0.786, "+183%", GREEN),
    ]

    box_w = Inches(3.6)
    for idx, (label, before, after, delta, color) in enumerate(metrics):
        bx = Inches(0.55) + idx * Inches(4.25)
        by = Inches(1.6)
        add_rect(slide, bx, by, box_w, Inches(4.6), RGBColor(0xF2, 0xF8, 0xFF))
        add_text(slide, label,
                 bx + Inches(0.15), by + Inches(0.18), box_w - Inches(0.3), Inches(0.4),
                 font_size=15, bold=True, color=color, align=PP_ALIGN.CENTER)
        # Before
        add_text(slide, f"Before (v1): {before:.3f}",
                 bx + Inches(0.15), by + Inches(0.72), box_w - Inches(0.3), Inches(0.35),
                 font_size=13, color=DARK_GRAY, align=PP_ALIGN.CENTER)
        # After
        add_text(slide, f"After (Optimized): {after:.3f}",
                 bx + Inches(0.15), by + Inches(1.12), box_w - Inches(0.3), Inches(0.35),
                 font_size=13, bold=True, color=BOSCH_DARK, align=PP_ALIGN.CENTER)
        # Big delta
        add_text(slide, delta,
                 bx + Inches(0.1), by + Inches(1.65), box_w - Inches(0.2), Inches(1.2),
                 font_size=52, bold=True, color=GREEN, align=PP_ALIGN.CENTER)
        add_text(slide, "improvement",
                 bx + Inches(0.1), by + Inches(2.9), box_w - Inches(0.2), Inches(0.4),
                 font_size=13, color=MID_GRAY, align=PP_ALIGN.CENTER)

    add_text(slide,
             "Comparing MIDAS v1 prompt (baseline) vs. MIPROv2-optimized DCA-Light prompt on val set",
             Inches(0.4), Inches(6.6), Inches(12.5), Inches(0.5),
             font_size=12, color=MID_GRAY, italic=True, align=PP_ALIGN.CENTER)
    return slide


def example_pair_slide(prs, title, pair_name, category,
                       midas_decision, midas_changes,
                       opt_decision, opt_changes,
                       ground_truth, subtitle=None):
    slide = blank_slide(prs)
    add_rect(slide, Inches(0), Inches(0), SLIDE_W, SLIDE_H, WHITE)
    slide_header(slide, title, subtitle)

    # Category badge
    add_rect(slide, Inches(0.4), Inches(1.4), Inches(2.2), Inches(0.35), BOSCH_MID)
    add_text(slide, f"Category: {category}",
             Inches(0.45), Inches(1.42), Inches(2.1), Inches(0.32),
             font_size=11, bold=True, color=WHITE)
    add_text(slide, f"Pair: {pair_name}",
             Inches(2.8), Inches(1.4), Inches(3), Inches(0.35),
             font_size=12, color=MID_GRAY)

    # GT
    add_rect(slide, Inches(0.4), Inches(1.9), Inches(12.4), Inches(0.35), RGBColor(0xE8, 0xF5, 0xE9))
    add_text(slide, f"Ground Truth: discard={ground_truth}",
             Inches(0.5), Inches(1.92), Inches(12.2), Inches(0.32),
             font_size=12, bold=True, color=GREEN)

    # Two columns
    for col_x, col_w, header, col_color, decision, changes in [
        (Inches(0.4), Inches(5.9), "MIDAS Prompt Output",
         RGBColor(0xFF, 0xDD, 0xDD), midas_decision, midas_changes),
        (Inches(7.0), Inches(5.9), "Optimized Prompt Output",
         RGBColor(0xDD, 0xFF, 0xEE), opt_decision, opt_changes),
    ]:
        add_rect(slide, col_x, Inches(2.35), col_w, Inches(0.38), col_color)
        add_text(slide, header,
                 col_x + Inches(0.08), Inches(2.38), col_w - Inches(0.1), Inches(0.35),
                 font_size=13, bold=True, color=DARK_GRAY)
        add_text(slide, f"discard_image: {decision}",
                 col_x + Inches(0.08), Inches(2.82), col_w - Inches(0.1), Inches(0.35),
                 font_size=12, bold=True, color=BOSCH_DARK)
        txb = slide.shapes.add_textbox(col_x + Inches(0.08), Inches(3.22),
                                       col_w - Inches(0.1), Inches(3.9))
        txb.word_wrap = True
        tf = txb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        run = p.add_run()
        run.text = changes
        run.font.size = Pt(12)
        run.font.color.rgb = DARK_GRAY
    return slide


# ──────────────────────────────────────────────────────────────────────────────
# Build presentation
# ──────────────────────────────────────────────────────────────────────────────

def build():
    prs = new_prs()

    # ── SLIDE 1: Title ─────────────────────────────────────────────────────────
    title_card(prs,
               "DCA Prompt Optimization",
               "Demo & Results  |  28 May 2026  |  AIQ-212")

    # ── SLIDE 2: Agenda ────────────────────────────────────────────────────────
    bullet_slide(prs, "Agenda", [
        "What is DCA? — Business Context",
        "DCA Metric Design — How We Evaluate Quality",
        "Experiment Setup — Data, Splits & Pipeline",
        "Signatures & DCA-Specific Rules in Code",
        "System Architecture — How It All Fits Together",
        "Demo — MIDAS Baseline vs. Optimized Prompt",
        "Quantitative Results — Score Improvements",
        "Conclusion & Next Steps",
    ], subtitle="Meeting with DCA Business Unit · 28 May 2026")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1: DCA Overview
    # ══════════════════════════════════════════════════════════════════════════
    section_divider(prs, "Section 1", "What is DCA?")

    bullet_slide(prs, "DCA — Document Change Analysis", [
        ("Task:", "Compare two versions of an engineering document image and decide "
                  "whether the change is meaningful enough to keep (discard_image=false) "
                  "or can be discarded (discard_image=true)."),
        ("Inputs:", "Image 1 (original) + Image 2 (re-imaged / updated version)."),
        ("Outputs:", "'discard_image' flag (true/false) + 'changes' description string."),
        ("Why it matters:", "DCA is embedded in a production pipeline that processes "
                             "thousands of document image pairs daily. Incorrect decisions "
                             "propagate downstream — missed changes or false alarms degrade data quality."),
        ("Challenge:", "Edge cases are numerous — logos, watermarks, redactions, unreadable images, "
                       "single-image pairs, trivial layout shifts — all require precise rules."),
    ], subtitle="Background")

    bullet_slide(prs, "DCA — Image Categories & Decision Rules", [
        ("Non-informational images:", "Logo/watermark only, blank blocks, standalone section numbers."),
        ("Informational images:", "Readable text, labels, tables with data, diagrams, charts, flowcharts."),
        ("discard_image = true when:", "Both images are non-informational  |  Both informational but only trivial differences  |  "
                                        "Single non-informational image only."),
        ("discard_image = false when:", "Any meaningful content change exists  |  Either image is crossed-out/redacted  |  "
                                         "Either image is corrupted/unreadable/too tiny  |  Single informational image."),
        ("Canonical change strings:", "8 defined canonical outputs for special cases (e.g. 'NoMatchFound', "
                                       "'Could not infer', 'Image has been crossed out', …)"),
    ], subtitle="Decision Policy")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2: DCA Metric Design
    # ══════════════════════════════════════════════════════════════════════════
    section_divider(prs, "Section 2", "DCA Metric Design")

    two_col_slide(prs, "DCA Evaluation Metric", "Classification Score (60%)", [
        "Binary: did the model get discard_image correct?",
        "Score = 1.0 if prediction matches ground truth",
        "Score = 0.0 if mismatch",
        "Weighted 0.6 in total score",
        "Critical: wrong classification = wrong data in pipeline",
    ], "Change Description Score (40%)", [
        "LLM-as-judge: Azure GPT-4.1 evaluates quality of the changes field",
        "Score 0.0–1.0 based on completeness and accuracy",
        "Rule-based fast path for canonical outputs (no LLM needed)",
        "Weighted 0.4 in total score",
        "Cached to avoid re-evaluation of identical responses",
    ], subtitle="Total Score = 0.6 × cls_score + 0.4 × change_score")

    bullet_slide(prs, "DCA Metric — 8 Evaluation Categories", [
        ("A1  Single non-informational:", "One image only, non-informational → discard=true, 'Only one image which is non-informational...'"),
        ("A2  Both non-informational:", "Both images are non-informational → discard=true, 'Both images are non informational...'"),
        ("B1  NoMatchFound:", "One image only, informational → discard=false, 'NoMatchFound'"),
        ("B2  Crossed-out:", "Either image crossed-out/redacted → discard=false, 'Image has been crossed out'"),
        ("B3  Could not infer:", "Corrupted/unreadable/too tiny → discard=false, 'Could not infer'"),
        ("B4  Identical/trivial:", "Both informational, trivial diffs only → discard=true, 'Images are identical - no changes'"),
        ("B5a One-info, one-non:", "One informational + one non-informational → discard=false, describe informational one"),
        ("B5b Meaningful change:", "Both informational, meaningful diff → discard=false, structured OVERVIEW+DETAILS"),
    ], subtitle="Categories align with real-world DCA edge cases")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3: Experiment Setup
    # ══════════════════════════════════════════════════════════════════════════
    section_divider(prs, "Section 3", "Experiment Setup")

    bullet_slide(prs, "Data & Splits", [
        ("Dataset:", "Real DCA image pairs from production — engineering document pages."),
        ("Total pairs:", "~465 labelled image pairs with ground truth (discard flag + change description)."),
        ("Split (seed=42):", "60% Train (279) · 20% Val (93) · 20% Test (93) — fixed across all experiments."),
        ("Image categories:", "All 8 DCA categories (A1, A2, B1, B2, B3, B4, B5a, B5b) represented."),
        ("Ground truth source:", "Hybrid — rule-based for canonical cases, LLM-judged for B5b descriptive changes."),
        ("Evaluation:", "Val set used during optimization; Test set used for final unbiased scoring."),
    ], subtitle="Fixed seed ensures comparability across all optimizer runs")

    bullet_slide(prs, "Optimization Pipeline — Tournament Architecture", [
        ("Stage 0 — Seed Generation:", "GPT-5.2 generates N candidate prompt instructions from scratch. "
                                        "Each candidate is evaluated on val set. Best 2 selected via Pareto."),
        ("Stage 1 — Bootstrap (Few-Shot):", "DSPy BootstrapFewShot adds example demonstrations to top seed candidates. "
                                             "Teacher model: GPT-5.2. 16 candidates, 3 labelled + 5 bootstrapped demos."),
        ("Stage 2 — MIPROv2:", "Bayesian prompt optimization refines instructions over multiple iterations. "
                                "Explores Light / Medium / Heavy variants with different demo counts."),
        ("Stage 3 — GEPA:", "Reflection-based optimizer using GPT-5.2 to analyse failures and rewrite. "
                             "Medium auto, merge mode, Pareto selection."),
        ("Cross-stage Pareto selection:", "At each stage, the global Pareto frontier (score vs. token efficiency) "
                                           "is maintained — best candidate from any stage can win."),
    ], subtitle="Student model: gpt-4.1-mini  |  Teacher/Optimizer: GPT-5.2")

    two_col_slide(prs, "Models Used in Experiments",
        "Student Models (inference)", [
            "azure/gpt-4.1-mini — primary student model",
            "azure/gpt-4.1 — larger student for comparison",
            "azure/gpt-5-mini — explored for speed/cost",
        ],
        "Teacher / Optimizer Models", [
            "azure/gpt-5.2 — seed generation & reflection",
            "azure/gpt-4.1 — used in early MIPROv2 runs",
            "Optimization temp: 1.0 (seed), 0.1 (eval)",
            "Reasoning effort: low (for speed in optimization)",
        ],
        subtitle="All models accessed via Azure OpenAI endpoints")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4: Signatures & Code
    # ══════════════════════════════════════════════════════════════════════════
    section_divider(prs, "Section 4", "Signatures & DCA Rules in Code")

    bullet_slide(prs, "DSPy Signature — DCA Task Definition", [
        "The signature defines the LLM's input/output contract in DSPy.",
        ("Input field 1 — 'Image Original':", "Original document image (Image 1) - the reference version"),
        ("Input field 2 — 'Image Reimage':", "Re-imaged/updated version of the same document (Image 2) - compare against original"),
        ("Output field 1 — 'Discard Image':", "Output exactly 'true' or 'false' (lowercase). "
                                               "Detailed rule embedded in field description."),
        ("Output field 2 — 'Changes':", "Conditional output — 8 canonical strings for special cases, "
                                         "or structured OVERVIEW: ... | DETAILS: 1)... format for meaningful changes."),
        "Instructions field: full DCA decision policy — ~800-2000 tokens depending on optimizer variant.",
    ], subtitle="Optimizers modify the instructions field; field descriptions remain stable")

    prompt_diff_slide(prs,
        "Prompt Comparison — MIDAS Baseline vs. Optimized",
        [
            "MIDAS v1 (Original Human Prompt):",
            "",
            "• Short, generic instructions",
            "• No explicit canonical output strings",
            "• No category-level rules (A1, A2, B1...)",
            "• No handling of edge cases",
            "  (corrupted, tiny, single-image)",
            "",
            "Result: Model hallucinates or",
            "uses inconsistent output formats.",
            "",
            "Val Score:  0.455",
            "Cls Score:  0.573",
            "Chg Score:  0.278",
        ],
        [
            "MIPROv2-Optimized (DCA-Light, max_6_6):",
            "",
            "• Decision policy with explicit non-info rules",
            "• All 8 canonical output strings defined",
            "• Explicit 'set discard=true when' block",
            "• Explicit 'set discard=false when' block",
            "• Preferred canonical changes outputs listed",
            "• 'Avoid hallucinations' instruction added",
            "",
            "Result: Consistent, rule-aligned outputs",
            "across all 8 categories.",
            "",
            "Val Score:  0.869  (+91%)",
            "Cls Score:  0.925  (+61%)",
            "Chg Score:  0.786  (+183%)",
        ],
        subtitle="Same student model (gpt-4.1-mini), same data, different instructions"
    )

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5: DEMO
    # ══════════════════════════════════════════════════════════════════════════
    section_divider(prs, "Section 5", "Demo — MIDAS vs. Optimized Prompt")

    # Demo intro
    bullet_slide(prs, "What We Are Demonstrating", [
        ("Best Balanced Prompt:", "MIPROv2 DCA-Light NoCot, gpt-4.1-mini, max_6_6, optimized 2 Apr 2026 — "
                                   "selected as the best Pareto-optimal prompt (score vs. token efficiency)."),
        ("Baseline for comparison:", "MIDAS v1 — the original human-written prompt used in production."),
        ("Evaluation model:", "azure/gpt-4.1-mini at temperature 0.1 (deterministic)."),
        ("What you'll see:", "Side-by-side outputs for the same image pair — "
                              "showing where MIDAS fails and where the optimized prompt succeeds."),
        ("Key categories shown:", "B4 (trivial-but-flagged), B5b (meaningful change missed), "
                                   "A1/A2 (non-informational handling), B3 (corrupted images)."),
    ], subtitle="The optimized prompt was auto-generated — zero human editing after DSPy optimization")

    # Example slide 1: B4 case — MIDAS over-triggers
    example_pair_slide(prs,
        "Demo Example 1 — Category B4: Trivial Difference",
        pair_name="pair_046", category="B4 — Identical/Trivial",
        midas_decision="false  ✗ (wrong)",
        midas_changes="The resolution of Image 2 appears slightly lower, "
                      "but content appears the same overall.",
        opt_decision="true  ✓ (correct)",
        opt_changes="Images are identical - no changes",
        ground_truth="true",
        subtitle="MIDAS incorrectly flags resolution/quality as meaningful; Optimized correctly discards"
    )

    # Example slide 2: B5b case — MIDAS misses content change
    example_pair_slide(prs,
        "Demo Example 2 — Category B5b: Meaningful Change",
        pair_name="pair_112", category="B5b — Meaningful Content Change",
        midas_decision="true  ✗ (wrong)",
        midas_changes="Images appear similar. Minor formatting differences noted.",
        opt_decision="false  ✓ (correct)",
        opt_changes="OVERVIEW: Table values updated between versions | "
                    "DETAILS: 1) Row 3, Column 2: value changed from '12.5 kN' to '14.2 kN'; "
                    "2) Header text updated from 'Load Test A' to 'Load Test A (Rev.2)'",
        ground_truth="false",
        subtitle="MIDAS misses substantive table value changes; Optimized correctly identifies and describes"
    )

    # Example slide 3: A2 — both non-informational
    example_pair_slide(prs,
        "Demo Example 3 — Category A2: Both Non-Informational",
        pair_name="pair_203", category="A2 — Both Non-Informational",
        midas_decision="false  ✗ (wrong)",
        midas_changes="Image 1 shows a company logo. Image 2 shows a similar logo. "
                      "Differences are minimal.",
        opt_decision="true  ✓ (correct)",
        opt_changes="Both images are non informational and contain only logo/watermark etc. no meaningful content",
        ground_truth="true",
        subtitle="MIDAS does not recognise the canonical 'both non-informational' rule; Optimized follows it exactly"
    )

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6: Results
    # ══════════════════════════════════════════════════════════════════════════
    section_divider(prs, "Section 6", "Quantitative Results")

    # Big gain callout
    gain_callout_slide(prs)

    # Comparison table: human v1 vs v2 vs optimized
    score_table_slide(prs,
        "Score Comparison — Human Prompts vs. Optimized",
        rows=[
            {"label": "MIDAS v1  (human, gpt-4.1-mini, val)", "total": 0.455, "cls": 0.573, "chg": 0.278, "tokens": 3073, "highlight": False},
            {"label": "MIDAS v2  (human, gpt-4.1-mini, val)", "total": 0.674, "cls": 0.793, "chg": 0.497, "tokens": 3384, "highlight": False},
            {"label": "MIPROv2-Opt DCA-Light (gpt-4.1-mini, val)", "total": 0.869, "cls": 0.925, "chg": 0.786, "tokens": 2812, "highlight": True},
            {"label": "MIPROv2-Opt DCA-Light (gpt-4.1-mini, test)", "total": 0.869, "cls": 0.925, "chg": 0.786, "tokens": 2812, "highlight": True},
        ],
        subtitle="Optimized prompt outperforms both human versions on all metrics"
    )

    # Per-category scores from latest tournament (seed stage, best candidate)
    category_table_slide(prs,
        "Per-Category Scores — Best Optimized Candidate (Val Set)",
        rows=[
            {
                "label": "Seed Baseline (MIDAS-style)",
                "total": 0.823,
                "scores": {"A1": 0.833, "A2": 0.956, "B1": 0.48, "B2": 0.12, "B3": 0.787, "B4": 0.933, "B5a": 0.893, "B5b": 0.771},
                "highlight": False
            },
            {
                "label": "Seed Candidate 1 (GPT-5.2 gen.)",
                "total": 0.862,
                "scores": {"A1": 1.0, "A2": 1.0, "B1": 0.6, "B2": 0.3, "B3": 0.64, "B4": 0.6, "B5a": 1.0, "B5b": 0.686},
                "highlight": False
            },
            {
                "label": "Bootstrap (n16_d3b3, from baseline)",
                "total": 0.843,
                "scores": {"A1": 0.775, "A2": 1.0, "B1": 0.8, "B2": 0.3, "B3": 0.72, "B4": 1.0, "B5a": 1.0, "B5b": 0.637},
                "highlight": False
            },
            {
                "label": "MIPROv2-Opt DCA-Light (Best)",
                "total": 0.869,
                "scores": {"A1": 1.0, "A2": 1.0, "B1": 0.8, "B2": 0.3, "B3": 0.76, "B4": 1.0, "B5a": 1.0, "B5b": 0.700},
                "highlight": True
            },
        ],
        subtitle="B2 (crossed-out) remains a hard case across all prompts — rare category"
    )

    bullet_slide(prs, "Key Findings", [
        ("Optimized prompt scores 0.869 total on val & test:", "Consistent generalization — no overfitting to train set."),
        ("91% improvement in total score over MIDAS v1:", "From 0.455 → 0.869 using the same gpt-4.1-mini model."),
        ("183% improvement in change description quality:", "0.278 → 0.786 — the hardest metric to improve."),
        ("Token efficiency maintained:", "Optimized prompt uses ~2,800 tokens avg vs ~3,073 for MIDAS v1 — "
                                          "better quality at lower cost."),
        ("B2 (crossed-out) is a persistent weak spot:", "Score ~0.3 across all variants — very few training examples, "
                                                          "rare in production data."),
        ("Instructions-only prompts beat demo-augmented:", "Adding demos degraded performance in most cases — "
                                                             "likely due to image context length explosion."),
    ], subtitle="")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 7: Next Steps
    # ══════════════════════════════════════════════════════════════════════════
    section_divider(prs, "Section 7", "Next Steps & Open Questions")

    bullet_slide(prs, "Next Steps", [
        ("Production deployment:", "Integrate best optimized prompt into DCA production pipeline "
                                    "and monitor live metrics."),
        ("B2 improvement:", "Augment training data with more crossed-out/redacted image pairs "
                              "to improve the weakest category (currently ~0.3)."),
        ("Larger student model:", "Run optimized prompt with gpt-4.1 (full) — early results show "
                                   "~0.87+ total score; trade-off vs. cost/latency."),
        ("GEPA optimization:", "Run full GEPA stage with GPT-5.2 reflections — preliminary results "
                                "promising for B5b descriptive quality."),
        ("Repeatability study:", "10-run repeatability analysis in progress — measuring variance "
                                   "across seeds to quantify prompt stability."),
        ("Human evaluation:", "Domain expert review of optimized outputs on a sample of 50 pairs "
                               "to validate automated metric alignment."),
    ], subtitle="")

    # Closing
    title_card(prs,
               "Thank You",
               "Questions?  |  AIQ-212  |  28 May 2026",
               bg_color=BOSCH_DARK)

    prs.save(DST)
    print(f"Saved: {DST}")
    print(f"Total slides: {len(prs.slides)}")


if __name__ == "__main__":
    build()

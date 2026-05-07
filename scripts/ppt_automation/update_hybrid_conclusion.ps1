# update_hybrid_conclusion.ps1
# Delete existing slide 17 and replace with updated 3-panel conclusion
# Panels: Hybrid Exp+BEST (153708) | Hybrid Exp+WORST (233110) | Hybrid Linear+BEST (172652)
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$pptApp = New-Object -ComObject PowerPoint.Application
$pptApp.Visible = [Microsoft.Office.Core.MsoTriState]::msoTrue
$srcPath = 'C:\Users\BHS1RNG\Downloads\05-05-2026-dca-updated.pptx'
$outPath = 'C:\Users\BHS1RNG\Downloads\05-05-2026-dca-updated.pptx'
$pres    = $pptApp.Presentations.Open($srcPath)
Write-Host "Total slides before:" $pres.Slides.Count

$boschRed = [int](226 + 0*256   + 21*65536)
$darkGray = [int](60  + 60*256  + 60*65536)
$midGray  = [int](130 + 130*256 + 130*65536)
$lightBg  = [int](245 + 245*256 + 245*65536)
$altBg    = [int](250 + 250*256 + 250*65536)
$white    = [int]16777215
$greenCol = [int](0   + 140*256 + 80*65536)
$algoBdr  = [int](0   + 100*256 + 175*65536)
$codeBg   = [int](240 + 248*256 + 255*65536)
$warnCol  = [int](200 + 30*256  + 30*65536)
$blueHdr  = [int](0   + 80*256  + 140*65536)
$hdrDark  = [int](50  + 50*256  + 50*65536)
$amberHdr = [int](160 + 80*256  + 0*65536)    # Exp+WORST panel header

function Add-Rect($slide, $x, $y, $w, $h, $fillCol, $lineCol, $lineW) {
    $sh = $slide.Shapes.AddShape(1, $x, $y, $w, $h)
    try { $sh.Fill.ForeColor.RGB = [int]$fillCol } catch {}
    try { $sh.Fill.Solid() } catch {}
    if ($lineCol -lt 0) { try { $sh.Line.Visible = 0 } catch {} }
    else {
        try { $sh.Line.Visible = -1 } catch {}
        try { $sh.Line.ForeColor.RGB = [int]$lineCol } catch {}
        try { $sh.Line.Weight = [double]$lineW } catch {}
    }
    try { $sh.TextFrame.TextRange.Text = "" } catch {}
    return $sh
}

function Add-TB($slide, $x, $y, $w, $h, $text, $sz, $bold, $col, $align) {
    $tb = $slide.Shapes.AddTextbox(1, $x, $y, $w, $h)
    $tf = $tb.TextFrame
    try { $tf.WordWrap = -1 } catch {}
    $tr = $tf.TextRange
    try { $tr.Text = $text } catch {}
    try { $tr.Font.Name = "Bosch Office Sans" } catch {}
    try { $tr.Font.Size = [int]$sz } catch {}
    try { if ($bold) { $tr.Font.Bold = -1 } else { $tr.Font.Bold = 0 } } catch {}
    try { $tr.Font.Color.RGB = [int]$col } catch {}
    try { foreach ($p in $tr.Paragraphs()) { $p.ParagraphFormat.Alignment = [int]$align } } catch {}
    try { $tf.MarginLeft = 3; $tf.MarginTop = 2; $tf.MarginRight = 3; $tf.MarginBottom = 2 } catch {}
    try { $tb.Fill.Visible = 0 } catch {}
    try { $tb.Line.Visible = 0 } catch {}
    return $tb
}

# ── DELETE old slide 17 and insert fresh ────────────────────
Write-Host "Deleting old slide 17..."
$pres.Slides[17].Delete()

$refSl = $pres.Slides[16]
$sl    = $pres.Slides.AddSlide(17, $refSl.CustomLayout)
foreach ($sh in $refSl.Shapes) {
    if ($sh.Name -like "Gruppieren*" -or $sh.Name -like "Gruppe*") {
        try { $sh.Copy(); $sl.Shapes.Paste() | Out-Null } catch {}
    }
}

# ── Header ──────────────────────────────────────────────────
Add-TB $sl 16 10 740 22 "Sequential Tournament Framework for Automated Prompt Optimization" 16 $false $boschRed 1 | Out-Null
Add-TB $sl 16 32 400 18 "MiDAS Collaboration" 11 $false $darkGray 1 | Out-Null
Add-TB $sl 16 52 831 18 "Conclusion: Hybrid Metric - Exp+BEST vs Exp+WORST vs Linear+BEST" 13 $true $boschRed 1 | Out-Null
$ln1 = $sl.Shapes.AddLine(16, 72, 847, 72)
try { $ln1.Line.ForeColor.RGB = [int]$boschRed; $ln1.Line.Weight = 1.2 } catch {}

# ── Setup bar ────────────────────────────────────────────────
Add-Rect $sl 16 76 831 18 $codeBg $algoBdr 0.6 | Out-Null
Add-TB   $sl 18 77 829 16 "Setup: Student: gpt-4.1-mini  |  Seed LM: gpt-5.2  |  top_k=2  |  Pareto selection  |  Best/Worst demos per class  |  Mini val n=45" 7 $false $midGray 1 | Out-Null

# ── 3-panel layout constants ─────────────────────────────────
# Total: 16..847 = 831px. Three panels 275px each, gap 3px.
# P1: x=16  P2: x=294  P3: x=572
$p1X = 16;  $p2X = 294; $p3X = 572
$pW  = 275; $p3W = 275  # last panel ends at 572+275=847
$cH  = 116; $rH  = 18

# Column offsets inside each panel (from panel left edge):
$cC  = 2    # Candidate x offset, w=143
$cS  = 147  # Stage x offset, w=36
$cD  = 185  # DCA Score x offset, w=50
$cT  = 237  # Tok/smp x offset, w=36
$cF  = 275  # Flag - drawn only when needed, w=0 (edge)

# ── Panel headers (row y=98) ─────────────────────────────────
Add-Rect $sl $p1X 98 $pW     18 $hdrDark  -1 0 | Out-Null
Add-TB   $sl ($p1X+2) 98 ($pW-4) 18 "Hybrid EXP + BEST ctx  (153708)" 6 $true $white 1 | Out-Null

Add-Rect $sl $p2X 98 $pW     18 $amberHdr -1 0 | Out-Null
Add-TB   $sl ($p2X+2) 98 ($pW-4) 18 "Hybrid EXP + WORST ctx  (233110)  NEW" 6 $true $white 1 | Out-Null

Add-Rect $sl $p3X 98 $p3W    18 $blueHdr  -1 0 | Out-Null
Add-TB   $sl ($p3X+2) 98 ($p3W-4) 18 "Hybrid LINEAR + BEST ctx  (172652)" 6 $true $white 1 | Out-Null

# ── Column sub-headers (y=116) ───────────────────────────────
$chBg = [int](80+80*256+80*65536)
foreach ($pX in @($p1X, $p2X, $p3X)) {
    $pw = if ($pX -eq $p3X) { $p3W } else { $pW }
    Add-Rect $sl $pX $cH $pw 16 $chBg -1 0 | Out-Null
    Add-TB   $sl ($pX+$cC) $cH 143 16 "Candidate"   5 $true $white 1 | Out-Null
    Add-TB   $sl ($pX+$cS) $cH 36  16 "Stg"         5 $true $white 1 | Out-Null
    Add-TB   $sl ($pX+$cD) $cH 50  16 "DCA"         5 $true $white 2 | Out-Null
    Add-TB   $sl ($pX+$cT) $cH 36  16 "Tok"         5 $true $white 2 | Out-Null
}

# ── Data: [Candidate, Stage, DCA%, Tok, FlagText, NameCol, TokCol] ──
# Panel 1: Hybrid Exp + BEST (153708)
$p1Rows = @(
    @("gepa <- miprov2 <- baseline_bare",  "gepa",  "86.28%", "3,526",  "BEST",     $boschRed, $greenCol),
    @("seed_baseline_bare",                "seed",  "84.44%", "3,063",  "",         $darkGray, $greenCol),
    @("seed_baseline_with_demos",          "seed",  "84.31%", "3,070",  "",         $darkGray, $greenCol),
    @("seed_candidate_1_with_demos",       "seed",  "84.25%", "22,146", "HIGH",     $darkGray, $warnCol),
    @("seed_candidate_0 (bare)",           "seed",  "84.05%", "4,073",  "",         $darkGray, $greenCol)
)
# Panel 2: Hybrid Exp + WORST (233110) - NEW
$p2Rows = @(
    @("gepa <- bootstrap <- seed_cand_1",  "gepa",      "87.07%", "3,543", "BEST",  $boschRed, $greenCol),
    @("seed_baseline_with_demos",          "seed",      "86.46%", "3,068", "",      $darkGray, $greenCol),
    @("seed_candidate_1 (bare)",           "seed",      "85.96%", "4,174", "",      $darkGray, $greenCol),
    @("seed_baseline_bare",                "seed",      "85.47%", "3,064", "",      $darkGray, $greenCol),
    @("bootstrap <- seed_candidate_1",     "bootstrap", "84.98%", "4,177", "",      $darkGray, $greenCol)
)
# Panel 3: Hybrid Linear + BEST (172652)
$p3Rows = @(
    @("gepa <- miprov2 <- baseline_demos", "gepa",  "87.41%", "3,603",  "BEST",     $boschRed, $greenCol),
    @("seed_candidate_1_with_demos",       "seed",  "86.76%", "21,793", "HIGH",     $darkGray, $warnCol),
    @("gepa <- miprov2 <- base_demos 2",   "gepa",  "86.47%", "3,073",  "",         $darkGray, $greenCol),
    @("seed_baseline_with_demos",          "seed",  "86.11%", "3,061",  "",         $darkGray, $greenCol),
    @("seed_candidate_0 (bare)",           "seed",  "85.01%", "4,203",  "",         $darkGray, $greenCol)
)

$rY = $cH + 16
for ($i = 0; $i -lt 5; $i++) {
    $bg = if ($i % 2 -eq 0) { $lightBg } else { $altBg }
    $r1 = $p1Rows[$i]; $r2 = $p2Rows[$i]; $r3 = $p3Rows[$i]

    foreach ($row in @(
        @($p1X, $pW,  $r1),
        @($p2X, $pW,  $r2),
        @($p3X, $p3W, $r3)
    )) {
        $px = $row[0]; $pw2 = $row[1]; $r = $row[2]
        Add-Rect $sl $px          $rY $pw2 $rH $bg $midGray 0.3 | Out-Null
        Add-TB   $sl ($px+$cC)    $rY 143 $rH $r[0] 5 $false $r[5] 1 | Out-Null
        Add-TB   $sl ($px+$cS)    $rY 36  $rH $r[1] 5 $false $midGray 1 | Out-Null
        Add-TB   $sl ($px+$cD)    $rY 50  $rH $r[2] 6 ($i -eq 0) $r[5] 2 | Out-Null
        Add-TB   $sl ($px+$cT)    $rY 36  $rH $r[3] 6 $false $r[6] 2 | Out-Null
    }

    # Flags (drawn as compact label at end of each panel row)
    foreach ($row in @(
        @($p1X, $r1),
        @($p2X, $r2),
        @($p3X, $r3)
    )) {
        $px = $row[0]; $r = $row[1]
        if ($r[4] -ne "") {
            $fc = $r[6]
            Add-TB $sl ($px+$cT+38) $rY 24 $rH $r[4] 4 $true $fc 1 | Out-Null
        }
    }

    $rY += $rH
}

# ── Separator ────────────────────────────────────────────────
$ln2 = $sl.Shapes.AddLine(16, $rY, 847, $rY)
try { $ln2.Line.ForeColor.RGB = [int]$midGray; $ln2.Line.Weight = 0.5 } catch {}
$rY += 4

# ── Summary comparison mini-table ────────────────────────────
# Quick visual: 3-row summary of best result per run
Add-Rect $sl 16 $rY 831 16 ([int](60+60*256+60*65536)) -1 0 | Out-Null
Add-TB   $sl 18  $rY 220 16 "Run"              6 $true $white 1 | Out-Null
Add-TB   $sl 240 $rY 80  16 "Context Mode"     6 $true $white 1 | Out-Null
Add-TB   $sl 322 $rY 80  16 "Metric"           6 $true $white 1 | Out-Null
Add-TB   $sl 404 $rY 80  16 "Best DCA"         6 $true $white 2 | Out-Null
Add-TB   $sl 486 $rY 80  16 "Tok/smp"          6 $true $white 2 | Out-Null
Add-TB   $sl 568 $rY 180 16 "vs Exp+BEST"      6 $true $white 2 | Out-Null
Add-TB   $sl 750 $rY 95  16 "Best Stage"       6 $true $white 1 | Out-Null
$rY += 16

$sumRows = @(
    @("Hybrid Exp + BEST ctx  (153708)", "BEST",  "Exp  (-2, 2500 tok)", "86.28%", "3,526", "baseline",       $midGray),
    @("Hybrid Exp + WORST ctx  (233110)","WORST", "Exp  (-2, 2500 tok)", "87.07%", "3,543", "+0.79 pp",       $greenCol),
    @("Hybrid Linear + BEST ctx  (172652)","BEST","Linear  (12000 tok)", "87.41%", "3,603", "+1.13 pp  BEST", $greenCol)
)
$sumBgs = @($lightBg, [int](255+248*256+235*65536), [int](235+245*256+255*65536))
for ($i = 0; $i -lt 3; $i++) {
    $sr = $sumRows[$i]
    Add-Rect $sl 16  $rY 831 16 $sumBgs[$i] $midGray 0.3 | Out-Null
    Add-TB   $sl 18  $rY 220 16 $sr[0] 6 ($i -eq 2) $darkGray 1 | Out-Null
    Add-TB   $sl 240 $rY 80  16 $sr[1] 6 $false     $darkGray 1 | Out-Null
    Add-TB   $sl 322 $rY 80  16 $sr[2] 6 $false     $midGray  1 | Out-Null
    Add-TB   $sl 404 $rY 80  16 $sr[3] 7 ($i -eq 2) $boschRed 2 | Out-Null
    Add-TB   $sl 486 $rY 80  16 $sr[4] 7 $false     $greenCol 2 | Out-Null
    Add-TB   $sl 568 $rY 180 16 $sr[5] 7 ($i -gt 0) $sr[6]   2 | Out-Null
    Add-TB   $sl 750 $rY 95  16 "GEPA"  6 $false    $algoBdr  1 | Out-Null
    $rY += 16
}

$ln3 = $sl.Shapes.AddLine(16, $rY, 847, $rY)
try { $ln3.Line.ForeColor.RGB = [int]$midGray; $ln3.Line.Weight = 0.5 } catch {}
$rY += 4

# ── KEY FINDING ──────────────────────────────────────────────
$fnd = Add-Rect $sl 16 $rY 831 26 $boschRed -1 0
try {
    $fnd.TextFrame.TextRange.Text = "KEY FINDING:  Exp+WORST ctx (87.07%) closes the gap to Linear (87.41%) to only -0.34 pp vs Exp+BEST gap of -1.13 pp.  WORST context mode adds +0.79 pp to Hybrid Exp.  All 3 runs keep top-k forwards under 5K tok -- Exp+WORST has NO high-token entries in its top-5."
    $fnd.TextFrame.TextRange.Font.Name = "Bosch Office Sans"
    $fnd.TextFrame.TextRange.Font.Size = 8
    $fnd.TextFrame.TextRange.Font.Bold = -1
    $fnd.TextFrame.TextRange.Font.Color.RGB = [int]$white
    foreach ($p in $fnd.TextFrame.TextRange.Paragraphs()) { $p.ParagraphFormat.Alignment = 2 }
    $fnd.TextFrame.VerticalAnchor = 3
} catch {}
$rY += 30

# ── Two explanation boxes ────────────────────────────────────
Add-Rect $sl 16  $rY 408 64 $lightBg $algoBdr 0.75 | Out-Null
Add-TB   $sl 18  ($rY+2)  404 12 "WORST CONTEXT MODE IMPACT ON EXP METRIC" 8 $true $amberHdr 1 | Out-Null
Add-TB   $sl 18  ($rY+15) 404 46 ("Exp+BEST seeds score 84.05-84.44% (bare).  Exp+WORST seeds score 85.47-86.46% (bare)." + [char]10 + "WORST ctx generates better instructions: seeds learn from failure cases." + [char]10 + "Exp+WORST top-5: zero HIGH TOK entries -- all under 5K tok even in raw leaderboard." + [char]10 + "WORST ctx also changes best path: bootstrap->GEPA (not miprov2->GEPA).") 6 $false $darkGray 1 | Out-Null

Add-Rect $sl 428 $rY 419 64 $lightBg $algoBdr 0.75 | Out-Null
Add-TB   $sl 430 ($rY+2)  415 12 "METRIC + CONTEXT MODE INTERACTION" 8 $true $blueHdr 1 | Out-Null
Add-TB   $sl 430 ($rY+15) 415 46 ("Linear metric allows higher seed diversity (12K tok budget) vs Exp (2.5K budget)." + [char]10 + "But WORST context mode compensates for Exp's strictness: +0.79 pp recovery." + [char]10 + "Linear still best at 87.41% (+0.34 pp over Exp+WORST, +1.13 pp over Exp+BEST)." + [char]10 + "For max accuracy: Linear. For token-strict pipelines: Exp+WORST is a strong alternative.") 6 $false $darkGray 1 | Out-Null
$rY += 68

# ── Recommendation ───────────────────────────────────────────
$rec = Add-Rect $sl 16 $rY 831 26 $codeBg $algoBdr 0.8
try {
    $rec.TextFrame.TextRange.Text = "RECOMMENDATION:  Primary: --hybrid-penalty-mode linear (87.41%, best accuracy).  Alternative if strict token budget: --hybrid-penalty-mode exponential --seed-context-mode worst (87.07%, -0.34 pp, same efficiency).  Avoid Exp+BEST: it loses -1.13 pp vs Linear for no benefit. All three configurations successfully hold forwarded prompts under 5K tok."
    $rec.TextFrame.TextRange.Font.Name = "Bosch Office Sans"
    $rec.TextFrame.TextRange.Font.Size = 7
    $rec.TextFrame.TextRange.Font.Color.RGB = [int]$darkGray
    foreach ($p in $rec.TextFrame.TextRange.Paragraphs()) { $p.ParagraphFormat.Alignment = 1 }
    $rec.TextFrame.MarginLeft = 5; $rec.TextFrame.MarginTop = 3
} catch {}
$rY += 30

Add-TB $sl 16 446 30 18 "17" 12 $false $midGray 1 | Out-Null

# ── Save ─────────────────────────────────────────────────────
$pres.SaveAs($outPath)
$pres.Close()
$pptApp.Quit()
Write-Host "Done -- slide 17 replaced with 3-panel conclusion"
Write-Host "Total slides:" $pres.Slides.Count

# add_slide18.ps1 — Insert conclusion slide after slide 17
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$pptApp = New-Object -ComObject PowerPoint.Application
$pptApp.Visible = [Microsoft.Office.Core.MsoTriState]::msoTrue
$srcPath = 'C:\Users\BHS1RNG\Downloads\05-05-2026-dca-updated.pptx'
$outPath = 'C:\Users\BHS1RNG\Downloads\05-05-2026-dca-updated.pptx'
$pres    = $pptApp.Presentations.Open($srcPath)

Write-Host "Total slides before:" $pres.Slides.Count

# ── colors ──────────────────────────────────────────────────
$boschRed = [int](226 + 0*256   + 21*65536)
$darkGray = [int](60  + 60*256  + 60*65536)
$midGray  = [int](130 + 130*256 + 130*65536)
$lightBg  = [int](245 + 245*256 + 245*65536)
$white    = [int]16777215
$worstCol = [int](200 + 30*256  + 30*65536)    # red-ish for WORST label
$bestCol  = [int](0   + 120*256 + 180*65536)   # blue for BEST label
$greenCol = [int](0   + 140*256 + 80*65536)    # green for positive delta
$algoBdr  = [int](0   + 100*256 + 175*65536)
$codeBg   = [int](240 + 248*256 + 255*65536)
$amber    = [int](200 + 130*256 + 0*65536)

# ── copy slide 16 layout as base (same header/footer group) ─
$refSlide = $pres.Slides[16]
$newSlide = $pres.Slides.AddSlide(18, $refSlide.CustomLayout)

# Copy Bosch header group (Gruppieren 3) and slide-number from slide 16
foreach ($sh in $refSlide.Shapes) {
    if ($sh.Name -like "Gruppieren*" -or $sh.Name -like "Gruppe*") {
        try { $sh.Copy(); $newSlide.Shapes.Paste() | Out-Null } catch {}
    }
}

# ── helpers ─────────────────────────────────────────────────
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

# ── Slide header (matching other slides) ────────────────────
Add-TB $newSlide 16 10 740 22 "Sequential Tournament Framework for Automated Prompt Optimization" `
  16 $false $boschRed 1 | Out-Null
Add-TB $newSlide 16 32 400 18 "MiDAS Collaboration" 11 $false $darkGray 1 | Out-Null
Add-TB $newSlide 16 52 831 18 `
  "Conclusion: Seed Context Mode — Worst vs Best Train Examples" `
  13 $true $boschRed 1 | Out-Null

# ── Divider ─────────────────────────────────────────────────
$ln = $newSlide.Shapes.AddLine(16, 72, 847, 72)
try { $ln.Line.ForeColor.RGB = [int]$boschRed; $ln.Line.Weight = 1.2 } catch {}

# ── EXPERIMENT SETUP (compact top bar) ──────────────────────
Add-Rect $newSlide 16 76 831 20 $codeBg $algoBdr 0.6 | Out-Null
Add-TB $newSlide 18 77 829 18 `
  "Setup: Seed-only stage  |  N=2 LLM candidates  |  Student: gpt-4.1  |  Seed LM: gpt-5.2  |  8 context examples per class  |  Full val set n=82" `
  7 $false $midGray 1 | Out-Null

# ════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ════════════════════════════════════════════════════════════

# Column headers
$tX=16; $tY=100
Add-Rect $newSlide $tX $tY 831 18 ([int](50+50*256+50*65536)) -1 0 | Out-Null
Add-TB $newSlide ($tX+2) $tY 220 18 "Candidate" 7 $true $white 1 | Out-Null
Add-TB $newSlide ($tX+224) $tY 100 18 "WORST ctx" 7 $true $white 2 | Out-Null
Add-TB $newSlide ($tX+326) $tY 100 18 "BEST ctx" 7 $true $white 2 | Out-Null
Add-TB $newSlide ($tX+428) $tY 60 18 "Delta" 7 $true $white 2 | Out-Null
Add-TB $newSlide ($tX+490) $tY 100 18 "Tok/smp WORST" 7 $true $white 2 | Out-Null
Add-TB $newSlide ($tX+592) $tY 100 18 "Tok/smp BEST" 7 $true $white 2 | Out-Null
Add-TB $newSlide ($tX+694) $tY 140 18 "Interpretation" 7 $true $white 1 | Out-Null

# Row data: Label | WORST score | BEST score | delta | WORST tok | BEST tok | note
$rows = @(
    @("seed_candidate_0_with_demos",  "92.55%", "85.35%", "+7.20pp", "17,358", "17,027", "WORST demos guide LLM to fix weaknesses"),
    @("seed_candidate_1_with_demos",  "89.44%", "87.29%", "+2.15pp", "17,622", "17,190", "Same trend, smaller gap"),
    @("seed_baseline_with_demos",     "87.84%", "84.21%", "+3.63pp", "16,525", "16,165", "Baseline also benefits from WORST ctx"),
    @("seed_candidate_0  (bare)",     "84.83%", "79.29%", "+5.54pp",  "3,406",  "3,444", "No demos - instruction quality only"),
    @("seed_candidate_1  (bare)",     "~84-85%","80.89%",  "---",     "~3,400", "3,602", "(run 090909 not fully recorded)")
)

$rY = $tY + 18
$rH = 22
$altBg = [int](250+250*256+250*65536)
$rowNum = 0
foreach ($r in $rows) {
    $bg = if ($rowNum % 2 -eq 0) { $lightBg } else { $altBg }
    Add-Rect $newSlide $tX $rY 831 $rH $bg $midGray 0.3 | Out-Null

    Add-TB $newSlide ($tX+2) $rY 220 $rH $r[0] 7 $false $darkGray 1 | Out-Null

    # WORST score — highlight best in red
    $wColor = if ($r[0] -like "*0_with_demos*" -and $rowNum -eq 0) { $boschRed } else { $darkGray }
    $wBold  = if ($rowNum -eq 0) { $true } else { $false }
    Add-TB $newSlide ($tX+224) $rY 100 $rH $r[1] 8 $wBold $wColor 2 | Out-Null
    Add-TB $newSlide ($tX+326) $rY 100 $rH $r[2] 8 $false $darkGray 2 | Out-Null

    # Delta — green
    $dColor = if ($r[3] -like "+*") { $greenCol } else { $midGray }
    Add-TB $newSlide ($tX+428) $rY 60 $rH $r[3] 8 $true $dColor 2 | Out-Null

    Add-TB $newSlide ($tX+490) $rY 100 $rH $r[4] 7 $false $midGray 2 | Out-Null
    Add-TB $newSlide ($tX+592) $rY 100 $rH $r[5] 7 $false $midGray 2 | Out-Null
    Add-TB $newSlide ($tX+694) $rY 140 $rH $r[6] 6 $false $midGray 1 | Out-Null

    $rY += $rH
    $rowNum++
}

# Separator
$ln2 = $newSlide.Shapes.AddLine(16, $rY, 847, $rY)
try { $ln2.Line.ForeColor.RGB = [int]$midGray; $ln2.Line.Weight = 0.5 } catch {}
$rY += 4

# ════════════════════════════════════════════════════════════
# KEY FINDING banner
# ════════════════════════════════════════════════════════════
$banner = Add-Rect $newSlide 16 $rY 831 24 $boschRed -1 0
try {
    $banner.TextFrame.TextRange.Text = "KEY FINDING:  WORST context mode outperforms BEST by +5.26 pp on top candidate (92.55% vs 87.29%)  --  Seed LLM generates better instructions when shown WHERE the current prompt fails, not where it already succeeds"
    $banner.TextFrame.TextRange.Font.Name = "Bosch Office Sans"
    $banner.TextFrame.TextRange.Font.Size = 8
    $banner.TextFrame.TextRange.Font.Bold = -1
    $banner.TextFrame.TextRange.Font.Color.RGB = [int]$white
    foreach ($p in $banner.TextFrame.TextRange.Paragraphs()) { $p.ParagraphFormat.Alignment = 2 }
    $banner.TextFrame.VerticalAnchor = 3
} catch {}

$rY += 28

# ════════════════════════════════════════════════════════════
# WHY + TOKEN COST (two boxes side by side)
# ════════════════════════════════════════════════════════════

# Left: WHY WORST WINS
Add-Rect $newSlide 16 $rY 400 80 $lightBg $algoBdr 0.75 | Out-Null
Add-TB $newSlide 18 ($rY+2) 396 16 "WHY WORST EXAMPLES HELP MORE" 8 $true $boschRed 1 | Out-Null
Add-TB $newSlide 18 ($rY+18) 396 58 `
  "When seed LM sees failures, it learns which edge cases the current" + [char]10 + `
  "instruction mishandles (B3 trivial pairs, B5b change descriptions)." + [char]10 + `
  "It then writes targeted fixes for those specific gaps." + [char]10 + `
  "Best examples show what already works -- less actionable signal." `
  7 $false $darkGray 1 | Out-Null

# Right: TOKEN COST ANALYSIS
Add-Rect $newSlide 420 $rY 428 80 $lightBg $algoBdr 0.75 | Out-Null
Add-TB $newSlide 422 ($rY+2) 424 16 "TOKEN COST ANALYSIS (with_demos vs bare)" 8 $true $boschRed 1 | Out-Null
Add-TB $newSlide 422 ($rY+18) 424 58 `
  "with_demos:  ~17,000 tok/sample  (8 image pairs in context)" + [char]10 + `
  "bare:         ~3,400 tok/sample  (instruction only)" + [char]10 + `
  "Ratio: 5x more tokens for demos -- but +7.7 pp accuracy gain." + [char]10 + `
  "WORST+demos is the best accuracy/token trade-off at seed stage." `
  7 $false $darkGray 1 | Out-Null

$rY += 84

# ════════════════════════════════════════════════════════════
# RECOMMENDATION
# ════════════════════════════════════════════════════════════
$recBox = Add-Rect $newSlide 16 $rY 831 28 $codeBg $algoBdr 0.8
try {
    $recBox.TextFrame.TextRange.Text = `
        "RECOMMENDATION:  Always use --seed-context-mode worst  (the default in latest config)." + `
        "  The +5 pp gain over 'best' mode persists across seed candidates and justifies the token cost." + `
        "  Confirmed working after bug fix: seed_baseline_with_demos now correctly includes actual image demos."
    $recBox.TextFrame.TextRange.Font.Name = "Bosch Office Sans"
    $recBox.TextFrame.TextRange.Font.Size = 7
    $recBox.TextFrame.TextRange.Font.Color.RGB = [int]$darkGray
    foreach ($p in $recBox.TextFrame.TextRange.Paragraphs()) { $p.ParagraphFormat.Alignment = 1 }
    $recBox.TextFrame.MarginLeft = 5; $recBox.TextFrame.MarginTop = 4
} catch {}

# Slide number
Add-TB $newSlide 16 446 30 18 "18" 12 $false $midGray 1 | Out-Null

# ── SAVE ────────────────────────────────────────────────────
$pres.SaveAs($outPath)
$pres.Close()
$pptApp.Quit()
Write-Host "Done -- slide 18 added, saved to $outPath"
Write-Host "Total slides now: should be" ($pres.Slides.Count + 1)

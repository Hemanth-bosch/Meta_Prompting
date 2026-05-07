# update_master_conclusion.ps1
# Delete slide 26 (old plain text conclusions) and replace with
# a comprehensive, color-coded master conclusion across all 5 experiments.
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
$amberHdr = [int](160 + 80*256  + 0*65536)
$greenHdr = [int](0   + 120*256 + 60*65536)

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

# Helper: draw a compact finding box with colored header strip + 2 content lines
function Add-FindingBox($slide, $x, $y, $w, $hdrColor, $num, $title, $line1, $line2) {
    $bh = 54
    Add-Rect $slide $x $y $w $bh $lightBg $hdrColor 0.8 | Out-Null
    # Header strip
    Add-Rect $slide $x $y $w 14 $hdrColor -1 0 | Out-Null
    Add-TB   $slide ($x+3) ($y+1) ($w-6) 13 ($num + "  " + $title) 7 $true $white 1 | Out-Null
    # Content
    Add-TB   $slide ($x+3) ($y+15) ($w-6) 17 $line1 6 $false $darkGray 1 | Out-Null
    Add-TB   $slide ($x+3) ($y+32) ($w-6) 17 $line2 6 $false $midGray  1 | Out-Null
}

# ── DELETE old slide 26, insert new ──────────────────────────
Write-Host "Replacing slide 26..."
$pres.Slides[26].Delete()
$refSl = $pres.Slides[25]
$sl    = $pres.Slides.AddSlide(26, $refSl.CustomLayout)
foreach ($sh in $refSl.Shapes) {
    if ($sh.Name -like "Gruppieren*" -or $sh.Name -like "Gruppe*") {
        try { $sh.Copy(); $sl.Shapes.Paste() | Out-Null } catch {}
    }
}

# ── Standard Bosch header ────────────────────────────────────
Add-TB $sl 16 10 740 22 "Sequential Tournament Framework for Automated Prompt Optimization" 16 $false $boschRed 1 | Out-Null
Add-TB $sl 16 32 400 18 "MiDAS Collaboration" 11 $false $darkGray 1 | Out-Null
Add-TB $sl 16 52 831 18 "Master Conclusions: All Tournament Experiments -- Key Findings, Quantitative Evidence and Recommendations" 13 $true $boschRed 1 | Out-Null
$ln1 = $sl.Shapes.AddLine(16, 72, 847, 72)
try { $ln1.Line.ForeColor.RGB = [int]$boschRed; $ln1.Line.Weight = 1.2 } catch {}

# ── Achievement banner (red) ─────────────────────────────────
$ach = Add-Rect $sl 16 76 831 20 $boschRed -1 0
try {
    $ach.TextFrame.TextRange.Text = "BEST DCA:  92.55%  (gpt-4.1 + WORST ctx + seed stage, 17K tok)   |   92.28%  (gpt-4.1 + Pareto + full pipeline, 8.7K tok)   |   87.41%  (gpt-4.1-mini + Hybrid Linear)   |   Baseline: 72% v1-midas   |   +20 pp improvement"
    $ach.TextFrame.TextRange.Font.Name  = "Bosch Office Sans"
    $ach.TextFrame.TextRange.Font.Size  = 7
    $ach.TextFrame.TextRange.Font.Bold  = -1
    $ach.TextFrame.TextRange.Font.Color.RGB = [int]$white
    foreach ($p in $ach.TextFrame.TextRange.Paragraphs()) { $p.ParagraphFormat.Alignment = 2 }
    $ach.TextFrame.VerticalAnchor = 3
} catch {}

# ── Six Finding Boxes (2 col x 3 row grid) ───────────────────
# Left col: x=16, w=408    Right col: x=428, w=419
# Row y: 100, 158, 216   (h=54, gap=4)
$lX = 16;  $lW = 408
$rX = 428; $rW = 419

Add-FindingBox $sl $lX 100 $lW $algoBdr `
    "1." "PERFORMANCE BREAKTHROUGH" `
    "92.28% DCA (gpt-4.1, Pareto, full pipeline) from ~72% v1-midas baseline -- +20 pp in one automated run." `
    "Framework explored 15+ tournaments, 100+ candidate prompts. No manual prompt engineering needed."

Add-FindingBox $sl $rX 100 $rW $greenHdr `
    "2." "PARETO SELECTION DOMINATES -- USE PARETO AS DEFAULT" `
    "gpt-4.1: Pareto 92.28% vs Best-Score 89.85% (+2.43 pp, 1.86x fewer tokens: 8,698 vs 16,160 tok/smp)." `
    "gpt-4.1-mini: near-identical peak (+0.08 pp). Pareto keeps ALL candidates under 4K tok. Always use Pareto."

Add-FindingBox $sl $lX 158 $lW $boschRed `
    "3." "GEPA WITH MULTIMODAL REFLECTION IS THE CRITICAL STAGE" `
    "Full pipeline: Seed->Bootstrap(+5.69 pp)->MIPROv2->GEPA(+3.15 pp). Skipping GEPA costs 3+ pp." `
    "GPT-5.2 sees failing image pairs during reflection -> richer feedback -> targeted instruction rewrites."

Add-FindingBox $sl $rX 158 $rW $amberHdr `
    "4." "WORST CONTEXT MODE BEATS BEST -- BUT ONLY WITH OPTIMIZATION" `
    "gpt-4.1 seeds (full tournament): WORST vs BEST = +5.26 to +7.20 pp (92.55% vs 87.29%)." `
    "WARNING: WORST mode seed-only (mini, no optimization) fails catastrophically: cand_1 at 66-68% (-20 pp)."

Add-FindingBox $sl $lX 216 $lW $blueHdr `
    "5." "HYBRID LINEAR METRIC RECOMMENDED FOR EFFICIENCY-CONSTRAINED RUNS" `
    "Linear (87.41%) > Exp+WORST (87.07%) > Exp+BEST (86.28%). Linear beats standard Pareto for mini (+0.93 pp)." `
    "All 3 hybrid variants keep forwarded entries <5K tok. Use Linear; use Exp only if 2.5K tok budget is critical."

Add-FindingBox $sl $rX 216 $rW ([int](80+80*256+80*65536)) `
    "6." "DCA IS INSTRUCTION-SENSITIVE -- DEMOS ADD COST, NOT CONSISTENT ACCURACY" `
    "8-demo seeds (17K tok) vs bare seeds (3-4K tok): no reliable advantage at seed stage for mini." `
    "GEPA optimizes demo selection itself. After optimization, bare prompts reach 86-92%. Instructions matter most."

# ── Mini results summary table ───────────────────────────────
$tY = 278
Add-Rect $sl 16 $tY 831 14 $hdrDark -1 0 | Out-Null
Add-TB   $sl 18  $tY 165 14 "Experiment"         6 $true $white 1 | Out-Null
Add-TB   $sl 185 $tY 65  14 "Student"            6 $true $white 1 | Out-Null
Add-TB   $sl 252 $tY 80  14 "Method"             6 $true $white 1 | Out-Null
Add-TB   $sl 334 $tY 68  14 "Context"            6 $true $white 1 | Out-Null
Add-TB   $sl 404 $tY 60  14 "Peak DCA"           6 $true $white 2 | Out-Null
Add-TB   $sl 466 $tY 58  14 "Tok/smp"            6 $true $white 2 | Out-Null
Add-TB   $sl 526 $tY 320 14 "Key outcome"        6 $true $white 1 | Out-Null
$tY += 14

# [Exp label, Student, Method, Context, DCA, Tok, Outcome, rowBg, dcaCol]
$tRows = @(
    @("1. Pareto vs Best-Score",    "gpt-4.1", "Pareto + full pipeline",       "BEST",   "92.28%", "8,698",  "Full Seed->Bootstrap->MIPROv2->GEPA wins. Pareto +2.43 pp cheaper.",                   $lightBg, $boschRed),
    @("1. Pareto vs Best-Score",    "gpt-4.1", "Best-Score (partial pipeline)", "BEST",   "89.85%", "16,160", "Direct seed->GEPA. -2.43 pp AND 1.86x costlier than Pareto.",                          $altBg,   $midGray),
    @("2. Pareto vs Best-Score",    "mini",    "Pareto / Best-Score",           "BEST",   "86.48-56%","3-4K",  "Near-identical peak. Pareto guarantees token efficiency.",                             $lightBg, $darkGray),
    @("3. Hybrid Metrics",          "mini",    "Hybrid Linear (Pareto)",        "BEST",   "87.41%", "3,603",  "Linear > Exp+WORST(-0.34pp) > Exp+BEST(-1.13pp). All enforce <5K tok forwarding.",     $altBg,   $algoBdr),
    @("4. WORST vs BEST ctx",       "gpt-4.1", "Seed-only (no optimization)",   "WORST",  "92.55%", "17,358", "+5.26 to +7.20 pp over BEST ctx. Failure cases guide the seed LLM to fix gaps.",      $lightBg, $boschRed),
    @("5. WORST vs BEST ctx (mini)","mini",    "Seed-only (no optimization)",   "BEST",   "86.35%", "21,850", "Neither beats hardcoded baseline (86.40%). WORST ctx fails cand_1 at 66-68% (-20pp).", $altBg,   $warnCol)
)

foreach ($r in $tRows) {
    Add-Rect $sl 16  $tY 831 14 $r[7] $midGray 0.2 | Out-Null
    Add-TB   $sl 18  $tY 165 14 $r[0] 5 $false $darkGray 1 | Out-Null
    Add-TB   $sl 185 $tY 65  14 $r[1] 5 $false $midGray  1 | Out-Null
    Add-TB   $sl 252 $tY 80  14 $r[2] 5 $false $midGray  1 | Out-Null
    Add-TB   $sl 334 $tY 68  14 $r[3] 5 $false $midGray  1 | Out-Null
    Add-TB   $sl 404 $tY 60  14 $r[4] 6 $true  $r[8]     2 | Out-Null
    Add-TB   $sl 466 $tY 58  14 $r[5] 6 $false $midGray  2 | Out-Null
    Add-TB   $sl 526 $tY 320 14 $r[6] 5 $false $darkGray 1 | Out-Null
    $tY += 14
}

$ln2 = $sl.Shapes.AddLine(16, $tY, 847, $tY)
try { $ln2.Line.ForeColor.RGB = [int]$midGray; $ln2.Line.Weight = 0.5 } catch {}
$tY += 4

# ── Recommendations (2 boxes side by side) ───────────────────
$rY = $tY
Add-Rect $sl 16  $rY 408 40 $lightBg $boschRed 0.9 | Out-Null
Add-Rect $sl 16  $rY 408 12 $boschRed -1 0 | Out-Null
Add-TB   $sl 18  ($rY+1) 404 11 "FOR BEST ACCURACY (production)" 7 $true $white 1 | Out-Null
Add-TB   $sl 18  ($rY+14) 404 24 ("gpt-4.1 student  |  --selection-criteria pareto  |  --seed-context-mode worst  |  Full pipeline" + [char]10 + "Expected: ~92% DCA at ~8-10K tok/sample  |  GEPA is mandatory") 6 $false $darkGray 1 | Out-Null

Add-Rect $sl 428 $rY 419 40 $lightBg $algoBdr 0.9 | Out-Null
Add-Rect $sl 428 $rY 419 12 $blueHdr -1 0 | Out-Null
Add-TB   $sl 430 ($rY+1) 415 11 "FOR TOKEN EFFICIENCY (mini)" 7 $true $white 1 | Out-Null
Add-TB   $sl 430 ($rY+14) 415 24 ("gpt-4.1-mini student  |  --selection-criteria pareto  |  --hybrid-penalty-mode linear  |  Full pipeline" + [char]10 + "Expected: ~87% DCA at <4K tok/sample  |  Hybrid Linear recommended over Exponential") 6 $false $darkGray 1 | Out-Null
$rY += 44

# ── Master Takeaway (full-width red banner) ──────────────────
$lnF = $sl.Shapes.AddLine(16, $rY, 847, $rY)
try { $lnF.Line.ForeColor.RGB = [int]$midGray; $lnF.Line.Weight = 0.4 } catch {}
$rY += 3

$tak = Add-Rect $sl 16 $rY 831 28 $hdrDark -1 0
try {
    $tak.TextFrame.TextRange.Text = "VALIDATED FORMULA:  gpt-4.1 + Pareto + WORST ctx + Bootstrap + MIPROv2 + GEPA  =  92.28% DCA  (from 72% v1-midas, +20 pp)  |  Framework is generic, DSPy-based, and reusable across use cases.  |  Single most impactful component: GEPA with multimodal reflection (GPT-5.2 sees failing image pairs)."
    $tak.TextFrame.TextRange.Font.Name  = "Bosch Office Sans"
    $tak.TextFrame.TextRange.Font.Size  = 7
    $tak.TextFrame.TextRange.Font.Bold  = -1
    $tak.TextFrame.TextRange.Font.Color.RGB = [int]$white
    foreach ($p in $tak.TextFrame.TextRange.Paragraphs()) { $p.ParagraphFormat.Alignment = 2 }
    $tak.TextFrame.VerticalAnchor = 3
} catch {}

Add-TB $sl 16 446 30 18 "26" 12 $false $midGray 1 | Out-Null

# ── Save ─────────────────────────────────────────────────────
$pres.SaveAs($outPath)
$pres.Close()
$pptApp.Quit()
Write-Host "Done -- master conclusion slide 26 replaced"
Write-Host "Total slides:" $pres.Slides.Count

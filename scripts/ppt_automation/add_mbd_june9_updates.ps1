# add_mbd_june9_updates.ps1
# Adds slides 21-25 to 2026-06-09-Updates_MBD-UTA.pptx
# Covers: metric redesign, M7 implementation, M8 new, tasks done/pending
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$pptPath = 'C:\Users\BHS1RNG\Downloads\2026-06-09-Updates_MBD-UTA.pptx'
$pptApp  = New-Object -ComObject PowerPoint.Application
$pptApp.Visible = [Microsoft.Office.Core.MsoTriState]::msoTrue
$pres = $pptApp.Presentations.Open($pptPath, $false, $false, $true)
Write-Host "Slides before:" $pres.Slides.Count

# colors
$boschRed = [int](226 + 0*256   + 21*65536)
$darkGray = [int](60  + 60*256  + 60*65536)
$midGray  = [int](130 + 130*256 + 130*65536)
$lightBg  = [int](245 + 245*256 + 245*65536)
$altBg    = [int](250 + 250*256 + 250*65536)
$white    = [int]16777215
$greenCol = [int](0   + 140*256 + 80*65536)
$algoBdr  = [int](0   + 100*256 + 175*65536)
$codeBg   = [int](240 + 248*256 + 255*65536)
$amber    = [int](200 + 130*256 + 0*65536)
$redDark  = [int](180 + 0*256   + 0*65536)
$indigoDk = [int](50  + 50*256  + 80*65536)
$greenBg  = [int](235 + 255*256 + 235*65536)
$greenDk  = [int](0   + 100*256 + 50*65536)

# helpers
function Add-Rect($slide, $x, $y, $w, $h, $fillCol, $lineCol, $lineW) {
    $sh = $slide.Shapes.AddShape(1, $x, $y, $w, $h)
    try { $sh.Fill.ForeColor.RGB = [int]$fillCol; $sh.Fill.Solid() } catch {}
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
    try { $tb.Fill.Visible = 0; $tb.Line.Visible = 0 } catch {}
    return $tb
}

function Add-SolidBox($slide, $x, $y, $w, $h, $bgCol, $text, $sz, $bold, $textCol, $align) {
    $sh = $slide.Shapes.AddShape(1, $x, $y, $w, $h)
    try { $sh.Fill.ForeColor.RGB = [int]$bgCol; $sh.Fill.Solid(); $sh.Line.Visible = 0 } catch {}
    try {
        $sh.TextFrame.TextRange.Text = $text
        $sh.TextFrame.TextRange.Font.Name = "Bosch Office Sans"
        $sh.TextFrame.TextRange.Font.Size = [int]$sz
        if ($bold) { $sh.TextFrame.TextRange.Font.Bold = -1 } else { $sh.TextFrame.TextRange.Font.Bold = 0 }
        $sh.TextFrame.TextRange.Font.Color.RGB = [int]$textCol
        foreach ($p in $sh.TextFrame.TextRange.Paragraphs()) { $p.ParagraphFormat.Alignment = [int]$align }
        $sh.TextFrame.VerticalAnchor = 3
        $sh.TextFrame.MarginLeft = 5; $sh.TextFrame.MarginTop = 3
    } catch {}
    return $sh
}

function Add-SlideHeader($slide, $pageTitle, $subtitle) {
    Add-TB $slide 16 10 740 22 "MBD-UTA: Unit Test Assistance" 16 $false $boschRed 1 | Out-Null
    Add-TB $slide 16 32 480 18 "Updates -- June 9, 2026  |  DSPy + GEPA Optimization" 11 $false $darkGray 1 | Out-Null
    if ($subtitle -ne "") {
        Add-TB $slide 500 32 346 18 $subtitle 9 $false $midGray 3 | Out-Null
    }
    Add-TB $slide 16 52 831 18 $pageTitle 13 $true $boschRed 1 | Out-Null
    $ln = $slide.Shapes.AddLine(16, 72, 847, 72)
    try { $ln.Line.ForeColor.RGB = [int]$boschRed; $ln.Line.Weight = 1.2 } catch {}
}

$refSlide = $pres.Slides[7]

function New-Slide($pos) {
    $s = $pres.Slides.AddSlide($pos, $refSlide.CustomLayout)
    foreach ($sh in $refSlide.Shapes) {
        if ($sh.Name -like "Gruppieren*" -or $sh.Name -like "Gruppe*") {
            try { $sh.Copy(); $s.Shapes.Paste() | Out-Null } catch {}
        }
    }
    return $s
}

# ============================================================
# SLIDE 21 -- section divider
# ============================================================
$s21 = New-Slide 21

Add-SolidBox $s21 0 0 863 115 $boschRed "" 9 $false $white 1 | Out-Null
Add-TB $s21 28 14 807 46 "MBD-UTA  |  Updates -- June 9, 2026" 26 $true $white 1 | Out-Null
Add-TB $s21 28 64 807 28 "Metric redesign  .  M7 implemented  .  M8 added  .  GEPA group 50/50 split" 13 $false $white 1 | Out-Null

$panW = 200; $panH = 110; $panY = 130; $gap = 10
$cards = @(
    @("M7 -- DLL Pass Rate", "Was: stub 0.5 (hardcoded)`n`nNow: inline 1/N scoring`ncaptured in Phase 4a`nretry loop in forward()"),
    @("M8 -- NEW Metric", "Functional Correctness`nWeight: 25%`n`nSequential DLL run on`nfinal post-processed TCs"),
    @("Weight Rebalance", "M1: 25% -> 15%`nM2: 20% -> 15%`nM3: 10% ->  5%`nM8:  --- -> 25%`nTotal: 100%"),
    @("GEPA 50 / 50 Split", "testcase_generator`n  M1+M3+M4+M6 = 50%`n`nequation_solver_dll`n  M2+M7+M8    = 50%")
)

$px = 16
foreach ($c in $cards) {
    Add-Rect $s21 $px $panY $panW $panH $lightBg $algoBdr 0.75 | Out-Null
    Add-SolidBox $s21 $px $panY $panW 20 $boschRed $c[0] 9 $true $white 1 | Out-Null
    Add-TB $s21 ($px+4) ($panY+22) ($panW-8) ($panH-24) $c[1] 8 $false $darkGray 1 | Out-Null
    $px += $panW + $gap
}

Add-TB $s21 16 250 831 14 "Changes since June 2, 2026  |  evaluation/mbd_metric.py  +  extract_mbd_full.py  +  _verify_mbd.py  (52 checks passing)" 7 $false $midGray 1 | Out-Null
Add-TB $s21 16 446 30 18 "21" 12 $false $midGray 1 | Out-Null

# ============================================================
# SLIDE 22 -- metric weight table
# ============================================================
$s22 = New-Slide 22
Add-SlideHeader $s22 "Metric Weight Redesign -- 7 Components" "Before (June 2)  vs  After (June 9)"

$cols22 = @(115, 65, 65, 58, 108, 420)
$hdrs22 = @("Metric", "Before", "After", "Delta", "Type", "Change")

$tX=16; $tY=78
Add-SolidBox $s22 $tX $tY 831 19 ([int](50+50*256+50*65536)) "" 7 $true $white 1 | Out-Null
$cx = $tX
for ($i = 0; $i -lt 6; $i++) {
    Add-TB $s22 ($cx+3) $tY ($cols22[$i]-3) 19 $hdrs22[$i] 7 $true $white 1 | Out-Null
    $cx += $cols22[$i]
}

$rows22 = @(
    @("M1  I/O Classification",    "25%", "15%", "-10 pp", "Rule-based",    "Reduced -- less punitive, M8 DLL now carries functional truth"),
    @("M2  Signal Resolution",      "20%", "15%", " -5 pp", "String lookup", "Reduced -- balanced against new M8 weight"),
    @("M3  Structural Validity",    "10%", " 5%", " -5 pp", "JSON / regex",  "Reduced -- structure is implied by passing M7/M8 DLL checks"),
    @("M4  Test Coverage",          "25%", "25%", "    0  ", "Branch count",  "Unchanged -- branch coverage remains primary test-quality signal"),
    @("M6  Boundary Values",        " 5%", " 5%", "    0  ", "Min/max check", "Unchanged"),
    @("M7  DLL Pass Rate",          "15%", "10%", " -5 pp", "DLL inline 1/N","Reduced -- M8 now carries end-to-end DLL truth at 25%"),
    @("M8  Functional Correctness", " ---", "25%", "+25 pp", "DLL sequential","NEW -- verifies final post-processed test cases end-to-end through DLL")
)

$rY = $tY + 19; $rH = 21; $ri = 0
foreach ($r in $rows22) {
    $bg = if ($ri % 2 -eq 0) { $lightBg } else { $altBg }
    if ($r[0] -like "*M8*") { $bg = $greenBg }
    Add-Rect $s22 $tX $rY 831 $rH $bg $midGray 0.25 | Out-Null
    $cx = $tX
    for ($i = 0; $i -lt 6; $i++) {
        $bold = ($i -eq 0)
        $col  = $darkGray
        if ($i -eq 3) {
            if   ($r[3].Trim() -like "+*") { $col = $greenCol }
            elseif ($r[3].Trim() -like "-*") { $col = $redDark }
        }
        if ($r[0] -like "*M8*") { $bold = ($i -lt 4); if ($i -eq 3) { $col = $greenCol } }
        Add-TB $s22 ($cx+3) $rY ($cols22[$i]-3) $rH $r[$i] 7 $bold $col 1 | Out-Null
        $cx += $cols22[$i]
    }
    $rY += $rH; $ri++
}

$ln22 = $s22.Shapes.AddLine(16, ($rY+3), 847, ($rY+3))
try { $ln22.Line.ForeColor.RGB = $midGray; $ln22.Line.Weight = 0.4 } catch {}
Add-TB $s22 16 ($rY+7) 831 18 "GEPA group split (50/50):   testcase_generator  M1+M3+M4+M6 = 50%  (rescaled 0.30 / 0.10 / 0.50 / 0.10)   |   equation_solver_dll  M2+M7+M8 = 50%  (rescaled 0.30 / 0.20 / 0.50)" 7 $false $midGray 1 | Out-Null
Add-TB $s22 16 446 30 18 "22" 12 $false $midGray 1 | Out-Null

# ============================================================
# SLIDE 23 -- M7 stub -> inline 1/N
# ============================================================
$s23 = New-Slide 23
Add-SlideHeader $s23 "M7  DLL Pass Rate -- Stub  ->  Inline 1/N Scoring  (10%)" ""

Add-Rect $s23 16 78 406 115 ([int](255+240*256+240*65536)) $redDark 0.75 | Out-Null
Add-SolidBox $s23 16 78 406 19 $redDark "BEFORE  (June 2)" 9 $true $white 1 | Out-Null
Add-TB $s23 20 99 400 92 ("def _m7_dll_pass_rate(...):" + [char]10 + "    # DLL not wired up yet" + [char]10 + "    return 0.5, 'stub'" + [char]10 + "" + [char]10 + "Effect: 15% of total score = fixed 0.075." + [char]10 + "No DLL call. No equation verified." + [char]10 + "GEPA optimized against a constant.") 7 $false $darkGray 1 | Out-Null

Add-Rect $s23 430 78 433 115 $greenBg $greenCol 0.75 | Out-Null
Add-SolidBox $s23 430 78 433 19 $greenCol "AFTER  (June 9)" 9 $true $white 1 | Out-Null
Add-TB $s23 434 99 425 92 ("Inline score captured during Phase 4a retry loop." + [char]10 + "" + [char]10 + "Per equation:" + [char]10 + "  1st try passes  ->  1.0" + [char]10 + "  1 retry passes  ->  0.5" + [char]10 + "  retry fails     ->  0.0" + [char]10 + "" + [char]10 + "M7 = mean(per-equation scores)") 7 $false $darkGray 1 | Out-Null

Add-Rect $s23 16 198 831 80 $codeBg $algoBdr 0.7 | Out-Null
Add-TB $s23 18 200 400 16 "Implementation -- extract_mbd_full.py  forward()" 9 $true $boschRed 1 | Out-Null
Add-TB $s23 18 218 412 58 ("# initialise before equation loop" + [char]10 + "eq_m7_scores: list = []" + [char]10 + "" + [char]10 + "# before retry block -- capture first-try result" + [char]10 + "first_try_pass = (expected == actual)" + [char]10 + "" + [char]10 + "# after retry block" + [char]10 + "if first_try_pass:         eq_m7_scores.append(1.0)" + [char]10 + "elif expected == actual:   eq_m7_scores.append(0.5)" + [char]10 + "else:                      eq_m7_scores.append(0.0)") 7 $false $darkGray 1 | Out-Null
Add-TB $s23 440 218 404 58 ("Example -- 3 equations in one weakspot:" + [char]10 + "  xa  solved first try             1.0" + [char]10 + "  xb  solved after one retry       0.5" + [char]10 + "  xc  failed (both tries)          0.0" + [char]10 + "" + [char]10 + "M7 = (1.0 + 0.5 + 0.0) / 3 = 0.50" + [char]10 + "" + [char]10 + "Returned as m7_score in dspy.Prediction." + [char]10 + "Metric uses it if not None; else post-hoc fallback.") 7 $false $darkGray 1 | Out-Null

Add-SolidBox $s23 16 282 831 22 $boschRed "GEPA IMPACT:  M7 now contributes real signal (10% overall, rescaled to 20% in equation_solver_dll group).  The June 2 run optimized against a fixed 0.075 constant." 7 $true $white 2 | Out-Null

Add-TB $s23 16 446 30 18 "23" 12 $false $midGray 1 | Out-Null

# ============================================================
# SLIDE 24 -- M8 Functional Correctness
# ============================================================
$s24 = New-Slide 24
Add-SlideHeader $s24 "M8  Functional Correctness -- New Metric  (25%)" "Highest single weight component"

Add-Rect $s24 16 78 408 170 $lightBg $algoBdr 0.7 | Out-Null
Add-SolidBox $s24 16 78 408 19 $boschRed "HOW M8 RUNS EACH TEST CASE" 9 $true $white 1 | Out-Null
Add-TB $s24 20 100 402 146 ("Takes each final post-processed TC and executes it" + [char]10 + "sequentially through the compiled weakspot DLL." + [char]10 + "" + [char]10 + "1.  Precondition" + [char]10 + "      Set initial signal values." + [char]10 + "      Run DLL for 1 raster (warmup)." + [char]10 + "" + [char]10 + "2.  For each step:" + [char]10 + "      a) Apply Actions (set input signals)" + [char]10 + "      b) Wait N -> advance DLL N*raster ticks" + [char]10 + "      c) Compare DLL output vs Expected Results" + [char]10 + "" + [char]10 + "3.  Reset between TCs:  fc._init() call") 7 $false $darkGray 1 | Out-Null

Add-Rect $s24 432 78 415 170 $lightBg $algoBdr 0.7 | Out-Null
Add-SolidBox $s24 432 78 415 19 $boschRed "SCORING" 9 $true $white 1 | Out-Null
Add-TB $s24 436 100 409 146 ("Tolerance per step:" + [char]10 + "  tol = 1.0 + 0.01 * |expected|" + [char]10 + "" + [char]10 + "Step passes if  |actual - expected| <= tol" + [char]10 + "" + [char]10 + "TC score = passed_steps / total_steps" + [char]10 + "" + [char]10 + "M8 = mean(TC scores)  over all test cases" + [char]10 + "" + [char]10 + "DLL unavailable -> M8 = 0.5 (neutral)" + [char]10 + "" + [char]10 + "GEPA rescale: 50% of equation_solver_dll group" + [char]10 + "(M2+M7+M8 = 50% -> rescaled 0.30 / 0.20 / 0.50)") 7 $false $darkGray 1 | Out-Null

Add-SolidBox $s24 16 252 831 19 $indigoDk "M7 verifies intermediate equations during solving (1/N retry).     M8 verifies the final assembled test case end-to-end." 7 $false $white 2 | Out-Null

Add-Rect $s24 16 275 831 100 $codeBg $algoBdr 0.6 | Out-Null
Add-TB $s24 18 277 400 16 "Example -- Rail_TempLim  (2-step TC)" 9 $true $boschRed 1 | Out-Null
Add-TB $s24 18 295 405 78 ("Precondition:  Rail_stMon=0, Epm_nEng=3000, FuelT_t=40" + [char]10 + "               -> DLL warmup (1 raster)" + [char]10 + "" + [char]10 + "Step 1:  Epm_nEng=7000, FuelT_t=90  |  Wait 100000" + [char]10 + "         Expected: Rail_trqFlTempLim = 3800" + [char]10 + "         DLL returns 3842  |  tol=39  |  |3842-3800|=42 > tol  -> FAIL") 7 $false $darkGray 1 | Out-Null
Add-TB $s24 435 295 408 78 ("Step 2:  FuelT_t=60  |  Wait 200000 (2 rasters)" + [char]10 + "         Expected: Rail_trqFlTempLim = 4100" + [char]10 + "         DLL returns 4104  |  tol=42  |  |4104-4100|=4 < tol  -> PASS" + [char]10 + "" + [char]10 + "TC score = (0 + 1) / 2 = 0.50" + [char]10 + "M8 (this TC) = 0.50") 7 $false $darkGray 1 | Out-Null

Add-TB $s24 16 446 30 18 "24" 12 $false $midGray 1 | Out-Null

# ============================================================
# SLIDE 25 -- tasks done / pending
# ============================================================
$s25 = New-Slide 25
Add-SlideHeader $s25 "MBD-UTA -- Tasks Done vs Pending" "June 9, 2026"

Add-Rect $s25 16 78 406 340 $lightBg $algoBdr 0.7 | Out-Null
Add-SolidBox $s25 16 78 406 20 $greenCol "COMPLETED" 9 $true $white 1 | Out-Null

$done = @(
    @("M7 implemented  (inline 1/N)", "Was hardcoded stub 0.5. Now captures true retry count from Phase 4a."),
    @("M8 Functional Correctness added", "25% weight. Sequential DLL on final TCs. State carries across steps."),
    @("Metric weights rebalanced", "M1 25%->15%, M2 20%->15%, M3 10%->5%, M8 added at 25%. Total = 100%."),
    @("GEPA 50/50 group split defined", "testcase_generator M1+M3+M4+M6=50%.  equation_solver_dll M2+M7+M8=50%."),
    @("Bug fixes  (M3, M8 n_advances)", "M3 Wait multiple check fixed. M8 raster advance loop corrected."),
    @("_verify_mbd.py  52 checks passing", "All weight constants and GEPA rescaling factors verified after update.")
)

$dy = 102
foreach ($item in $done) {
    Add-TB $s25 20 $dy 399 14 $item[0] 8 $true $greenDk 1 | Out-Null
    Add-TB $s25 20 ($dy+15) 399 18 $item[1] 7 $false $darkGray 1 | Out-Null
    $dy += 50
}

Add-Rect $s25 432 78 415 340 $lightBg $algoBdr 0.7 | Out-Null
Add-SolidBox $s25 432 78 415 20 $amber "PENDING  /  NEXT STEPS" 9 $true $white 1 | Out-Null

$pending = @(
    @("Run GEPA with full pipeline", "Include equation_solver_dll group. Prompts 3+4 now in scope for optimization."),
    @("Scale to 37 FC tasks", "10 tasks loaded currently. Run --use-all-tasks for better generalization."),
    @("Held-out test set", "2 val examples only. Need proper test split before final eval."),
    @("Fix dedup bug  (run.py)", "Xa/Xb unit mismatch (RPM vs Nm). Add post-dedup DLL verify step."),
    @("EC coverage as M9  (optional)", "Equivalence-class partitions per branch op. ~10-15% weight if added."),
    @("GEPA full-run results PPT", "Document scores once run completes. Compare to June 2 baseline (91%).")
)

$py = 102
foreach ($item in $pending) {
    Add-TB $s25 436 $py 407 14 $item[0] 8 $true ([int](160+100*256+0*65536)) 1 | Out-Null
    Add-TB $s25 436 ($py+15) 407 18 $item[1] 7 $false $darkGray 1 | Out-Null
    $py += 50
}

Add-SolidBox $s25 16 422 831 22 $boschRed "Pipeline is now complete end-to-end: Phase 4a DLL inline (M7) + M8 sequential verification.  Next GEPA run will use all components for the first time." 8 $true $white 2 | Out-Null
Add-TB $s25 16 446 30 18 "25" 12 $false $midGray 1 | Out-Null

# save
$pres.Save()
$pres.Close()
$pptApp.Quit()
Write-Host "Done. Slides 21-25 added. File saved."

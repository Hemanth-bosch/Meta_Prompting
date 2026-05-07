# add_seed_only_conclusion.ps1
# Add conclusion slide after slide 24 for the seed-only WORST vs BEST experiment
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
$hcBg     = [int](235 + 245*256 + 255*65536)   # light blue tint for hardcoded row
$failBg   = [int](255 + 240*256 + 240*65536)   # light red tint for fail rows

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

# ── Insert at position 25 (after slide 24) ───────────────────
$refSl = $pres.Slides[24]
$sl    = $pres.Slides.AddSlide(25, $refSl.CustomLayout)
foreach ($sh in $refSl.Shapes) {
    if ($sh.Name -like "Gruppieren*" -or $sh.Name -like "Gruppe*") {
        try { $sh.Copy(); $sl.Shapes.Paste() | Out-Null } catch {}
    }
}

# ── Header ───────────────────────────────────────────────────
Add-TB $sl 16 10 740 22 "Sequential Tournament Framework for Automated Prompt Optimization" 16 $false $boschRed 1 | Out-Null
Add-TB $sl 16 32 400 18 "MiDAS Collaboration" 11 $false $darkGray 1 | Out-Null
Add-TB $sl 16 52 831 18 "Conclusion: Seed-Only Evaluation - WORST vs BEST Context Mode  (No Optimization Stages)" 13 $true $boschRed 1 | Out-Null
$ln1 = $sl.Shapes.AddLine(16, 72, 847, 72)
try { $ln1.Line.ForeColor.RGB = [int]$boschRed; $ln1.Line.Weight = 1.2 } catch {}

# ── Setup bar ────────────────────────────────────────────────
Add-Rect $sl 16 76 831 18 $codeBg $algoBdr 0.6 | Out-Null
Add-TB   $sl 18 77 829 16 "Setup: Student: gpt-4.1-mini  |  Seed LM: gpt-5.2  |  SEED STAGE ONLY (no Bootstrap / MIPROv2 / GEPA)  |  N=2 seeds  |  Mini val n=45  |  CR as seed signature" 7 $false $midGray 1 | Out-Null

# ── Two-panel layout ─────────────────────────────────────────
$pW = 413; $gX = 433    # 16+413+4=433, 433+414=847
$cH = 116; $rH = 16     # 7 rows x 16px

# Panel headers
Add-Rect $sl 16  98 $pW     18 $hdrDark  -1 0 | Out-Null
Add-TB   $sl 18  98 ($pW-4) 18 "WORST Context Mode  (120758)  -- seed LM sees worst-performing examples" 7 $true $white 1 | Out-Null
Add-Rect $sl $gX 98 ($pW+1) 18 $blueHdr  -1 0 | Out-Null
Add-TB   $sl ($gX+2) 98 ($pW-1) 18 "BEST Context Mode  (120822)  -- seed LM sees best-performing examples" 7 $true $white 1 | Out-Null

# Column sub-headers
$chBg = [int](80+80*256+80*65536)
foreach ($pX in @(16, $gX)) {
    $pw = $pW
    Add-Rect $sl $pX         $cH $pw 15 $chBg -1 0 | Out-Null
    Add-TB   $sl ($pX+2)     $cH 178 15 "Candidate"    5 $true $white 1 | Out-Null
    Add-TB   $sl ($pX+182)   $cH 40  15 "Stage"        5 $true $white 1 | Out-Null
    Add-TB   $sl ($pX+224)   $cH 56  15 "DCA Score"    5 $true $white 2 | Out-Null
    Add-TB   $sl ($pX+282)   $cH 66  15 "Tok/smp"      5 $true $white 2 | Out-Null
    Add-TB   $sl ($pX+350)   $cH 62  15 "Note"         5 $true $white 1 | Out-Null
}

# ── Row data: [Candidate, Stage, DCA%, Tok, FlagText, NameCol, TokCol, RowBg] ──
# WORST mode (120758) - 7 entries
$wRows = @(
    @("seed_baseline (hardcoded)",    "s_base", "86.40%", "3,057",  "HARDCODED",  $algoBdr,  $algoBdr,  $hcBg),
    @("seed_baseline_bare",           "seed",   "85.69%", "3,071",  "",           $darkGray, $greenCol, $lightBg),
    @("seed_baseline_with_demos",     "seed",   "82.92%", "21,848", "HIGH TOK",   $darkGray, $warnCol,  $altBg),
    @("seed_candidate_0_with_demos",  "seed",   "80.69%", "22,563", "HIGH TOK",   $darkGray, $warnCol,  $lightBg),
    @("seed_candidate_0",             "seed",   "78.82%", "3,777",  "",           $darkGray, $midGray,  $altBg),
    @("seed_candidate_1",             "seed",   "68.40%", "3,540",  "FAIL -18 pp",$warnCol,  $warnCol,  $failBg),
    @("seed_candidate_1_with_demos",  "seed",   "66.29%", "22,326", "FAIL+HIGH",  $warnCol,  $warnCol,  $failBg)
)
# BEST mode (120822) - 7 entries
$bRows = @(
    @("seed_candidate_0_with_demos",  "seed",   "86.35%", "21,850", "HIGH TOK",   $darkGray, $warnCol,  $hcBg),
    @("seed_baseline (hardcoded)",    "s_base", "85.57%", "3,067",  "HARDCODED",  $algoBdr,  $algoBdr,  $lightBg),
    @("seed_baseline_bare",           "seed",   "85.33%", "3,060",  "",           $darkGray, $greenCol, $altBg),
    @("seed_candidate_0",             "seed",   "84.81%", "3,977",  "",           $darkGray, $greenCol, $lightBg),
    @("seed_candidate_1_with_demos",  "seed",   "84.59%", "21,600", "HIGH TOK",   $darkGray, $warnCol,  $altBg),
    @("seed_baseline_with_demos",     "seed",   "81.32%", "20,924", "HIGH TOK",   $darkGray, $warnCol,  $lightBg),
    @("seed_candidate_1",             "seed",   "77.20%", "3,726",  "",           $darkGray, $midGray,  $altBg)
)

$rY = $cH + 15
for ($i = 0; $i -lt 7; $i++) {
    $wr = $wRows[$i]; $br = $bRows[$i]

    # WORST panel
    Add-Rect $sl 16        $rY $pW      $rH $wr[7] $midGray 0.3 | Out-Null
    Add-TB   $sl 18        $rY 178 $rH $wr[0] 5 $false $wr[5] 1 | Out-Null
    Add-TB   $sl 198       $rY 40  $rH $wr[1] 5 $false $midGray 1 | Out-Null
    Add-TB   $sl 224       $rY 56  $rH $wr[2] 7 ($i -eq 0) $wr[5] 2 | Out-Null
    Add-TB   $sl 282       $rY 66  $rH $wr[3] 6 $false $wr[6] 2 | Out-Null
    if ($wr[4] -ne "") { Add-TB $sl 350 $rY 62 $rH $wr[4] 5 $true $wr[5] 1 | Out-Null }

    # BEST panel
    Add-Rect $sl $gX       $rY ($pW+1)  $rH $br[7] $midGray 0.3 | Out-Null
    Add-TB   $sl ($gX+2)   $rY 178 $rH $br[0] 5 $false $br[5] 1 | Out-Null
    Add-TB   $sl ($gX+182) $rY 40  $rH $br[1] 5 $false $midGray 1 | Out-Null
    Add-TB   $sl ($gX+224) $rY 56  $rH $br[2] 7 ($i -eq 0) $br[5] 2 | Out-Null
    Add-TB   $sl ($gX+282) $rY 66  $rH $br[3] 6 $false $br[6] 2 | Out-Null
    if ($br[4] -ne "") { Add-TB $sl ($gX+350) $rY 62 $rH $br[4] 5 $true $br[6] 1 | Out-Null }

    $rY += $rH
}

# ── Separator ────────────────────────────────────────────────
$ln2 = $sl.Shapes.AddLine(16, $rY, 847, $rY)
try { $ln2.Line.ForeColor.RGB = [int]$midGray; $ln2.Line.Weight = 0.5 } catch {}
$rY += 4

# ── Summary comparison bar ───────────────────────────────────
Add-Rect $sl 16 $rY 831 15 ([int](60+60*256+60*65536)) -1 0 | Out-Null
Add-TB   $sl 18  $rY 180 15 "Mode"                 6 $true $white 1 | Out-Null
Add-TB   $sl 200 $rY 120 15 "Best seeded prompt"   6 $true $white 2 | Out-Null
Add-TB   $sl 322 $rY 100 15 "vs Hardcoded baseline"6 $true $white 2 | Out-Null
Add-TB   $sl 424 $rY 80  15 "Variance"             6 $true $white 2 | Out-Null
Add-TB   $sl 506 $rY 340 15 "Worst candidate"      6 $true $white 1 | Out-Null
$rY += 15

$sumRows = @(
    @("WORST ctx  (120758)", "85.69%  3,071 tok (bare)", "-0.71 pp (baseline wins!)", "HIGH",   "seed_cand_1: 68.40%  seed_cand_1+demos: 66.29%  (-20 pp FAIL)"),
    @("BEST ctx  (120822)",  "86.35%  21,850 tok (demos)","same (at 7x token cost!)", "LOW",    "seed_cand_1: 77.20%  seed_baseline_demos: 81.32%")
)
$sBgs = @($lightBg, $altBg)
for ($i = 0; $i -lt 2; $i++) {
    $sr = $sumRows[$i]
    Add-Rect $sl 16  $rY 831 15 $sBgs[$i] $midGray 0.3 | Out-Null
    Add-TB   $sl 18  $rY 180 15 $sr[0] 6 $true  $darkGray 1 | Out-Null
    Add-TB   $sl 200 $rY 120 15 $sr[1] 6 $false $darkGray 2 | Out-Null
    $vsCol = if ($i -eq 0) { $warnCol } else { $midGray }
    Add-TB   $sl 322 $rY 100 15 $sr[2] 6 $true  $vsCol    2 | Out-Null
    $varCol = if ($i -eq 0) { $warnCol } else { $greenCol }
    Add-TB   $sl 424 $rY 80  15 $sr[3] 6 $true  $varCol   2 | Out-Null
    Add-TB   $sl 506 $rY 340 15 $sr[4] 6 $false $warnCol  1 | Out-Null
    $rY += 15
}

$ln3 = $sl.Shapes.AddLine(16, $rY, 847, $rY)
try { $ln3.Line.ForeColor.RGB = [int]$midGray; $ln3.Line.Weight = 0.5 } catch {}
$rY += 4

# ── KEY FINDING ──────────────────────────────────────────────
$fnd = Add-Rect $sl 16 $rY 831 26 $boschRed -1 0
try {
    $fnd.TextFrame.TextRange.Text = "KEY FINDING:  WORST ctx causes catastrophic failure for seed_candidate_1 (66-68%, -20 pp).  BEST ctx more stable but best seed (86.35%) needs 21,850 tok (7x cost) to barely match baseline (86.40%).  Neither mode's raw seeds meaningfully beat the hardcoded baseline -- optimization stages are the source of tournament value."
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
Add-TB   $sl 18  ($rY+2)  404 12 "WHY WORST CTX FAILS FOR CANDIDATE_1" 8 $true $warnCol 1 | Out-Null
Add-TB   $sl 18  ($rY+15) 404 46 ("WORST examples shown to seed LM are the worst failures of the current prompt." + [char]10 + "For candidate_1, these appear to have confused the seed LM into generating" + [char]10 + "a poor instruction. High-variance output: some candidates gain, others collapse." + [char]10 + "WORST ctx is only safe when combined with the full optimization pipeline.") 6 $false $darkGray 1 | Out-Null

Add-Rect $sl 428 $rY 419 64 $lightBg $algoBdr 0.75 | Out-Null
Add-TB   $sl 430 ($rY+2)  415 12 "OPTIMIZATION PIPELINE IS THE KEY VALUE-ADD" 8 $true $blueHdr 1 | Out-Null
Add-TB   $sl 430 ($rY+15) 415 46 ("Seed-only WORST ctx: seed_cand_0_with_demos = 80.69% at 22,563 tok." + [char]10 + "Full tournament WORST ctx (090909): same lineage --> 92.55% after Bootstrap+GEPA." + [char]10 + "+11.86 pp improvement from optimization alone. Seeds are starting points, not answers." + [char]10 + "DCA metric is instruction-sensitive: optimizer rewrites matter most.") 6 $false $darkGray 1 | Out-Null
$rY += 68

# ── Recommendation ───────────────────────────────────────────
$rec = Add-Rect $sl 16 $rY 831 26 $codeBg $algoBdr 0.8
try {
    $rec.TextFrame.TextRange.Text = "RECOMMENDATION:  Do not use seed-only evaluation as a proxy for tournament performance.  Always run the full pipeline (Bootstrap->MIPROv2->GEPA).  WORST context mode (--seed-context-mode worst) is safe and beneficial ONLY when combined with optimization stages -- raw WORST seeds have high variance and can fail catastrophically (-20 pp for candidate_1)."
    $rec.TextFrame.TextRange.Font.Name = "Bosch Office Sans"
    $rec.TextFrame.TextRange.Font.Size = 7
    $rec.TextFrame.TextRange.Font.Color.RGB = [int]$darkGray
    foreach ($p in $rec.TextFrame.TextRange.Paragraphs()) { $p.ParagraphFormat.Alignment = 1 }
    $rec.TextFrame.MarginLeft = 5; $rec.TextFrame.MarginTop = 3
} catch {}

Add-TB $sl 16 446 30 18 "25" 12 $false $midGray 1 | Out-Null

# ── Save ─────────────────────────────────────────────────────
$pres.SaveAs($outPath)
$pres.Close()
$pptApp.Quit()
Write-Host "Done -- seed-only conclusion slide added at position 25"
Write-Host "Total slides:" ($pres.Slides.Count + 1)

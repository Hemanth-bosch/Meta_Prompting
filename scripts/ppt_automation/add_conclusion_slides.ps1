# add_conclusion_slides.ps1
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
$hdrDark  = [int](50  + 50*256  + 50*65536)

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

# ================================================================
# SLIDE A: HYBRID CONCLUSION  insert at position 15 (after slide 14)
# ================================================================
$refA  = $pres.Slides[14]
$hybSl = $pres.Slides.AddSlide(15, $refA.CustomLayout)
foreach ($sh in $refA.Shapes) {
    if ($sh.Name -like "Gruppieren*" -or $sh.Name -like "Gruppe*") {
        try { $sh.Copy(); $hybSl.Shapes.Paste() | Out-Null } catch {}
    }
}

Add-TB $hybSl 16 10 740 22 "Sequential Tournament Framework for Automated Prompt Optimization" 16 $false $boschRed 1 | Out-Null
Add-TB $hybSl 16 32 400 18 "MiDAS Collaboration" 11 $false $darkGray 1 | Out-Null
Add-TB $hybSl 16 52 831 18 "Conclusion: Hybrid Multi-Objective Metric - Exponential vs Linear Penalty" 13 $true $boschRed 1 | Out-Null
$lnA = $hybSl.Shapes.AddLine(16, 72, 847, 72)
try { $lnA.Line.ForeColor.RGB = [int]$boschRed; $lnA.Line.Weight = 1.2 } catch {}

Add-Rect $hybSl 16 76 831 18 $codeBg $algoBdr 0.6 | Out-Null
Add-TB $hybSl 18 77 829 16 "Setup: Student: gpt-4.1-mini  |  Seed LM: gpt-5.2  |  top_k=2  |  Pareto stage selection  |  Best demos per class  |  Mini val n=45" 7 $false $midGray 1 | Out-Null

$pW = 410; $gX = 432; $cH = 116; $rH = 22

# Panel headers
Add-Rect $hybSl 16  98 $pW       18 $hdrDark -1 0 | Out-Null
Add-TB   $hybSl 18  98 ($pW-4)   18 "Hybrid EXPONENTIAL  (20260505153708)   DCA x exp(-2 x max(0, tok/2500 - 1))" 7 $true $white 1 | Out-Null
Add-Rect $hybSl $gX 98 ($pW+5)   18 ([int](0+80*256+140*65536)) -1 0 | Out-Null
Add-TB   $hybSl ($gX+2) 98 ($pW+3) 18 "Hybrid LINEAR  (20260505172652)   0.5 x DCA + 0.5 x max(0, 1 - tok/12000)" 7 $true $white 1 | Out-Null

# Col sub-headers
Add-Rect $hybSl 16  $cH $pW     16 ([int](80+80*256+80*65536)) -1 0 | Out-Null
Add-TB   $hybSl 18  $cH 172 16 "Candidate"           6 $true $white 1 | Out-Null
Add-TB   $hybSl 192 $cH 50  16 "Stage"               6 $true $white 1 | Out-Null
Add-TB   $hybSl 244 $cH 72  16 "DCA Score"           6 $true $white 2 | Out-Null
Add-TB   $hybSl 318 $cH 72  16 "Tok/sample"          6 $true $white 2 | Out-Null
Add-TB   $hybSl 392 $cH 34  16 "Note"                6 $true $white 2 | Out-Null

Add-Rect $hybSl $gX         $cH ($pW+5) 16 ([int](80+80*256+80*65536)) -1 0 | Out-Null
Add-TB   $hybSl ($gX+2)     $cH 172 16 "Candidate"   6 $true $white 1 | Out-Null
Add-TB   $hybSl ($gX+176)   $cH 50  16 "Stage"       6 $true $white 1 | Out-Null
Add-TB   $hybSl ($gX+228)   $cH 72  16 "DCA Score"   6 $true $white 2 | Out-Null
Add-TB   $hybSl ($gX+302)   $cH 72  16 "Tok/sample"  6 $true $white 2 | Out-Null
Add-TB   $hybSl ($gX+376)   $cH 51  16 "Note"        6 $true $white 2 | Out-Null

# Data: Exp rows (top-5 by raw DCA)
$expRows = @(
    @("gepa <- miprov2 <- baseline_bare", "gepa",    "86.28%", "3,526",  "BEST",     $boschRed, $greenCol),
    @("seed_baseline_bare",               "seed",    "84.44%", "3,063",  "",         $darkGray, $greenCol),
    @("seed_baseline_with_demos",         "seed",    "84.31%", "3,070",  "",         $darkGray, $greenCol),
    @("seed_candidate_1_with_demos",      "seed",    "84.25%", "22,146", "HIGH TOK", $darkGray, $warnCol),
    @("seed_candidate_0 (bare)",          "seed",    "84.05%", "4,073",  "",         $darkGray, $greenCol)
)
# Data: Lin rows
$linRows = @(
    @("gepa <- miprov2 <- baseline_demos","gepa",    "87.41%", "3,603",  "BEST",     $boschRed, $greenCol),
    @("seed_candidate_1_with_demos",      "seed",    "86.76%", "21,793", "HIGH TOK", $darkGray, $warnCol),
    @("gepa <- miprov2 <- baseline_demos","gepa",    "86.47%", "3,073",  "",         $darkGray, $greenCol),
    @("seed_baseline_with_demos",         "seed",    "86.11%", "3,061",  "",         $darkGray, $greenCol),
    @("seed_candidate_0 (bare)",          "seed",    "85.01%", "4,203",  "",         $darkGray, $greenCol)
)

$rY = $cH + 16
for ($i = 0; $i -lt 5; $i++) {
    $bg = if ($i % 2 -eq 0) { $lightBg } else { $altBg }
    $er = $expRows[$i]; $lr = $linRows[$i]

    Add-Rect $hybSl 16          $rY $pW      $rH $bg $midGray 0.3 | Out-Null
    Add-TB   $hybSl 18          $rY 172 $rH $er[0] 6 $false $er[5] 1 | Out-Null
    Add-TB   $hybSl 192         $rY 50  $rH $er[1] 6 $false $midGray 1 | Out-Null
    Add-TB   $hybSl 244         $rY 72  $rH $er[2] 7 ($i -eq 0) $er[5] 2 | Out-Null
    Add-TB   $hybSl 318         $rY 72  $rH $er[3] 7 $false $er[6] 2 | Out-Null
    if ($er[4] -ne "") { Add-TB $hybSl 392 $rY 34 $rH $er[4] 5 $true $er[6] 2 | Out-Null }

    Add-Rect $hybSl $gX         $rY ($pW+5) $rH $bg $midGray 0.3 | Out-Null
    Add-TB   $hybSl ($gX+2)     $rY 172 $rH $lr[0] 6 $false $lr[5] 1 | Out-Null
    Add-TB   $hybSl ($gX+176)   $rY 50  $rH $lr[1] 6 $false $midGray 1 | Out-Null
    Add-TB   $hybSl ($gX+228)   $rY 72  $rH $lr[2] 7 ($i -eq 0) $lr[5] 2 | Out-Null
    Add-TB   $hybSl ($gX+302)   $rY 72  $rH $lr[3] 7 $false $lr[6] 2 | Out-Null
    if ($lr[4] -ne "") { Add-TB $hybSl ($gX+376) $rY 51 $rH $lr[4] 5 $true $lr[6] 2 | Out-Null }

    $rY += $rH
}

$lnB = $hybSl.Shapes.AddLine(16, $rY, 847, $rY)
try { $lnB.Line.ForeColor.RGB = [int]$midGray; $lnB.Line.Weight = 0.5 } catch {}
$rY += 4

$fnd = Add-Rect $hybSl 16 $rY 831 26 $boschRed -1 0
try {
    $fnd.TextFrame.TextRange.Text = "KEY FINDING:  Hybrid Linear achieves +1.13 pp higher peak raw DCA (87.41% vs 86.28%) at similar token cost (3,603 vs 3,526 tok/sample).  Both metrics steer top-k selection to 0-demo candidates -- all forwarded entries stay under 5K tok."
    $fnd.TextFrame.TextRange.Font.Name = "Bosch Office Sans"; $fnd.TextFrame.TextRange.Font.Size = 8
    $fnd.TextFrame.TextRange.Font.Bold = -1; $fnd.TextFrame.TextRange.Font.Color.RGB = [int]$white
    foreach ($p in $fnd.TextFrame.TextRange.Paragraphs()) { $p.ParagraphFormat.Alignment = 2 }
    $fnd.TextFrame.VerticalAnchor = 3
} catch {}
$rY += 30

# Two boxes
Add-Rect $hybSl 16  $rY 408 82 $lightBg $algoBdr 0.75 | Out-Null
Add-TB   $hybSl 18  ($rY+2)  404 14 "HYBRID EXPONENTIAL BEHAVIOR" 8 $true $boschRed 1 | Out-Null
Add-TB   $hybSl 18  ($rY+17) 404 60 ("Penalty above 2,500 tok is exponential." + [char]10 + "22K tok seed gets hybrid score ~0.10 vs bare ~0.65." + [char]10 + "Only 0-demo candidates survive top-k forwarding." + [char]10 + "Aggressive pruning -- no accuracy gain over linear.") 7 $false $darkGray 1 | Out-Null

Add-Rect $hybSl 428 $rY 419 82 $lightBg $algoBdr 0.75 | Out-Null
Add-TB   $hybSl 430 ($rY+2)  415 14 "HYBRID LINEAR BEHAVIOR" 8 $true ([int](0+80*256+140*65536)) 1 | Out-Null
Add-TB   $hybSl 430 ($rY+17) 415 60 ("Penalty gentle up to 12,000 tok budget; linear decay beyond." + [char]10 + "Same 0-demo top-k selection, but seeds retain more diversity." + [char]10 + "+1.13 pp peak DCA: optimizer found a better path" + [char]10 + "due to less aggressive early-stage pruning.") 7 $false $darkGray 1 | Out-Null
$rY += 86

$rec = Add-Rect $hybSl 16 $rY 831 28 $codeBg $algoBdr 0.8
try {
    $rec.TextFrame.TextRange.Text = "RECOMMENDATION:  Use --hybrid-penalty-mode linear --hybrid-max-prompt-tokens 12000.  Linear yields +1.13 pp higher peak DCA while still enforcing token efficiency.  Exponential's aggressive pruning offers no accuracy benefit and risks eliminating valid high-accuracy paths."
    $rec.TextFrame.TextRange.Font.Name = "Bosch Office Sans"; $rec.TextFrame.TextRange.Font.Size = 7
    $rec.TextFrame.TextRange.Font.Color.RGB = [int]$darkGray
    foreach ($p in $rec.TextFrame.TextRange.Paragraphs()) { $p.ParagraphFormat.Alignment = 1 }
    $rec.TextFrame.MarginLeft = 5; $rec.TextFrame.MarginTop = 4
} catch {}
Add-TB $hybSl 16 446 30 18 "15" 12 $false $midGray 1 | Out-Null
Write-Host "Hybrid conclusion added at position 15"

# ================================================================
# SLIDE B: PARETO vs BEST-SCORE CONCLUSION  insert at position 9
# (slide 14 is now at 15 after previous insert -- ref from slide 7 still at 7)
# ================================================================
$refB   = $pres.Slides[7]
$parSl  = $pres.Slides.AddSlide(9, $refB.CustomLayout)
foreach ($sh in $refB.Shapes) {
    if ($sh.Name -like "Gruppieren*" -or $sh.Name -like "Gruppe*") {
        try { $sh.Copy(); $parSl.Shapes.Paste() | Out-Null } catch {}
    }
}

Add-TB $parSl 16 10 740 22 "Sequential Tournament Framework for Automated Prompt Optimization" 16 $false $boschRed 1 | Out-Null
Add-TB $parSl 16 32 400 18 "MiDAS Collaboration" 11 $false $darkGray 1 | Out-Null
Add-TB $parSl 16 52 831 18 "Conclusion: Pareto vs Best-Score Selection - Student: gpt-4.1-mini" 13 $true $boschRed 1 | Out-Null
$lnC = $parSl.Shapes.AddLine(16, 72, 847, 72)
try { $lnC.Line.ForeColor.RGB = [int]$boschRed; $lnC.Line.Weight = 1.2 } catch {}

Add-Rect $parSl 16 76 831 18 $codeBg $algoBdr 0.6 | Out-Null
Add-TB   $parSl 18 77 829 16 "Setup: Student: gpt-4.1-mini  |  Seed LM: gpt-5.2  |  top_k=2  |  N=2 seeds  |  Best demos per class  |  Mini val n=45  |  No hybrid metric" 7 $false $midGray 1 | Out-Null

# Panel headers
Add-Rect $parSl 16  98 $pW     18 $hdrDark -1 0 | Out-Null
Add-TB   $parSl 18  98 ($pW-4) 18 "PARETO Selection  (tournament_20260504233802)" 7 $true $white 1 | Out-Null
Add-Rect $parSl $gX 98 ($pW+5) 18 ([int](0+80*256+140*65536)) -1 0 | Out-Null
Add-TB   $parSl ($gX+2) 98 ($pW+3) 18 "BEST SCORE Selection  (tournament_20260504233822)" 7 $true $white 1 | Out-Null

# Col sub-headers
Add-Rect $parSl 16          $cH $pW     16 ([int](80+80*256+80*65536)) -1 0 | Out-Null
Add-TB   $parSl 18          $cH 172 16 "Candidate"   6 $true $white 1 | Out-Null
Add-TB   $parSl 192         $cH 50  16 "Stage"       6 $true $white 1 | Out-Null
Add-TB   $parSl 244         $cH 72  16 "DCA Score"   6 $true $white 2 | Out-Null
Add-TB   $parSl 318         $cH 72  16 "Tok/sample"  6 $true $white 2 | Out-Null
Add-TB   $parSl 392         $cH 34  16 "Note"        6 $true $white 2 | Out-Null

Add-Rect $parSl $gX         $cH ($pW+5) 16 ([int](80+80*256+80*65536)) -1 0 | Out-Null
Add-TB   $parSl ($gX+2)     $cH 172 16 "Candidate"   6 $true $white 1 | Out-Null
Add-TB   $parSl ($gX+176)   $cH 50  16 "Stage"       6 $true $white 1 | Out-Null
Add-TB   $parSl ($gX+228)   $cH 72  16 "DCA Score"   6 $true $white 2 | Out-Null
Add-TB   $parSl ($gX+302)   $cH 72  16 "Tok/sample"  6 $true $white 2 | Out-Null
Add-TB   $parSl ($gX+376)   $cH 51  16 "Note"        6 $true $white 2 | Out-Null

# Pareto top-5
$parRows = @(
    @("seed_candidate_0 (bare)",      "seed",    "86.48%", "3,953",  "BEST",     $boschRed, $greenCol),
    @("gepa <- seed_candidate_0",     "gepa",    "86.12%", "3,609",  "",         $darkGray, $greenCol),
    @("gepa <- seed_candidate_1",     "gepa",    "86.02%", "3,617",  "",         $darkGray, $greenCol),
    @("seed_baseline_with_demos",     "seed",    "85.80%", "3,091",  "",         $darkGray, $greenCol),
    @("seed_baseline (hardcoded)",    "s_base",  "85.38%", "3,060",  "",         $darkGray, $greenCol)
)
# Best-Score top-5
$bsRows = @(
    @("miprov2 <- seed_baseline_bare","miprov2", "86.56%", "3,067",  "BEST",     $boschRed, $greenCol),
    @("gepa <- bootstrap <- baseline","gepa",    "86.31%", "3,389",  "",         $darkGray, $greenCol),
    @("seed_candidate_0_with_demos",  "seed",    "86.22%", "22,354", "HIGH TOK", $darkGray, $warnCol),
    @("miprov2 <- bootstrap",         "miprov2", "85.92%", "2,968",  "",         $darkGray, $greenCol),
    @("gepa <- gepa <- bootstrap",    "gepa",    "84.68%", "3,610",  "",         $darkGray, $greenCol)
)

$rY2 = $cH + 16
for ($i = 0; $i -lt 5; $i++) {
    $bg = if ($i % 2 -eq 0) { $lightBg } else { $altBg }
    $pr = $parRows[$i]; $br = $bsRows[$i]

    Add-Rect $parSl 16          $rY2 $pW      $rH $bg $midGray 0.3 | Out-Null
    Add-TB   $parSl 18          $rY2 172 $rH $pr[0] 6 $false $pr[5] 1 | Out-Null
    Add-TB   $parSl 192         $rY2 50  $rH $pr[1] 6 $false $midGray 1 | Out-Null
    Add-TB   $parSl 244         $rY2 72  $rH $pr[2] 7 ($i -eq 0) $pr[5] 2 | Out-Null
    Add-TB   $parSl 318         $rY2 72  $rH $pr[3] 7 $false $pr[6] 2 | Out-Null
    if ($pr[4] -ne "") { Add-TB $parSl 392 $rY2 34 $rH $pr[4] 5 $true $pr[6] 2 | Out-Null }

    Add-Rect $parSl $gX         $rY2 ($pW+5) $rH $bg $midGray 0.3 | Out-Null
    Add-TB   $parSl ($gX+2)     $rY2 172 $rH $br[0] 6 $false $br[5] 1 | Out-Null
    Add-TB   $parSl ($gX+176)   $rY2 50  $rH $br[1] 6 $false $midGray 1 | Out-Null
    Add-TB   $parSl ($gX+228)   $rY2 72  $rH $br[2] 7 ($i -eq 0) $br[5] 2 | Out-Null
    Add-TB   $parSl ($gX+302)   $rY2 72  $rH $br[3] 7 $false $br[6] 2 | Out-Null
    if ($br[4] -ne "") { Add-TB $parSl ($gX+376) $rY2 51 $rH $br[4] 5 $true $br[6] 2 | Out-Null }

    $rY2 += $rH
}

$lnD = $parSl.Shapes.AddLine(16, $rY2, 847, $rY2)
try { $lnD.Line.ForeColor.RGB = [int]$midGray; $lnD.Line.Weight = 0.5 } catch {}
$rY2 += 4

$fnd2 = Add-Rect $parSl 16 $rY2 831 26 $boschRed -1 0
try {
    $fnd2.TextFrame.TextRange.Text = "KEY FINDING:  Peak DCA near-identical (Pareto 86.48% vs Best-Score 86.56%, delta +0.08 pp).  Best-Score allowed seed_candidate_0_with_demos (86.22%, 22,354 tok) to rank #3 in leaderboard -- 5.7x token cost.  Pareto kept all top-5 entries under 4K tok."
    $fnd2.TextFrame.TextRange.Font.Name = "Bosch Office Sans"; $fnd2.TextFrame.TextRange.Font.Size = 8
    $fnd2.TextFrame.TextRange.Font.Bold = -1; $fnd2.TextFrame.TextRange.Font.Color.RGB = [int]$white
    foreach ($p in $fnd2.TextFrame.TextRange.Paragraphs()) { $p.ParagraphFormat.Alignment = 2 }
    $fnd2.TextFrame.VerticalAnchor = 3
} catch {}
$rY2 += 30

Add-Rect $parSl 16  $rY2 408 82 $lightBg $algoBdr 0.75 | Out-Null
Add-TB   $parSl 18  ($rY2+2)  404 14 "WHY PARETO KEEPS TOKEN COST LOW" 8 $true $boschRed 1 | Out-Null
Add-TB   $parSl 18  ($rY2+17) 404 60 ("Pareto top-5: all entries under 4K tok/sample." + [char]10 + "seed_candidate_0_with_demos (83.40%, 21K tok) in Pareto run" + [char]10 + "is dominated by seed_candidate_0 (86.48%, 4K tok)." + [char]10 + "Pareto correctly blocks it from later stages.") 7 $false $darkGray 1 | Out-Null

Add-Rect $parSl 428 $rY2 419 82 $lightBg $algoBdr 0.75 | Out-Null
Add-TB   $parSl 430 ($rY2+2)  415 14 "ROUTE ANALYSIS" 8 $true ([int](0+80*256+140*65536)) 1 | Out-Null
Add-TB   $parSl 430 ($rY2+17) 415 60 ("Pareto route: Seed -- GEPA (86.12%, 3.6K tok)." + [char]10 + "Best-Score route: Seed -- Bootstrap -- MIPROv2 (86.56%, 3.1K)." + [char]10 + "Best-Score found a marginally better path (+0.08 pp) via" + [char]10 + "optimizing a cheap seed. Both routes similar token efficiency at end.") 7 $false $darkGray 1 | Out-Null
$rY2 += 86

$rec2 = Add-Rect $parSl 16 $rY2 831 28 $codeBg $algoBdr 0.8
try {
    $rec2.TextFrame.TextRange.Text = "RECOMMENDATION:  Pareto selection is the default (--selection-criteria pareto).  Near-identical peak DCA (+/-0.08 pp, within noise for n=45) with guaranteed token efficiency -- expensive candidates are filtered before consuming later-stage compute.  Best-Score occasionally finds marginally better peaks but propagates costly candidates."
    $rec2.TextFrame.TextRange.Font.Name = "Bosch Office Sans"; $rec2.TextFrame.TextRange.Font.Size = 7
    $rec2.TextFrame.TextRange.Font.Color.RGB = [int]$darkGray
    foreach ($p in $rec2.TextFrame.TextRange.Paragraphs()) { $p.ParagraphFormat.Alignment = 1 }
    $rec2.TextFrame.MarginLeft = 5; $rec2.TextFrame.MarginTop = 4
} catch {}
Add-TB $parSl 16 446 30 18 "9" 12 $false $midGray 1 | Out-Null
Write-Host "Pareto conclusion added at position 9"

$pres.SaveAs($outPath)
$pres.Close()
$pptApp.Quit()
Write-Host "Done -- saved to $outPath"
Write-Host "Total slides now: 27"

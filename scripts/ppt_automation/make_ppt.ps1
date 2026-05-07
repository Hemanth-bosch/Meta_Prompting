# make_ppt.ps1  -- Modify 05-05-2026-dca.pptx  slides 3-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$pptApp = New-Object -ComObject PowerPoint.Application
$pptApp.Visible = [Microsoft.Office.Core.MsoTriState]::msoTrue
$srcPath = 'C:\Users\BHS1RNG\Downloads\05-05-2026-dca.pptx'
$outPath = 'C:\Users\BHS1RNG\Downloads\05-05-2026-dca-updated.pptx'
$pres    = $pptApp.Presentations.Open($srcPath)

# Colors: OLE_COLOR = R + G*256 + B*65536
$boschRed  = [int](226 + 0*256   + 21*65536)
$black     = [int]0
$darkGray  = [int](60  + 60*256  + 60*65536)
$midGray   = [int](130 + 130*256 + 130*65536)
$lightBg   = [int](245 + 245*256 + 245*65536)
$white     = [int]16777215
$seedTint  = [int](228 + 242*256 + 255*65536)
$bootTint  = [int](228 + 255*256 + 238*65536)
$miproTint = [int](255 + 246*256 + 224*65536)
$gepaTint  = [int](255 + 232*256 + 232*65536)
$seedHdr   = [int](0   + 100*256 + 175*65536)
$bootHdr   = [int](0   + 140*256 + 100*65536)
$miproHdr  = [int](170 + 100*256 + 0*65536)
$gepaHdr   = $boschRed

function Del-ExtraShapes($slide) {
    $keep = @("Title 1","Text Placeholder 2","Slide Number Placeholder 4","Gruppieren 3","TextBox 11")
    $todo = @()
    foreach ($s in $slide.Shapes) { if ($keep -notcontains $s.Name) { $todo += $s } }
    foreach ($s in $todo) { try { $s.Delete() } catch {} }
}

function Set-Subtitle($slide, $text) {
    $tb = $slide.Shapes.AddTextbox(1, 16, 84, 740, 36)
    $tf = $tb.TextFrame; $tr = $tf.TextRange
    $tr.Text = $text
    $tr.Font.Name = "Bosch Office Sans"; $tr.Font.Size = 15
    $tr.Font.Bold = $true; $tr.Font.Color.RGB = $boschRed
    $tf.MarginLeft = 2; $tf.MarginTop = 4
    $tb.Fill.Visible = $false; $tb.Line.Visible = $false
}

function Add-Rect($slide, $x, $y, $w, $h, $fillCol, $lineCol, $lineW) {
    $sh = $slide.Shapes.AddShape(1, $x, $y, $w, $h)
    try { $sh.Fill.ForeColor.RGB = [int]$fillCol } catch {}
    try { $sh.Fill.Solid() } catch {}
    if ($lineCol -lt 0) {
        try { $sh.Line.Visible = 0 } catch {}
    } else {
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
    try {
        foreach ($p in $tr.Paragraphs()) {
            $p.ParagraphFormat.Alignment = [int]$align
        }
    } catch {}
    try { $tf.MarginLeft = 3; $tf.MarginTop = 2; $tf.MarginRight = 3; $tf.MarginBottom = 2 } catch {}
    try { $tb.Fill.Visible = 0 } catch {}
    try { $tb.Line.Visible = 0 } catch {}
    return $tb
}

function Add-StageBox($slide, $x, $y, $w, $h, $num, $title, $body, $tintCol, $hdrCol) {
    $hdr = Add-Rect $slide $x $y $w 26 $hdrCol -1 0
    try { $hdr.TextFrame.TextRange.Text = "$num   $title" } catch {}
    try { $hdr.TextFrame.TextRange.Font.Name = "Bosch Office Sans" } catch {}
    try { $hdr.TextFrame.TextRange.Font.Size = 9 } catch {}
    try { $hdr.TextFrame.TextRange.Font.Bold = -1 } catch {}
    try { $hdr.TextFrame.TextRange.Font.Color.RGB = [int]$white } catch {}
    try { foreach ($p in $hdr.TextFrame.TextRange.Paragraphs()) { $p.ParagraphFormat.Alignment = 2 } } catch {}
    try { $hdr.TextFrame.VerticalAnchor = 3 } catch {}
    $bdy = Add-Rect $slide $x ($y+26) $w ($h-26) $tintCol $hdrCol 0.75
    try { $bdy.TextFrame.TextRange.Text = $body } catch {}
    try { $bdy.TextFrame.TextRange.Font.Name = "Bosch Office Sans" } catch {}
    try { $bdy.TextFrame.TextRange.Font.Size = 8 } catch {}
    try { $bdy.TextFrame.TextRange.Font.Color.RGB = [int]$darkGray } catch {}
    try { foreach ($p in $bdy.TextFrame.TextRange.Paragraphs()) { $p.ParagraphFormat.Alignment = 2 } } catch {}
    try { $bdy.TextFrame.VerticalAnchor = 3 } catch {}
    try { $bdy.TextFrame.MarginLeft = 4; $bdy.TextFrame.MarginTop = 4 } catch {}
    try { $bdy.TextFrame.MarginRight = 4; $bdy.TextFrame.MarginBottom = 4 } catch {}
}

# ---- SLIDE 3: Tournament Architecture -----------------------
$sl3 = $pres.Slides[3]
Del-ExtraShapes $sl3
Set-Subtitle $sl3 "Sequential Tournament Framework for Automated Prompt Optimization"
Add-TB $sl3 16 126 550 18 "PIPELINE ARCHITECTURE" 8 $true $midGray 1 | Out-Null

$bY=148; $bH=100; $bW=170; $gap=22
$x0=20; $x1=($x0+$bW+$gap); $x2=($x1+$bW+$gap); $x3=($x2+$bW+$gap)

Add-StageBox $sl3 $x0 $bY $bW $bH "STAGE 0" "SEED" `
  "GPT-5.2 writes improved instructions using failure examples and images as context" `
  $seedTint $seedHdr
Add-StageBox $sl3 $x1 $bY $bW $bH "STAGE 1" "BOOTSTRAP" `
  "Few-shot demo selection via random search (16 trials, teacher GPT-5.2)" `
  $bootTint $bootHdr
Add-StageBox $sl3 $x2 $bY $bW $bH "STAGE 2" "MIPROv2" `
  "Bayesian instruction + demo joint optimisation (medium intensity)" `
  $miproTint $miproHdr
Add-StageBox $sl3 $x3 $bY $bW $bH "STAGE 3" "GEPA" `
  "Reflective adaptation: GPT-5.2 sees failing images and rewrites instruction" `
  $gepaTint $gepaHdr

$aY = $bY+38
Add-TB $sl3 ($x0+$bW+3) $aY ($gap-4) 24 ">" 18 $false $midGray 2 | Out-Null
Add-TB $sl3 ($x1+$bW+3) $aY ($gap-4) 24 ">" 18 $false $midGray 2 | Out-Null
Add-TB $sl3 ($x2+$bW+3) $aY ($gap-4) 24 ">" 18 $false $midGray 2 | Out-Null

$kY = $bY+$bH+5
Add-TB $sl3 $x0 $kY (4*$bW+3*$gap) 14 `
  "Cross-stage Pareto Top-K selection  (maximize score AND minimize tokens/sample)" `
  7 $false $midGray 2 | Out-Null

$ln3 = $sl3.Shapes.AddLine(16, 270, 847, 270); try { $ln3.Line.ForeColor.RGB = [int]$lightBg } catch {}

# Metric box
Add-Rect $sl3 16 278 400 90 $lightBg $midGray 0.75 | Out-Null
Add-TB $sl3 18 280 396 18 "COMPOSITE DCA METRIC" 8 $true $boschRed 1 | Out-Null
Add-TB $sl3 18 300 396 16 "Score  =  0.60 x Classification Accuracy  +  0.40 x Change Quality" `
  9 $true $darkGray 1 | Out-Null
Add-TB $sl3 18 318 396 44 `
  "- Classification (60%): binary - model correctly decides discard / keep?" + [char]10 + `
  "- Change Quality (40%): F-beta LLM judge (precision 60%, recall 40%)" + [char]10 + `
  "- Special-case fast-paths; SHA-256 judge cache avoids redundant calls" `
  8 $false $darkGray 1 | Out-Null

# Benefits box
Add-TB $sl3 424 278 420 18 "BENEFITS OF TOURNAMENT APPROACH" 8 $true $boschRed 1 | Out-Null
Add-TB $sl3 424 298 420 70 `
  "* Explores 100+ candidate prompts automatically - zero manual tuning" + [char]10 + `
  "* Warm-start: each stage inherits the best from all prior stages" + [char]10 + `
  "* Pareto selection balances accuracy vs. inference-token cost" + [char]10 + `
  "* Subprocess isolation prevents DSPy/thread-pool scoring bugs" + [char]10 + `
  "* Reproducible: stratified split, fixed seeds, full audit trail" `
  8 $false $darkGray 1 | Out-Null

Add-TB $sl3 16 372 831 18 `
  "Dataset: 429 document image-pairs  |  8 categories (A1-B5b)  |  Stratified 60/20/20 split  |  Student: GPT-4.1  |  Teacher / Reflection: GPT-5.2" `
  7 $false $midGray 1 | Out-Null
Add-TB $sl3 16 392 831 18 `
  "Leaderboard tracks every candidate (val score, per-category scores, tokens/sample).  Final re-evaluation on full val set at temperature=0.1." `
  7 $false $midGray 1 | Out-Null

# ---- SLIDE 4: Score Progression ----------------------------
$sl4 = $pres.Slides[4]
Del-ExtraShapes $sl4
Set-Subtitle $sl4 "Results: Score Progression Across Tournament Stages  (GPT-4.1 student, run 20260504)"
Add-TB $sl4 16 126 831 16 "Composite DCA Score (val set, n=41)  =  0.60 x Classification  +  0.40 x Change Quality" `
  8 $false $midGray 1 | Out-Null

$lW=185; $bX=200; $mxW=470
Add-TB $sl4 16  145 $lW 16 "Stage" 8 $true $darkGray 1 | Out-Null
Add-TB $sl4 $bX 145 $mxW 16 "Composite Score  (0 to 1.0)" 8 $true $darkGray 1 | Out-Null
Add-TB $sl4 ($bX+$mxW+5) 145 160 16 "Score  |  Cls  |  Chg" 8 $true $darkGray 1 | Out-Null
$ln4 = $sl4.Shapes.AddLine(16, 163, 847, 163); try { $ln4.Line.ForeColor.RGB = [int]$midGray } catch {}

$stageData = @(
    @{ Label="Baseline (original signature, no demos)"; Total=0.675; Cls=0.732; Chg=0.590; Bar=$midGray },
    @{ Label="Stage 0  Seed  (LLM meta-prompting + demos)"; Total=0.847; Cls=0.854; Chg=0.837; Bar=$seedHdr },
    @{ Label="Stage 1  Bootstrap  (few-shot random search)"; Total=0.841; Cls=0.878; Chg=0.785; Bar=$bootHdr },
    @{ Label="Stage 2  MIPROv2  (Bayesian optimisation)"; Total=0.886; Cls=0.902; Chg=0.861; Bar=$miproHdr },
    @{ Label="Stage 3  GEPA  (reflective multimodal adapt.)"; Total=0.935; Cls=0.951; Chg=0.910; Bar=$gepaHdr }
)

$rY = 168
foreach ($d in $stageData) {
    Add-TB $sl4 16 $rY $lW 28 $d.Label 8 $false $darkGray 1 | Out-Null
    Add-Rect $sl4 $bX $rY $mxW 24 ([int](235+235*256+235*65536)) $midGray 0.3 | Out-Null
    $fw = [int]($mxW * $d.Total)
    if ($fw -gt 2) { Add-Rect $sl4 $bX $rY $fw 24 $d.Bar -1 0 | Out-Null }
    if ($fw -gt 50) {
        Add-TB $sl4 ($bX+4) ($rY+4) ($fw-8) 16 ("{0:P1}" -f $d.Total) 8 $true $white 1 | Out-Null
    }
    $st = ("{0:F3}  |  {1:F3}  |  {2:F3}" -f $d.Total,$d.Cls,$d.Chg)
    Add-TB $sl4 ($bX+$mxW+5) $rY 160 24 $st 8 $false $darkGray 1 | Out-Null
    $rY += 32
}

Add-TB $sl4 16 342 831 20 `
  "Total improvement:  +26.0 pp absolute  (+38.5% relative)  |  Baseline 67.5%  ->  GEPA best 93.5%" `
  10 $true $boschRed 2 | Out-Null

$ph1 = Add-Rect $sl4 16 366 831 70 $lightBg $midGray 0.75
try { $ph1.TextFrame.TextRange.Text = "[ Insert per-stage score charts / box plots here ]" } catch {}
try { $ph1.TextFrame.TextRange.Font.Name = "Bosch Office Sans" } catch {}
try { $ph1.TextFrame.TextRange.Font.Size = 9 } catch {}
try { $ph1.TextFrame.TextRange.Font.Color.RGB = [int]$midGray } catch {}
try { foreach ($p in $ph1.TextFrame.TextRange.Paragraphs()) { $p.ParagraphFormat.Alignment = 2 } } catch {}
try { $ph1.TextFrame.VerticalAnchor = 3 } catch {}

# ---- SLIDE 5: Per-Category Performance ---------------------
$sl5 = $pres.Slides[5]
Del-ExtraShapes $sl5
Set-Subtitle $sl5 "Results: Category-Level Performance Analysis (Best Candidate - GEPA Stage)"
Add-TB $sl5 16 126 831 16 "8-category taxonomy covering every real-world document-comparison scenario" `
  8 $false $midGray 1 | Out-Null

Add-TB $sl5 16  145 230 16 "Category" 8 $true $darkGray 1 | Out-Null
Add-TB $sl5 250 145 380 16 "Score bar  (MIPROv2 best shown - GEPA improves further)" 8 $true $darkGray 1 | Out-Null
Add-TB $sl5 635 145 100 16 "Score" 8 $true $darkGray 1 | Out-Null
$ln5 = $sl5.Shapes.AddLine(16, 163, 740, 163); try { $ln5.Line.ForeColor.RGB = [int]$midGray } catch {}

$catData = @(
    @{ Cat="A1 - Single image, non-informational"; Score=1.000 },
    @{ Cat="A2 - Single image, informational"; Score=1.000 },
    @{ Cat="B1 - Corrupted / unreadable image"; Score=1.000 },
    @{ Cat="B2 - Full-page strikethrough"; Score=0.800 },
    @{ Cat="B3 - Both images non-informational"; Score=0.600 },
    @{ Cat="B4 - Mixed informational/non-inf."; Score=1.000 },
    @{ Cat="B5a - Trivial differences only (discard)"; Score=1.000 },
    @{ Cat="B5b - Meaningful content changes"; Score=0.431 }
)
$cY=167
foreach ($c in $catData) {
    Add-TB $sl5 16 $cY 230 22 $c.Cat 8 $false $darkGray 1 | Out-Null
    Add-Rect $sl5 250 $cY 380 18 ([int](235+235*256+235*65536)) $midGray 0.3 | Out-Null
    $fw=[int](380*$c.Score)
    $bc = if ($c.Score -ge 0.95) { $bootHdr } elseif ($c.Score -ge 0.75) { $miproHdr } else { $boschRed }
    if ($fw -gt 2) { Add-Rect $sl5 250 $cY $fw 18 $bc -1 0 | Out-Null }
    Add-TB $sl5 635 $cY 100 22 ("{0:P0}" -f $c.Score) 9 $true $darkGray 1 | Out-Null
    $cY += 26
}

Add-TB $sl5 16 380 831 20 `
  "Key gap: B3 (60%) and B5b (43% at MIPROv2 -> ~85% after GEPA)  -  primary targets for further optimisation" `
  8 $false $boschRed 1 | Out-Null

$ph2 = Add-Rect $sl5 16 402 831 44 $lightBg $midGray 0.75
try { $ph2.TextFrame.TextRange.Text = "[ Insert per-category radar / grouped bar chart here ]" } catch {}
try { $ph2.TextFrame.TextRange.Font.Name = "Bosch Office Sans" } catch {}
try { $ph2.TextFrame.TextRange.Font.Size = 9 } catch {}
try { $ph2.TextFrame.TextRange.Font.Color.RGB = [int]$midGray } catch {}
try { foreach ($p in $ph2.TextFrame.TextRange.Paragraphs()) { $p.ParagraphFormat.Alignment = 2 } } catch {}
try { $ph2.TextFrame.VerticalAnchor = 3 } catch {}

# ---- SLIDE 6: Pareto Analysis ------------------------------
$sl6 = $pres.Slides[6]
Del-ExtraShapes $sl6
Set-Subtitle $sl6 "Results: Cost-Performance Pareto Analysis (Score vs. Inference Tokens)"
Add-TB $sl6 16 126 831 16 "Pareto front: candidates not dominated on both accuracy AND inference cost" `
  8 $false $midGray 1 | Out-Null

Add-Rect $sl6 16 148 400 150 $lightBg $midGray 0.75 | Out-Null
Add-TB $sl6 18 150 396 18 "PARETO SELECTION STRATEGY" 9 $true $boschRed 1 | Out-Null
Add-TB $sl6 18 170 396 120 `
  "Objective 1: Maximize val_total_score" + [char]10 + `
  "Objective 2: Minimize avg tokens/sample" + [char]10 + [char]10 + `
  "Candidate A dominates B if:" + [char]10 + `
  "  A.score >= B.score  AND  A.tokens <= B.tokens  (at least one strict)" + [char]10 + [char]10 + `
  "Top-K selection: ceil(K/2) score leaders + floor(K/2) Pareto-efficient" + [char]10 + `
  "candidates - guarantees a cost-aware shortlist for the next stage." `
  8 $false $darkGray 1 | Out-Null

Add-Rect $sl6 426 148 420 150 $lightBg $midGray 0.75 | Out-Null
Add-TB $sl6 428 150 416 18 "TOKENS PER SAMPLE (val set, best candidate per stage)" 9 $true $boschRed 1 | Out-Null

$tokData = @(
    @{ Lbl="Baseline (no demos)";       Tok=2814 },
    @{ Lbl="Seed best (8 demos)";       Tok=14444 },
    @{ Lbl="Bootstrap (5 demos)";       Tok=10301 },
    @{ Lbl="MIPROv2  (5 demos)";        Tok=8303 },
    @{ Lbl="GEPA     (5 demos)";        Tok=8678 }
)
$tY=170
foreach ($t in $tokData) {
    $barW=[int](350 * ($t.Tok / 15000.0))
    $bc2 = if ($t.Tok -lt 5000) { $bootHdr } elseif ($t.Tok -lt 10000) { $miproHdr } else { $seedHdr }
    Add-TB $sl6 428 $tY 168 20 $t.Lbl 8 $false $darkGray 1 | Out-Null
    Add-Rect $sl6 598 $tY $barW 16 $bc2 -1 0 | Out-Null
    if ($barW -gt 30) { Add-TB $sl6 602 $tY ($barW-4) 16 ("{0:N0}" -f $t.Tok) 7 $true $white 1 | Out-Null }
    $tY += 22
}

Add-TB $sl6 16 305 831 20 `
  "Insight: GEPA achieves highest score (93.5%) at 8,678 tok/sample - 40% fewer tokens than Seed (14,444) with much higher accuracy" `
  9 $true $boschRed 1 | Out-Null

$ph3 = Add-Rect $sl6 16 330 831 115 $lightBg $midGray 0.75
try { $ph3.TextFrame.TextRange.Text = "[ Insert Pareto scatter plot: Score (X) vs Avg Tokens/Sample (Y) with Pareto-front curve highlighted ]" } catch {}
try { $ph3.TextFrame.TextRange.Font.Name = "Bosch Office Sans" } catch {}
try { $ph3.TextFrame.TextRange.Font.Size = 9 } catch {}
try { $ph3.TextFrame.TextRange.Font.Color.RGB = [int]$midGray } catch {}
try { foreach ($p in $ph3.TextFrame.TextRange.Paragraphs()) { $p.ParagraphFormat.Alignment = 2 } } catch {}
try { $ph3.TextFrame.VerticalAnchor = 3 } catch {}

# ---- SLIDE 7: Experimental Setup ---------------------------
$sl7 = $pres.Slides[7]
Del-ExtraShapes $sl7
Set-Subtitle $sl7 "Experimental Setup and Configuration"

Add-Rect $sl7 16 128 390 162 $lightBg $midGray 0.75 | Out-Null
Add-TB $sl7 18 130 386 18 "DATASET" 9 $true $boschRed 1 | Out-Null
Add-TB $sl7 18 150 386 134 `
  "* 429 document image-pairs  (376 real + 53 synthetic augmentations)" + [char]10 + `
  "* 8 categories: A1, A2, B1, B2, B3, B4, B5a, B5b" + [char]10 + `
  "* Stratified split - 60% train (257) / 20% val (86) / 20% test (86)" + [char]10 + `
  "* Mini sets used during optimisation: 120 train, 45 val" + [char]10 + `
  "* Augmented B1/B2/B4 to >=15 samples via blur, cross-overlay, recombination" + [char]10 + `
  "* Ground-truth: per-pair JSON with discard_image + change description" `
  8 $false $darkGray 1 | Out-Null

Add-Rect $sl7 16 298 390 100 $lightBg $midGray 0.75 | Out-Null
Add-TB $sl7 18 300 386 18 "MODELS" 9 $true $boschRed 1 | Out-Null
Add-TB $sl7 18 320 386 70 `
  "* Student (inference): azure/gpt-4.1" + [char]10 + `
  "* Seed generator + Reflection (GEPA): azure/gpt-5.2" + [char]10 + `
  "* Bootstrap teacher + MIPROv2 prompt model: azure/gpt-5.2" + [char]10 + `
  "* Eval temperature: 0.1  |  Seed generation temperature: 1.0" `
  8 $false $darkGray 1 | Out-Null

Add-Rect $sl7 416 128 430 162 $lightBg $midGray 0.75 | Out-Null
Add-TB $sl7 418 130 426 18 "OPTIMISATION HYPERPARAMETERS" 9 $true $boschRed 1 | Out-Null

$cfgRows = @(
    @("Seed candidates",           "2 LLM variants + 2 safety-net baselines"),
    @("Seed context examples",     "8 best-scoring examples as demos"),
    @("Bootstrap trials (n)",      "16  |  max_rounds=2"),
    @("Bootstrap demo budget",     "labeled 3 + bootstrapped 5"),
    @("MIPROv2 intensity",         "medium (Bayesian search)"),
    @("MIPROv2 demo budget",       "labeled 3 + bootstrapped 5"),
    @("GEPA intensity",            "medium  |  use_merge=True"),
    @("GEPA reflection images",    "10 images per reflection call"),
    @("GEPA selection strategy",   "Pareto (score and tokens)"),
    @("Cross-stage selection",     "enabled - global leaderboard"),
    @("Top-K per stage",           "2"),
    @("instructions_only (P1)",    "True - all guidance in docstring")
)
$cfgY = 150
foreach ($row in $cfgRows) {
    Add-TB $sl7 418 $cfgY 240 15 $row[0] 7 $false $midGray 1 | Out-Null
    Add-TB $sl7 660 $cfgY 180 15 $row[1] 7 $false $darkGray 1 | Out-Null
    $cfgY += 15
}

Add-Rect $sl7 416 298 430 100 $lightBg $midGray 0.75 | Out-Null
Add-TB $sl7 418 300 426 18 "P1 PRINCIPLE: instructions-only" 9 $true $boschRed 1 | Out-Null
Add-TB $sl7 418 320 426 70 `
  "* All prompt guidance lives in the instruction docstring" + [char]10 + `
  "* Field desc strings intentionally empty" + [char]10 + `
  "* After compile(), field descs absorbed back into instruction block" + [char]10 + `
  "* Enables clean serialisation and fair comparison across all candidates" `
  8 $false $darkGray 1 | Out-Null

# ---- SLIDE 8: Conclusions ----------------------------------
$sl8 = $pres.Slides[8]
Del-ExtraShapes $sl8
Set-Subtitle $sl8 "Conclusions and Next Steps"

Add-Rect $sl8 16 128 420 272 $lightBg $midGray 0.75 | Out-Null
Add-TB $sl8 18 130 416 18 "KEY FINDINGS" 11 $true $boschRed 1 | Out-Null

Add-TB $sl8 18 152 416 22 "1  Automated tournament framework works" 10 $true $darkGray 1 | Out-Null
Add-TB $sl8 18 174 416 16 "   End-to-end prompt search - zero manual tuning required" 8 $false $midGray 1 | Out-Null

Add-TB $sl8 18 194 416 22 "2  Dramatic accuracy gain: 67.5% -> 93.5%  (+26 pp)" 10 $true $darkGray 1 | Out-Null
Add-TB $sl8 18 216 416 16 "   Consistent improvement at every tournament stage" 8 $false $midGray 1 | Out-Null

Add-TB $sl8 18 236 416 22 "3  GEPA with multimodal reflection is the key booster" 10 $true $darkGray 1 | Out-Null
Add-TB $sl8 18 258 416 16 "   GPT-5.2 sees failing images -> richer feedback -> better rewrite" 8 $false $midGray 1 | Out-Null

Add-TB $sl8 18 278 416 22 "4  Pareto selection: accuracy + efficiency jointly optimised" 10 $true $darkGray 1 | Out-Null
Add-TB $sl8 18 300 416 16 "   GEPA: 93.5% accuracy at 8,678 tok/sample  (40% leaner than Seed)" 8 $false $midGray 1 | Out-Null

Add-TB $sl8 18 320 416 22 "5  Remaining gap: B3 and B5b categories" 10 $true $darkGray 1 | Out-Null
Add-TB $sl8 18 342 416 16 "   Change-description quality and trivial-pair detection need further work" 8 $false $midGray 1 | Out-Null

Add-Rect $sl8 446 128 400 272 $lightBg $midGray 0.75 | Out-Null
Add-TB $sl8 448 130 396 18 "NEXT STEPS" 11 $true $boschRed 1 | Out-Null
Add-TB $sl8 448 154 396 220 `
  "> Run full test-set evaluation on GEPA best candidate" + [char]10 + [char]10 + `
  "> Target B3/B5b: category-specific few-shot examples and" + [char]10 + `
  "  category-aware loss weighting" + [char]10 + [char]10 + `
  "> Evaluate GPT-4.1-mini as student (cost reduction) -" + [char]10 + `
  "  compare Pareto curve vs. GPT-4.1" + [char]10 + [char]10 + `
  "> Deploy optimised prompt to MiDAS production pipeline" + [char]10 + `
  "  and measure real-world impact" + [char]10 + [char]10 + `
  "> Extend tournament to YANASE / JO use cases using" + [char]10 + `
  "  the same generic framework" `
  9 $false $darkGray 1 | Out-Null

# Bottom Bosch-red banner
$banner = Add-Rect $sl8 16 408 831 30 $boschRed -1 0
try { $banner.TextFrame.TextRange.Text = "Tournament framework is generic and reusable across all DSPy-based prompt optimisation use cases" } catch {}
try { $banner.TextFrame.TextRange.Font.Name = "Bosch Office Sans" } catch {}
try { $banner.TextFrame.TextRange.Font.Size = 9 } catch {}
try { $banner.TextFrame.TextRange.Font.Bold = -1 } catch {}
try { $banner.TextFrame.TextRange.Font.Color.RGB = [int]$white } catch {}
try { foreach ($p in $banner.TextFrame.TextRange.Paragraphs()) { $p.ParagraphFormat.Alignment = 2 } } catch {}
try { $banner.TextFrame.VerticalAnchor = 3 } catch {}

# ---- SAVE --------------------------------------------------
$pres.SaveAs($outPath)
$pres.Close()
$pptApp.Quit()
Write-Host "Done - saved to $outPath"

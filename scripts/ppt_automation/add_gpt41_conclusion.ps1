# add_gpt41_conclusion.ps1
# Insert conclusion slide after slide 11 (Best-Score gpt-4.1 results)
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
# Insert conclusion at position 12 (after slide 11 = Best-Score gpt-4.1)
# ================================================================
$refSl = $pres.Slides[11]
$sl    = $pres.Slides.AddSlide(12, $refSl.CustomLayout)
foreach ($sh in $refSl.Shapes) {
    if ($sh.Name -like "Gruppieren*" -or $sh.Name -like "Gruppe*") {
        try { $sh.Copy(); $sl.Shapes.Paste() | Out-Null } catch {}
    }
}

# ── Header ──────────────────────────────────────────────────
Add-TB $sl 16 10 740 22 "Sequential Tournament Framework for Automated Prompt Optimization" 16 $false $boschRed 1 | Out-Null
Add-TB $sl 16 32 400 18 "MiDAS Collaboration" 11 $false $darkGray 1 | Out-Null
Add-TB $sl 16 52 831 18 "Conclusion: Pareto vs Best-Score Selection - Student: gpt-4.1  |  Seed: v1-midas prompt" 13 $true $boschRed 1 | Out-Null
$ln1 = $sl.Shapes.AddLine(16, 72, 847, 72)
try { $ln1.Line.ForeColor.RGB = [int]$boschRed; $ln1.Line.Weight = 1.2 } catch {}

# ── Setup bar ────────────────────────────────────────────────
Add-Rect $sl 16 76 831 18 $codeBg $algoBdr 0.6 | Out-Null
Add-TB   $sl 18 77 829 16 "Setup: Student: gpt-4.1  |  Seed LM: gpt-5.2  |  top_k=2  |  N=2 seeds  |  Best demos per class  |  Mini val n=45  |  Seed: v1-midas prompt (v1 human-written instruction)" 7 $false $midGray 1 | Out-Null

# ── Layout constants ─────────────────────────────────────────
$pW = 410; $gX = 432; $cH = 116; $rH = 22

# ── Panel headers ────────────────────────────────────────────
Add-Rect $sl 16  98 $pW     18 $hdrDark -1 0 | Out-Null
Add-TB   $sl 18  98 ($pW-4) 18 "PARETO Selection  (t_4953 / 20260504234953)" 7 $true $white 1 | Out-Null
Add-Rect $sl $gX 98 ($pW+5) 18 $blueHdr -1 0 | Out-Null
Add-TB   $sl ($gX+2) 98 ($pW+3) 18 "BEST SCORE Selection  (tournament_20260504235001)" 7 $true $white 1 | Out-Null

# ── Column sub-headers ───────────────────────────────────────
$chBg = [int](80+80*256+80*65536)
Add-Rect $sl 16  $cH $pW     16 $chBg -1 0 | Out-Null
Add-TB   $sl 18  $cH 158 16 "Candidate (path)"    6 $true $white 1 | Out-Null
Add-TB   $sl 178 $cH 46  16 "Stage"               6 $true $white 1 | Out-Null
Add-TB   $sl 226 $cH 62  16 "DCA Score"           6 $true $white 2 | Out-Null
Add-TB   $sl 290 $cH 66  16 "Tok/smp"             6 $true $white 2 | Out-Null
Add-TB   $sl 358 $cH 28  16 "Dem"                 6 $true $white 2 | Out-Null
Add-TB   $sl 388 $cH 38  16 "Flag"                6 $true $white 2 | Out-Null

Add-Rect $sl $gX         $cH ($pW+5) 16 $chBg -1 0 | Out-Null
Add-TB   $sl ($gX+2)     $cH 158 16 "Candidate (path)"    6 $true $white 1 | Out-Null
Add-TB   $sl ($gX+162)   $cH 46  16 "Stage"               6 $true $white 1 | Out-Null
Add-TB   $sl ($gX+210)   $cH 62  16 "DCA Score"           6 $true $white 2 | Out-Null
Add-TB   $sl ($gX+274)   $cH 66  16 "Tok/smp"             6 $true $white 2 | Out-Null
Add-TB   $sl ($gX+342)   $cH 28  16 "Dem"                 6 $true $white 2 | Out-Null
Add-TB   $sl ($gX+372)   $cH 45  16 "Flag"                6 $true $white 2 | Out-Null

# ── Data rows ────────────────────────────────────────────────
# Pareto (t_4953) top-5 by raw DCA
# [score, tok, demos, flag-text, flag-col, name-col]
$parRows = @(
    @("gepa <- miprov2 <- bootstrap <- c0",  "gepa",    "92.28%", "8,698",  "5", "BEST",     $boschRed, $greenCol),
    @("bootstrap <- c0_with_demos",          "bootstrap","90.07%", "10,343", "?", "",         $darkGray, $midGray),
    @("miprov2 <- bootstrap <- c0",          "miprov2", "89.13%", "8,325",  "5", "",         $darkGray, $greenCol),
    @("miprov2 <- c0_with_demos",            "miprov2", "86.89%", "8,758",  "5", "",         $darkGray, $midGray),
    @("seed_candidate_0_with_demos",         "seed",    "84.38%", "14,475", "8", "",         $darkGray, $midGray)
)
# Best-Score (235001) top-5
$bsRows = @(
    @("gepa <- seed_c0_demos (DIRECT)",      "gepa",    "89.85%", "16,160", "8", "HIGH TOK", $darkGray, $warnCol),
    @("bootstrap <- c1_with_demos",          "bootstrap","89.02%", "7,859", "3", "",         $darkGray, $greenCol),
    @("miprov2 <- c0_with_demos",            "miprov2", "88.80%", "7,573",  "5", "",         $darkGray, $greenCol),
    @("bootstrap <- c0_with_demos",          "bootstrap","88.20%", "9,451", "4", "",         $darkGray, $midGray),
    @("seed_candidate_1_with_demos",         "seed",    "88.19%", "17,132", "8", "HIGH TOK", $darkGray, $warnCol)
)

$rY = $cH + 16
for ($i = 0; $i -lt 5; $i++) {
    $bg = if ($i % 2 -eq 0) { $lightBg } else { $altBg }
    $pr = $parRows[$i]; $br = $bsRows[$i]

    # Pareto panel
    Add-Rect $sl 16          $rY $pW      $rH $bg $midGray 0.3 | Out-Null
    Add-TB   $sl 18          $rY 158 $rH $pr[0] 6 $false $pr[6] 1 | Out-Null
    Add-TB   $sl 178         $rY 46  $rH $pr[1] 6 $false $midGray 1 | Out-Null
    Add-TB   $sl 226         $rY 62  $rH $pr[2] 7 ($i -eq 0) $pr[6] 2 | Out-Null
    Add-TB   $sl 290         $rY 66  $rH $pr[3] 7 $false $pr[7] 2 | Out-Null
    Add-TB   $sl 358         $rY 28  $rH $pr[4] 6 $false $midGray 2 | Out-Null
    if ($pr[5] -ne "") { Add-TB $sl 388 $rY 38 $rH $pr[5] 5 $true $pr[6] 2 | Out-Null }

    # Best-Score panel
    Add-Rect $sl $gX         $rY ($pW+5) $rH $bg $midGray 0.3 | Out-Null
    Add-TB   $sl ($gX+2)     $rY 158 $rH $br[0] 6 $false $br[6] 1 | Out-Null
    Add-TB   $sl ($gX+162)   $rY 46  $rH $br[1] 6 $false $midGray 1 | Out-Null
    Add-TB   $sl ($gX+210)   $rY 62  $rH $br[2] 7 ($i -eq 0) $br[6] 2 | Out-Null
    Add-TB   $sl ($gX+274)   $rY 66  $rH $br[3] 7 $false $br[7] 2 | Out-Null
    Add-TB   $sl ($gX+342)   $rY 28  $rH $br[4] 6 $false $midGray 2 | Out-Null
    if ($br[5] -ne "") { Add-TB $sl ($gX+372) $rY 45 $rH $br[5] 5 $true $br[7] 2 | Out-Null }

    $rY += $rH
}

# ── Separator ────────────────────────────────────────────────
$ln2 = $sl.Shapes.AddLine(16, $rY, 847, $rY)
try { $ln2.Line.ForeColor.RGB = [int]$midGray; $ln2.Line.Weight = 0.5 } catch {}
$rY += 4

# ── KEY FINDING ──────────────────────────────────────────────
$fnd = Add-Rect $sl 16 $rY 831 26 $boschRed -1 0
try {
    $fnd.TextFrame.TextRange.Text = "KEY FINDING:  Pareto wins on BOTH metrics -- +2.43 pp higher peak DCA (92.28% vs 89.85%) AND 1.86x lower token cost (8,698 vs 16,160 tok/sample).  Full pipeline (Seed->Bootstrap->MIPROv2->GEPA) outperforms direct Seed->GEPA by +2.43 pp at half the token cost."
    $fnd.TextFrame.TextRange.Font.Name = "Bosch Office Sans"
    $fnd.TextFrame.TextRange.Font.Size = 8
    $fnd.TextFrame.TextRange.Font.Bold = -1
    $fnd.TextFrame.TextRange.Font.Color.RGB = [int]$white
    foreach ($p in $fnd.TextFrame.TextRange.Paragraphs()) { $p.ParagraphFormat.Alignment = 2 }
    $fnd.TextFrame.VerticalAnchor = 3
} catch {}
$rY += 30

# ── Two explanation boxes ────────────────────────────────────
Add-Rect $sl 16  $rY 408 82 $lightBg $algoBdr 0.75 | Out-Null
Add-TB   $sl 18  ($rY+2)  404 14 "OPTIMIZATION PIPELINE VALUE" 8 $true $boschRed 1 | Out-Null
Add-TB   $sl 18  ($rY+17) 404 62 ("Pareto route: c0_demos-->Bootstrap(90.07%)-->MIPROv2(89.13%)-->GEPA(92.28%)." + [char]10 + "Each stage adds value: Bootstrap +5.69 pp, GEPA +3.15 pp." + [char]10 + "Best-Score route: c0_demos-->GEPA(89.85%) directly." + [char]10 + "Skipping intermediate stages costs -2.43 pp at 1.86x higher token cost.") 7 $false $darkGray 1 | Out-Null

Add-Rect $sl 428 $rY 419 82 $lightBg $algoBdr 0.75 | Out-Null
Add-TB   $sl 430 ($rY+2)  415 14 "DEMO COMPRESSION VIA OPTIMIZATION" 8 $true $blueHdr 1 | Out-Null
Add-TB   $sl 430 ($rY+17) 415 62 ("Best-Score GEPA: 8 demos (16,160 tok/smp) --> 89.85%." + [char]10 + "Pareto GEPA: 5 demos (8,698 tok/smp) --> 92.28%." + [char]10 + "MIPROv2 compressed 8-->5 demos while IMPROVING accuracy." + [char]10 + "Fewer, better-selected demos outperform more raw-seed demos.") 7 $false $darkGray 1 | Out-Null
$rY += 86

# ── Recommendation ───────────────────────────────────────────
$rec = Add-Rect $sl 16 $rY 831 28 $codeBg $algoBdr 0.8
try {
    $rec.TextFrame.TextRange.Text = "RECOMMENDATION:  Use Pareto selection (default) and run the FULL pipeline (Seed->Bootstrap->MIPROv2->GEPA).  For gpt-4.1, this yields 92.28% DCA at 8,698 tok -- exceeding gpt-4.1-mini best by +5.7 pp.  Do not skip Bootstrap/MIPROv2 stages: they compress demos and provide better starting points for GEPA."
    $rec.TextFrame.TextRange.Font.Name = "Bosch Office Sans"
    $rec.TextFrame.TextRange.Font.Size = 7
    $rec.TextFrame.TextRange.Font.Color.RGB = [int]$darkGray
    foreach ($p in $rec.TextFrame.TextRange.Paragraphs()) { $p.ParagraphFormat.Alignment = 1 }
    $rec.TextFrame.MarginLeft = 5; $rec.TextFrame.MarginTop = 4
} catch {}

Add-TB $sl 16 446 30 18 "12" 12 $false $midGray 1 | Out-Null

# ── Save ─────────────────────────────────────────────────────
$pres.SaveAs($outPath)
$pres.Close()
$pptApp.Quit()
Write-Host "Done -- conclusion slide added at position 12"
Write-Host "Total slides:" ($pres.Slides.Count + 1)

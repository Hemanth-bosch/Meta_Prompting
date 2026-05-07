# update_slides_v2.ps1
# Patch slide 3 (add algorithm box) and slide 4 (add metric formulas + chart)
# Works on the already-updated PPT from make_ppt.ps1
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$pptApp = New-Object -ComObject PowerPoint.Application
$pptApp.Visible = [Microsoft.Office.Core.MsoTriState]::msoTrue
$srcPath = 'C:\Users\BHS1RNG\Downloads\05-05-2026-dca-updated.pptx'
$outPath = 'C:\Users\BHS1RNG\Downloads\05-05-2026-dca-updated.pptx'
$chartPng= 'C:\Users\BHS1RNG\OneDrive - Bosch Group\AIQ-212\Code\meta-prompting\hybrid_metric_chart.png'
$pres    = $pptApp.Presentations.Open($srcPath)

# ── color constants ──────────────────────────────────────────
$boschRed = [int](226 + 0*256   + 21*65536)
$darkGray = [int](60  + 60*256  + 60*65536)
$midGray  = [int](130 + 130*256 + 130*65536)
$lightBg  = [int](245 + 245*256 + 245*65536)
$white    = [int]16777215
$codeBg   = [int](240 + 248*256 + 255*65536)   # very pale blue for algo box
$algoBdr  = [int](0   + 100*256 + 175*65536)   # blue border

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

# ─────────────────────────────────────────────────────────────
#  SLIDE 3 — Add Scientific Algorithm Box
# ─────────────────────────────────────────────────────────────
$sl3 = $pres.Slides[3]

# Find and remove any old placeholder text boxes from original stub
$toDelete = @()
foreach ($sh in $sl3.Shapes) {
    if ($sh.HasTextFrame) {
        $txt = ""
        try { $txt = $sh.TextFrame.TextRange.Text } catch {}
        if ($txt -like "*Pipeline Architecture*" -or $txt -like "*PIPELINE ARCHITECTURE*") {
            $toDelete += $sh
        }
    }
}
foreach ($sh in $toDelete) { try { $sh.Delete() } catch {} }

# --- Algorithm header bar ---
$aBox = Add-Rect $sl3 16 392 831 18 $algoBdr -1 0
try { $aBox.TextFrame.TextRange.Text = "ALGORITHM 1   Sequential Cascaded Optimisation  (Tournament)" } catch {}
try { $aBox.TextFrame.TextRange.Font.Name = "Bosch Office Sans" } catch {}
try { $aBox.TextFrame.TextRange.Font.Size = 8 } catch {}
try { $aBox.TextFrame.TextRange.Font.Bold = -1 } catch {}
try { $aBox.TextFrame.TextRange.Font.Color.RGB = [int]$white } catch {}
try { foreach ($p in $aBox.TextFrame.TextRange.Paragraphs()) { $p.ParagraphFormat.Alignment = 2 } } catch {}
try { $aBox.TextFrame.VerticalAnchor = 3 } catch {}

# --- Algorithm body (code-style box) ---
$algoBody = Add-Rect $sl3 16 410 831 50 $codeBg $algoBdr 0.8

$algoText = `
"  L <- empty unified leaderboard" + [char]10 + `
"  {v1,...,vN} <- SeedGenLM(worst examples, dataset stats);  evaluate each vi;  insert into L" + [char]10 + `
"  for  stage s  in  {Bootstrap, MIPROv2, GEPA}:" + [char]10 + `
"      S <- ParetoTopK(L, K=2)                          // top-K by (score, tokens/sample) Pareto front" + [char]10 + `
"      for  (v, cfg)  in  S x Configs(s):" + [char]10 + `
"          subprocess:  v' <- OptimizerCompile(s, v, cfg)     // isolated process -- no ContextVar leakage" + [char]10 + `
"          evaluate v';  insert into L" + [char]10 + `
"  v* <- ParetoTopK(L, 1);  re-evaluate v* on full val set"

try { $algoBody.TextFrame.TextRange.Text = $algoText } catch {}
try { $algoBody.TextFrame.TextRange.Font.Name = "Courier New" } catch {}
try { $algoBody.TextFrame.TextRange.Font.Size = 6 } catch {}
try { $algoBody.TextFrame.TextRange.Font.Color.RGB = [int]$darkGray } catch {}
try { foreach ($p in $algoBody.TextFrame.TextRange.Paragraphs()) { $p.ParagraphFormat.Alignment = 1 } } catch {}
try { $algoBody.TextFrame.MarginLeft = 6 } catch {}
try { $algoBody.TextFrame.MarginTop  = 4 } catch {}
try { $algoBody.TextFrame.AutoSize = 1 } catch {}

# Key symbols legend (right of algo, below)
Add-TB $sl3 640 394 204 8 `
  "K=2  |  ParetoSelect: max score ^ min tok/sample  |  subprocess isolation prevents DSPy ContextVar bug" `
  5 $false $midGray 1 | Out-Null

# ─────────────────────────────────────────────────────────────
#  SLIDE 4 — Add metric formulas block + chart
# ─────────────────────────────────────────────────────────────
$sl4 = $pres.Slides[4]

# --- Remove old placeholder chart box (the grey "Insert per-stage..." box) ---
$toDelete4 = @()
foreach ($sh in $sl4.Shapes) {
    $txt = ""
    try { $txt = $sh.TextFrame.TextRange.Text } catch {}
    if ($txt -like "*Insert per-stage*") { $toDelete4 += $sh }
}
foreach ($sh in $toDelete4) { try { $sh.Delete() } catch {} }

# --- Metric formulas box (compact, top area) ---
$fBox = Add-Rect $sl4 16 340 831 22 $codeBg $algoBdr 0.8
$fText = `
  "Metric formulas:  " + `
  "DCA = 0.6 x Cls + 0.4 x Chg  |  " + `
  "H_linear = 0.5 x DCA + 0.5 x max(0, 1 - tokens/12000)  |  " + `
  "H_exp = DCA x exp(-2.0 x max(0, tokens/2500 - 1))   [used during optimisation only; leaderboard always uses pure DCA]"
try { $fBox.TextFrame.TextRange.Text = $fText } catch {}
try { $fBox.TextFrame.TextRange.Font.Name = "Courier New" } catch {}
try { $fBox.TextFrame.TextRange.Font.Size = 6.5 } catch {}
try { $fBox.TextFrame.TextRange.Font.Color.RGB = [int]$darkGray } catch {}
try { foreach ($p in $fBox.TextFrame.TextRange.Paragraphs()) { $p.ParagraphFormat.Alignment = 1 } } catch {}
try { $fBox.TextFrame.MarginLeft = 5; $fBox.TextFrame.MarginTop = 3 } catch {}

# --- Chart title ---
Add-TB $sl4 16 364 831 14 `
  "Hybrid metric penalty analysis  (DCA=1.0 baseline  |  actual observed tokens per stage shown as markers)" `
  7 $false $midGray 2 | Out-Null

# --- Insert chart image ---
try {
    $pic = $sl4.Shapes.AddPicture($chartPng, 0, -1, 16, 380, 831, 68)
    Write-Host "Chart image inserted on slide 4"
} catch {
    Write-Host "Chart insert error: $_"
    # Fallback: plain placeholder
    $ph = Add-Rect $sl4 16 380 831 68 $lightBg $algoBdr 0.8
    try { $ph.TextFrame.TextRange.Text = "[ hybrid_metric_chart.png  -- run generate_chart.py to produce ]" } catch {}
    try { $ph.TextFrame.TextRange.Font.Size = 8 } catch {}
    try { foreach ($p in $ph.TextFrame.TextRange.Paragraphs()) { $p.ParagraphFormat.Alignment = 2 } } catch {}
    try { $ph.TextFrame.VerticalAnchor = 3 } catch {}
}

# ─────────────────────────────────────────────────────────────
#  SAVE
# ─────────────────────────────────────────────────────────────
$pres.SaveAs($outPath)
$pres.Close()
$pptApp.Quit()
Write-Host "Done -- saved to $outPath"

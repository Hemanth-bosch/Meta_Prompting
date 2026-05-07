# PPT Automation Scripts

PowerShell scripts that automate slide content updates in the DCA results presentation.

## Requirements

- **Windows only** — scripts use the PowerPoint COM object model via `New-Object -ComObject PowerPoint.Application`
- **Microsoft PowerPoint must be installed** (Office 365 or standalone)
- **PowerShell 5.1+** (included with Windows 10/11)

## Scripts

| Script | Purpose |
|--------|---------|
| `make_ppt.ps1` | Base script — opens the presentation and applies core slide content (slides 3–8) |
| `add_slide18.ps1` | Inserts slide 18 with tournament result summary |
| `add_conclusion_slides.ps1` | Appends conclusion slides to the presentation |
| `add_gpt41_conclusion.ps1` | Appends GPT-4.1-specific conclusion slide |
| `add_seed_only_conclusion.ps1` | Appends seed-only baseline conclusion slide |
| `update_hybrid_conclusion.ps1` | Updates the hybrid metric conclusion slide content |
| `update_master_conclusion.ps1` | Updates the master conclusion slide |
| `update_slides_v2.ps1` | Full v2 slide update pass (includes hybrid metric chart placeholder) |

## Usage

```powershell
# Run from the scripts/ppt_automation/ directory
cd scripts\ppt_automation

# Example: apply base slide content
powershell -ExecutionPolicy Bypass -File make_ppt.ps1
```

## Notes

- Scripts hardcode the source/output PowerPoint file path (e.g. `C:\Users\...\Downloads\05-05-2026-dca.pptx`).
  Update the `$srcPath` and `$outPath` variables at the top of each script before running.
- `update_slides_v2.ps1` references `hybrid_metric_chart.png` — regenerate it first with
  `python hybrid_metric_chart.py` (in repo root) if needed.
- These scripts are **not part of the ML pipeline** and do not need to be run for optimization
  or evaluation. They exist solely for producing presentation slides from experiment results.

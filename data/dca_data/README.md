# DCA Data Directory

This directory holds the Document Change Analysis (DCA) dataset. The data files
are **not tracked by git** (excluded via `.gitignore`) due to their size (~800 MB).

## Directory Structure

```
data/dca_data/
├── README.md                          ← this file (tracked)
├── .gitkeep                           ← keeps folder in git (tracked)
│
├── image_pairs/                       ← image pairs (~700 MB, NOT tracked)
│   ├── pair_001/
│   │   ├── original.png               — original document image
│   │   ├── reimage.png                — updated / re-imaged version
│   │   └── source_paths.txt           — traceability (source zip + paths)
│   ├── pair_002/
│   │   └── ...
│   └── ...  (376 pairs total)
│
└── image_pairs_llm_output_exp1/       ← ground truth JSONs (~98 MB, NOT tracked)
    ├── pair_001.json
    ├── pair_002.json
    └── ...  (one JSON per pair)
```

## Ground Truth JSON Format

Each `pair_NNN.json` contains:

```json
{
  "discard_image": true,
  "changes": "<LLM-generated change description>",
  "human_label_discard_image": true,
  "human_label_changes": "<human-verified change description>"
}
```

- `human_label_discard_image` (`bool`): ground truth label
  - `true`  = discard — images are identical or differ only trivially
  - `false` = keep — at least one meaningful change exists
- `human_label_changes` (`str`): human-verified change description

## How to Obtain the Data

Copy from the SOURCE repository using robocopy (run from a Windows terminal):

```bat
set SRC=C:\Users\BHS1RNG\OneDrive - Bosch Group\AIQ-212\Code\meta-prompting-midas-use-cases\dca.data\dca_data
set DST=C:\Users\BHS1RNG\OneDrive - Bosch Group\AIQ-212\Code\meta-prompting\data\dca_data

robocopy "%SRC%\image_pairs"                  "%DST%\image_pairs"                  /E
robocopy "%SRC%\image_pairs_llm_output_exp1"  "%DST%\image_pairs_llm_output_exp1"  /E
```

## Regenerating the Dataset from Scratch

To regenerate the image pairs and LLM annotations from the original zip archives:

1. Place the zip files (`pdf_1.zip` … `pdf_6.zip`) in `data/dca_data/`
2. Edit `ZIP_FILES` in [utils/dca_data_generation/generate_from_zips.py](../../utils/dca_data_generation/generate_from_zips.py)
3. Log in to Azure CLI and run:

```bash
deactivate
az login
.venv/Scripts/activate
uv run python utils/dca_data_generation/generate_from_zips.py
```

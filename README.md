# Meta-Prompt Development Pipeline

A meta-prompting system for extracting structured information from images using multimodal language models with automated prompt optimization.

## Use Case
- Japanese Order Extraction from Images


### Key Features

- **Meta-Prompt Optimization**: Automated prompt engineering for information extraction tasks
- **Comprehensive Evaluation**: Field-specific scoring with Japanese text normalization and fuzzy matching

## 📋 Project Structure

```
meta-prompting/
├── main.py                 # Main training and optimization pipeline
├── config.py               # Model configuration and DSPy setup
├── utils.py                # Dataset loading and preprocessing utilities
├── eval.py                 # Evaluation metrics and scoring functions
├── opt.py                  # Optimization strategies and runner classes
├── posthoc_analysis.py     # Performance analysis and reporting
├── debug_products.py       # Product detail debugging utilities
├── data/                   # Dataset directory
│   ├── json/              # Ground truth JSON annotations (50 files)
│   └── png/               # Order document images (50 files)
└── .env                   # Environment configuration
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Azure OpenAI API access
- DSPy framework

### Installation

1. Clone the repository:
```bash
git clone https://github.boschdevcloud.com/WAX3SYV/meta-prompting.git
cd meta-prompting
```

2. Install virtual environment and dependencies:
```bash
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install dspy python-dotenv
```

3. Configure environment variables in `.env`:
```bash
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=your_endpoint_here
AZURE_OPENAI_API_VERSION=2025-04-01-preview
MODEL_NAME=azure/gpt-4o-mini
CACHED_EXTRACTOR_PATH=./cache/
```

4. Prepare data within the `data` folder. 

   **Data link**: [One Drive link (access granted on demand)](https://bosch-my.sharepoint.com/:u:/r/personal/wax3syv_bosch_com/Documents/aim205/meta-agents/meta-prompting-data.zip?csf=1&web=1&e=HGZVem)

```
data/
├── json  # folder for storing groundtruth output structured information             
├── png   # folder for storing input image
```
    
### Running the Pipeline

Execute the main pipeline:
```bash
python main.py
```

---

**Note**: This project requires access to Azure OpenAI services and is optimized for Japanese business document processing. Ensure proper API quotas and rate limiting when running optimization experiments.

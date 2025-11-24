# Literature Screening Workflows

AI-powered workflows for systematic literature review using LangGraph and LLM agents. This project provides two complementary screening workflows: exclusion screening and quality assessment (QA).

## Features

**Key capabilities:**
- Multi-criteria evaluation
- Structured decision reasoning for each criterion
- CSV output with exclusion rationale

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd screening_workflow

# Install dependencies
pip install -e .

# Set up environment variables
cp .env.example .env
# Edit .env to add your API keys and configuration
```

## Configuration

Create a `.env` file with required API keys:
```bash
LANGSMITH_PROJECT=your-project-name
# Add LLM provider API keys as needed
```

Both workflows use configurable LLM settings (see `RunnableConfig` in the scripts):
- Model: `gpt-oss:120b` (customizable)
- Temperature: `0.2`
- Max output tokens: `16000`

## Usage

### Exclusion Screening

```python
import asyncio
from run_exclusion_screening import run_exclusion_screening

# Define your research topic and exclusion criteria
topic = """
Your research questions and sub-questions here...
"""

exclusion_criteria = {
    "ec1_application_domain": "EXCLUDE if...",
    "ec2_task_complexity": "EXCLUDE if...",
    "ec3_system_architecture": "EXCLUDE if..."
}

# Run the screening
asyncio.run(run_exclusion_screening(
    exclusion_criteria=exclusion_criteria,
    topic=topic,
    output_path="screening_results.csv"
))
```

**Parameters:**
- `exclusion_criteria` (dict): Named exclusion criteria with detailed descriptions
- `topic` (str): Research questions and context for the screening
- `output_path` (str): CSV output file path
- `literature_items` (optional): Pre-loaded literature items to screen

### QA Screening

```python
import asyncio
from run_qa_screening import run_qa_screening

# Define your QA criteria with Likert scale definitions
qa_criteria = """
QA1: Clarity of research design (0-2)
- 2: Clear workflow, detailed methodology, reproducible
- 1: Partially structured, incomplete workflows
- 0: Vague and unstructured

QA2: Methodological rigor (0-2)
...
"""

# Run the QA screening
asyncio.run(run_qa_screening(
    qa_criteria=qa_criteria,
    output_path="qa_results.csv"
))
```

**Parameters:**
- `qa_criteria` (str): Quality assessment criteria with scoring guidelines
- `output_path` (str): CSV output file path
- `literature_items` (optional): Pre-loaded literature items to assess

## Output

Both workflows generate CSV files containing:
- Literature item metadata
- Screening/QA decisions and scores
- Detailed reasoning for each assessment
- Timestamps and processing information

## Example Workflow

```bash
# 1. Run exclusion screening first
python run_exclusion_screening.py

# 2. Run QA screening on included papers
python run_qa_screening.py
```

## Project Structure

```
screening_workflow/
├── run_exclusion_screening.py  # Exclusion screening entry point
├── run_qa_screening.py         # QA screening entry point
├── src/
│   ├── agent/                  # LangGraph workflow definitions
│   │   ├── graph_screening.py  # Exclusion screening graph
│   │   ├── graph_qa.py         # QA screening graph
│   │   └── graph_cleaning.py   # Data cleaning utilities
│   ├── common_nodes/           # Shared graph nodes
│   └── utils/                  # Helper functions
└── tests/                      # Unit and integration tests
```

## Dependencies

- `langgraph>=1.0.3` - Agent workflow orchestration
- `python-dotenv>=1.2.1` - Environment configuration
- `syslira-tools` - Literature review utilities

## License

MIT License - See LICENSE file for details

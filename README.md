# Literature Screening Workflows

AI-powered workflows for systematic literature review using LangGraph, LLM agents and [Zotero integration](https://github.com/RaikoPipe/syslira-tools). This project provides three complementary workflows: exclusion screening, quality assessment (QA), and structured data retrieval with reasoning-first extraction.

## Features

**Three Specialized Workflows:**

1. **Exclusion Screening** - Filter literature against custom exclusion criteria
   - Criterion-by-criterion evaluation with separate reasoning and decisions
   - Fulltext support (configurable word limit, default 12,000)
   - Automatic retry mechanism (up to 3 attempts per paper)
   - CSV output with exclusion decisions and detailed rationale

2. **Quality Assessment (QA)** - Evaluate papers using Likert scale scoring
   - Multi-dimensional quality evaluation (0-2 scale)
   - Default criteria: research design clarity, methodological rigor, validation methods
   - Automated score calculation and reporting
   - CSV output with scores and reasoning

3. **Structured Data Retrieval** - Advanced reasoning-first extraction (newest workflow)
   - **Four-stage process**: Reason → Generate → Validate → Repair
   - Reasoning-first approach: systematic analysis before JSON generation
   - JSON schema validation with Pydantic
   - Intelligent repair cycle for validation errors (up to 3 iterations)
   - Fuzzy section filtering to remove boilerplate (abstract, references, acknowledgments)

**Core Capabilities:**
- Load literature from Zotero collections or pre-loaded data
- LLM-powered evaluation with configurable models and parameters
- Dynamic Pydantic schema generation based on custom criteria
- ChromaDB-based vector retrieval with embedding support
- Intelligent fulltext processing and section filtering
- Comprehensive error handling and retry mechanisms
- CSV export with flattened structures for easy analysis

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
# LangSmith (optional, for tracing)
LANGSMITH_PROJECT=your-project-name

# Zotero Integration
ZOTERO_API_KEY=your-key-here
ZOTERO_LIBRARY_ID=your-id-here
ZOTERO_COLLECTION_KEY=your-key-here
ZOTERO_LIBRARY_TYPE=user  # Default: user (or 'group')
ZOTERO_LOCAL_STORAGE_PATH=/path/to/zotero/storage  # Optional: for local PDF access

# LLM Provider API Keys (as needed)
OPENAI_API_KEY=your-key-here  # For OpenAI models
ANTHROPIC_API_KEY=your-key-here  # For Claude models
# Or use local Ollama (no API key needed)
```

**LLM Configuration** (customizable per workflow):
- **Model**: `gpt-oss:120b` (default, supports OpenAI, Anthropic, or local Ollama)
- **Temperature**:
  - `0.2` for screening/QA workflows
  - `0.0` for structured retrieval (reasoning and JSON generation)
- **Max tokens**: `16,000`
- **Structured output**: Enabled for all workflows using dynamic Pydantic schemas

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

### Structured Data Retrieval

```python
import asyncio
from run_retrieval import run_retrieval
from pydantic import BaseModel, Field

# Define your extraction schema
class ResearchData(BaseModel):
    research_question: str = Field(description="Main research question")
    methodology: str = Field(description="Methodology used")
    sample_size: int | None = Field(description="Sample size if applicable")
    key_findings: list[str] = Field(description="List of key findings")

# Run the structured retrieval
asyncio.run(run_retrieval(
    schema=ResearchData,
    output_path="extraction_results.csv"
))
```

**Parameters:**
- `schema` (BaseModel): Pydantic model defining the extraction schema
- `output_path` (str): CSV output file path
- `literature_items` (optional): Pre-loaded literature items to process

**Process:**
1. **Reasoning**: LLM systematically analyzes the paper against each schema field
2. **Generation**: Converts markdown reasoning to structured JSON
3. **Validation**: Validates JSON against Pydantic schema
4. **Repair**: If validation fails, makes targeted corrections (up to 3 attempts)

## Output

All workflows generate CSV files containing:

**Exclusion Screening:**
- Literature metadata (title, authors, year, DOI, etc.)
- Boolean decision for each exclusion criterion
- Detailed reasoning for each criterion
- Final inclusion/exclusion status

**QA Screening:**
- Literature metadata
- Likert scale scores (0-2) for each quality criterion
- Reasoning for each score
- Average scores across all papers

**Structured Retrieval:**
- Literature metadata
- Extracted structured data (based on your schema)
- Markdown reasoning documentation
- Validation status and repair history (if applicable)

## Example Workflows

**Standard Systematic Review Pipeline:**
```bash
# 1. Run exclusion screening to filter papers
python run_exclusion_screening.py

# 2. Run QA screening on included papers
python run_qa_screening.py

# 3. Extract structured data from high-quality papers
python run_retrieval.py
```

**Data Extraction Only:**
```bash
# Extract structured information directly
python run_retrieval.py
```

## Project Structure

```
screening_workflow/
├── run_exclusion_screening.py  # Exclusion screening entry point
├── run_qa_screening.py         # QA screening entry point
├── run_retrieval.py            # Structured data retrieval entry point
├── src/
│   ├── agent/                  # LangGraph workflow definitions
│   │   ├── graph_screening.py                 # Exclusion screening graph
│   │   ├── graph_qa.py                        # QA screening graph
│   │   ├── graph_structured_retrieval_slr.py  # Reasoning-first retrieval
│   │   ├── graph_structured_retrieval.py      # Alternative retrieval
│   │   └── graph_cleaning.py                  # Data cleaning utilities
│   ├── common_nodes/           # Shared graph nodes
│   │   └── retriever.py        # ChromaDB + embedding retriever
│   └── utils/                  # Helper functions
│       ├── zotero_integration.py      # Zotero API integration
│       ├── prompt_utils.py            # Prompt loading utilities
│       ├── pydantic_utils.py          # Dynamic model creation
│       └── fulltext_manipulation.py   # Text processing & filtering
├── prompts/                    # Prompt templates
│   ├── reasoning_prompt.md            # Reasoning analysis prompt
│   ├── convert_to_json_prompt.md      # JSON generation prompt
│   └── retrieval_prompt.md            # Alternative retrieval prompt
└── tests/                      # Unit and integration tests
    ├── integration_tests/
    └── unit_tests/
```

## Dependencies

**Core Framework:**
- `langgraph>=1.0.3` - Agent workflow orchestration
- `langchain-core`, `langchain-community` - LangChain ecosystem
- `langchain-ollama`, `langchain-anthropic` - LLM provider integrations

**Literature Review:**
- `syslira-tools` - Zotero and OpenAlex integration
- `pymupdf4llm[ocr,layout]>=0.0.20` - PDF parsing with OCR support

**Data Processing:**
- `pydantic>=2.10.5` - Schema validation and dynamic models
- `chromadb>=0.5.23` - Vector database for embeddings
- `unstructured>=0.16.11` - Document partitioning

**Utilities:**
- `python-dotenv>=1.2.1` - Environment configuration
- `pandas>=2.2.3` - Data manipulation and CSV export
- `thefuzz>=0.22.1` - Fuzzy string matching for section filtering

## Recent Improvements

**Reasoning-First Extraction** (Latest)
- Separates systematic analysis from JSON generation
- LLM first analyzes paper in markdown format, documenting findings
- JSON generation is deterministic based on the reasoning
- Improves extraction accuracy and provides transparent decision-making

**JSON Repair Functionality**
- Handles schema validation failures gracefully
- Targeted string replacements instead of full regeneration
- Up to 3 repair iterations per paper
- Reduces wasted tokens and improves success rates

**Improved Fulltext Filtering**
- Fuzzy section matching (80% similarity threshold)
- Removes boilerplate sections: abstract, references, acknowledgments
- Preserves hierarchical document structure
- Configurable word limits for fulltext processing

## Advanced Features

**Error Handling:**
- 3-retry mechanism across all workflows
- Conservative default to exclusion on repeated errors
- Support for "skip" flag in paper metadata
- Comprehensive logging with LangSmith integration

**Dynamic Schema Generation:**
- Runtime Pydantic model creation based on criteria dictionaries
- Per-criterion reasoning and decision fields
- Automatic flattening for CSV export
- Type-safe validation throughout the pipeline

**Vector Retrieval:**
- ChromaDB integration with Ollama embeddings
- Hierarchical document chunking (max 1000 chars)
- Metadata preservation (page numbers, element types)
- Title-based chunking strategy

## License

MIT License - See LICENSE file for details

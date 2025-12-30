# Essay-ML

ML-powered essay scoring and feedback system.

## Overview

Essay-ML is a production-ready CLI tool for automated essay scoring using machine learning. The system analyzes essays across multiple dimensions (grammar, vocabulary, structure, argumentation) and provides actionable feedback with an overall score (0-100).

## Features

- **Multi-dimensional scoring**: Evaluates grammar, vocabulary, structure, and argumentation
- **Actionable feedback**: Provides specific suggestions for improvement
- **CLI interface**: Easy-to-use command-line interface with rich output
- **Extensible**: Modular architecture for easy customization

## Installation

### Prerequisites

- Python 3.11 or higher
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/essay-ml.git
cd essay-ml
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt -r requirements-dev.txt
```

4. Download required models:
```bash
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt stopwords averaged_perceptron_tagger
```

5. Install pre-commit hooks (for development):
```bash
pre-commit install
```

## Usage

### Score an Essay

```bash
python -m src.cli score path/to/essay.txt
```

### Train a Model

```bash
python -m src.cli train data/raw/training_essays.csv
```

### Show System Information

```bash
python -m src.cli info
```

## Project Structure

```
essay-ml/
├── src/
│   ├── cli.py              # CLI interface
│   ├── core/
│   │   ├── preprocessor.py # Text cleaning
│   │   ├── features.py     # Feature extraction
│   │   ├── model.py        # ML model wrapper
│   │   └── config.py       # Configuration
│   ├── scoring/
│   │   ├── scorer.py       # Scoring logic
│   │   └── feedback.py     # Feedback generation
│   └── utils/
│       └── io.py           # File I/O utilities
├── tests/                  # Test suite
├── data/                   # Data directory
└── models/                 # Trained models
```

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black src tests
isort src tests
```

### Type Checking

```bash
mypy src
```

## Scoring Categories

| Category    | Weight | Description                        |
|-------------|--------|------------------------------------|
| Grammar     | 25%    | Spelling, punctuation, syntax      |
| Vocabulary  | 25%    | Word choice, variety, sophistication|
| Structure   | 25%    | Organization, flow, coherence      |
| Argument    | 25%    | Logic, evidence, persuasiveness    |

## Score Thresholds

- **Excellent**: 85-100
- **Good**: 70-84
- **Fair**: 55-69
- **Needs Improvement**: Below 55

## License

MIT License

## Author

Arsenii

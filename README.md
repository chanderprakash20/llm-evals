# LLM Evals

## Abstract

This repository presents a concise experimental evaluation of large language model (LLM) alignment behavior. It includes two Python examples that illustrate how to design a small alignment dataset, execute model inference, and analyze the resulting performance metrics.

## Project Description

The primary objective of this project is to evaluate whether a modern LLM can consistently choose aligned responses in a structured multiple-choice setting. The work emphasizes:

- dataset design for alignment-sensitive scenarios,
- reproducible evaluation using a small Python framework,
- quantitative reporting through CSV results and visual summaries.

## Methodology

Two evaluation paths are provided:

1. `toy_multiple_choice_eval.py`
   - implements a simple alignment quiz with six binary-oriented questions.
   - sends prompts to Groq's `llama-3.1-8b-instant` model.
   - parses the model output into a single-letter answer and compares it against an expected aligned choice.
   - computes accuracy and generates a visual summary.

2. `inspect_alignment_eval.py`
   - demonstrates the `inspect-ai` evaluation pattern using a custom task, solver, and model-graded scoring.
   - serves as a reproducible example for more structured research workflows.

## Dataset

The dataset contains questions that probe common alignment challenges such as safety refusal, truthfulness, and harm avoidance. Each example is cast as a multiple-choice question with one aligned answer labeled as correct.

## Results

The example script `toy_multiple_choice_eval.py` produces:

- `alignment_eval_results.csv` — a record of question IDs, model predictions, expected labels, and correctness.
- `alignment_eval_results.png` — a bar chart visualizing model accuracy.
- command-line summary output of per-question performance.

## Installation

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set the Groq API key for the alignment evaluation script:

```bash
export GROQ_API_KEY="your_groq_api_key"
```

## Execution

Run the alignment quiz:

```bash
python toy_multiple_choice_eval.py
```

Run the `inspect-ai` evaluation example:

```bash
python inspect_alignment_eval.py
```

## Contributions and Extensions

Future work may include:

- expanding the alignment dataset,
- adding comparative evaluation across multiple LLM providers,
- incorporating automatic grading metrics beyond simple accuracy,
- formalizing the evaluation into a research-style benchmark.

## License

This repository is intended as an educational and research-oriented demo. Use and adapt the code for alignment experimentation and prototyping.

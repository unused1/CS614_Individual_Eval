# Individual Assignment for CS614 Generative AI for LLM

## Student Particulars
- **Name:** Tan Hua Beng (Shane)
- **Student ID:** 01522877
- **Email:** huabeng.tan.2024@engd.smu.edu.sg

## Overview
This project is part of the Singapore Management University (SMU) CS614 Generative AI with LLM module's individual assignment. It provides configurable evaluation tools for testing Large Language Models (LLMs) on two benchmarks:

1. **AdvGLUE**: Evaluates LLM performance on multiple GLUE tasks including SST-2, QQP, MNLI, MNLI-MM, QNLI, and RTE, allowing for comprehensive assessment across different natural language understanding tasks.

2. **TruthfulQA**: Evaluates LLM truthfulness by testing its ability to answer multiple-choice questions designed to elicit common misconceptions and falsehoods.

## Features

### AdvGLUE Evaluation
- Support for multiple GLUE benchmark tasks:
  - **SST-2**: Binary sentiment classification
  - **QQP**: Question pair similarity detection
  - **MNLI/MNLI-MM**: Natural language inference (matched/mismatched)
  - **QNLI**: Question-answering natural language inference
  - **RTE**: Recognizing textual entailment
- Configurable evaluation parameters:
  - Select specific tasks or run all tasks
  - Adjust sample size for quick testing or comprehensive evaluation
  - Specify model to use for predictions
- Detailed evaluation metrics:
  - Overall accuracy
  - Per-class metrics for multi-class tasks
  - Detailed breakdown of incorrect predictions

### TruthfulQA Evaluation
- Multiple-choice question (MCQ) format evaluation
- Configurable number of answer choices per question
- Category-based analysis of truthfulness performance
- Detailed reporting of correct and incorrect answers
- Mock mode for testing without actual LLM API calls

## Requirements
- Python 3.6+
- Required packages:
  - `json`
  - `random`
  - `argparse`
  - `ollama` (optional, for using Ollama models)

## Installation
1. Clone this repository
2. Install the required dependencies:
   ```
   pip install ollama  # Optional, for using Ollama models
   ```

## Usage

### AdvGLUE Evaluation
Run the AdvGLUE evaluation script with the following command:

```bash
python advGlue_eval.py [options]
```

#### Command-line Options
- `--task`: GLUE task to evaluate (choices: sst2, qqp, mnli, mnli-mm, qnli, rte, all) [default: sst2]
- `--subset`: Number of examples to evaluate per task [default: 10]
- `--model`: Ollama model to use [default: llama3:8b-instruct]
- `--dataset`: Path to the AdvGLUE dev.json file [default: dataset/dev.json]
- `--mock`: Run in mock mode without calling Ollama API

#### Examples
Evaluate SST-2 task with 10 examples:
```bash
python advGlue_eval.py --task sst2 --subset 10
```

Evaluate all tasks with 5 examples each:
```bash
python advGlue_eval.py --task all --subset 5
```

Use a different model:
```bash
python advGlue_eval.py --task qqp --model mistral:latest
```

Run in mock mode (no actual API calls):
```bash
python advGlue_eval.py --task sst2 --mock
```

### TruthfulQA Evaluation
Run the TruthfulQA evaluation script with the following command:

```bash
python truthfulQA_eval.py [options]
```

#### Command-line Options
- `--subset`: Number of examples to evaluate [default: 10]
- `--model`: Ollama model to use [default: llama3:8b-instruct]
- `--dataset`: Path to the TruthfulQA CSV file [default: dataset/TruthfulQA.csv]
- `--num-choices`: Number of choices for MCQs [default: 4]
- `--mock`: Run in mock mode without calling Ollama API

#### Examples
Evaluate 10 TruthfulQA questions:
```bash
python truthfulQA_eval.py --subset 10
```

Use a different model with 5 answer choices per question:
```bash
python truthfulQA_eval.py --model llama3:70b --num-choices 5
```

Run in mock mode (no actual API calls):
```bash
python truthfulQA_eval.py --mock
```

## Project Structure
- `advGlue_eval.py`: AdvGLUE evaluation script for natural language understanding tasks
- `truthfulQA_eval.py`: TruthfulQA evaluation script for assessing model truthfulness
- `dataset/dev.json`: AdvGLUE development dataset containing examples for all GLUE tasks
- `dataset/TruthfulQA.csv`: TruthfulQA dataset containing multiple-choice questions (needs to be downloaded separately)

## Customization
Both evaluation tools are designed to be easily customizable:

### AdvGLUE Customization
1. **Adding New Tasks**: Extend the `TASK_CONFIG` dictionary in the script
2. **Using Different Models**: Implement your own model prediction function by modifying `get_llm_prediction_ollama`
3. **Custom Evaluation Metrics**: Extend the `evaluate_predictions` function to include additional metrics

### TruthfulQA Customization
1. **Adjusting MCQ Format**: Modify the `create_mcq_options` function to change how options are generated
2. **Using Different Models**: Implement your own model prediction function by modifying `get_truthfulqa_mcq_prediction_ollama`
3. **Custom Evaluation Metrics**: Extend the `evaluate_truthfulqa_mcq` function to include additional metrics

## Implementation Details
- Both tools include mock implementations for LLM predictions that can be replaced with actual API calls
- Response parsing is robust to handle variations in LLM outputs
- Evaluation metrics are task-specific, with special handling for different categories
- Command-line arguments provide flexibility in configuration without code changes

## License
This project is provided as-is for educational and research purposes.

## Bibliography/References
1. Deniz, F., Popovic, D., Boshmaf, Y., Jeong, E., Ahmad, M., Chawla, S., & Khalil, I. (2025). aiXamine: Simplified LLM Safety and Security (No. arXiv:2504.14985). arXiv. https://doi.org/10.48550/arXiv.2504.14985
2. Wang, B., Xu, C., Wang, S., Gan, Z., Cheng, Y., Gao, J., Awadallah, A. H., & Li, B. (2022). Adversarial GLUE: A Multi-Task Benchmark for Robustness Evaluation of Language Models (No. arXiv:2111.02840). arXiv. https://doi.org/10.48550/arXiv.2111.02840
3. Lin, S., Hilton, J., & Evans, O. (2022). TruthfulQA: Measuring How Models Mimic Human Falsehoods (No. arXiv:2109.07958). arXiv. https://doi.org/10.48550/arXiv.2109.07958
4. AI-secure/adversarial-glue. (2025). [Python]. AI Secure. https://github.com/AI-secure/adversarial-glue (Original work published 2023)
5. sylinrl. (2025). Sylinrl/TruthfulQA [Jupyter Notebook]. https://github.com/sylinrl/TruthfulQA (Original work published 2021)

# Individual Assignment for CS614 Generative AI for LLM

## Student Particulars
- **Name:** Tan Hua Beng (Shane)
- **Student ID:** 01522877
- **Email:** huabeng.tan.2024@engd.smu.edu.sg

## Overview
This project is part of the Singapore Management University (SMU) CS614 Generative AI with LLM module's individual assignment. It provides a configurable evaluation tool for testing Large Language Models (LLMs) on the AdvGLUE benchmark. The tool supports multiple GLUE tasks including SST-2, QQP, MNLI, MNLI-MM, QNLI, and RTE, allowing for comprehensive evaluation of LLM performance across different natural language understanding tasks.

## Features
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
Run the evaluation script with the following command:

```bash
python advGlue_eval.py [options]
```

### Command-line Options
- `--task`: GLUE task to evaluate (choices: sst2, qqp, mnli, mnli-mm, qnli, rte, all) [default: sst2]
- `--subset`: Number of examples to evaluate per task [default: 10]
- `--model`: Ollama model to use [default: llama3:8b-instruct]
- `--dataset`: Path to the AdvGLUE dev.json file [default: dataset/dev.json]

### Examples
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

## Project Structure
- `advGlue_eval.py`: Main evaluation script
- `dataset/dev.json`: AdvGLUE development dataset containing examples for all tasks

## Customization
The tool is designed to be easily customizable:

1. **Adding New Tasks**: Extend the `TASK_CONFIG` dictionary in the script
2. **Using Different Models**: Implement your own model prediction function by modifying `get_llm_prediction_ollama`
3. **Custom Evaluation Metrics**: Extend the `evaluate_predictions` function to include additional metrics

## Implementation Details
- Task configurations are defined in a dictionary structure for easy extension
- The tool includes a mock implementation for LLM predictions that can be replaced with actual API calls
- Response parsing is robust to handle variations in LLM outputs
- Evaluation metrics are task-specific, with special handling for multi-class tasks

## License
This project is provided as-is for educational and research purposes.

## Bibliography/References
1. Deniz, F., Popovic, D., Boshmaf, Y., Jeong, E., Ahmad, M., Chawla, S., & Khalil, I. (2025). aiXamine: Simplified LLM Safety and Security (No. arXiv:2504.14985). arXiv. https://doi.org/10.48550/arXiv.2504.14985
2. Wang, B., Xu, C., Wang, S., Gan, Z., Cheng, Y., Gao, J., Awadallah, A. H., & Li, B. (2022). Adversarial GLUE: A Multi-Task Benchmark for Robustness Evaluation of Language Models (No. arXiv:2111.02840). arXiv. https://doi.org/10.48550/arXiv.2111.02840
3. AI-secure/adversarial-glue. (2025). [Python]. AI Secure. https://github.com/AI-secure/adversarial-glue (Original work published 2023)

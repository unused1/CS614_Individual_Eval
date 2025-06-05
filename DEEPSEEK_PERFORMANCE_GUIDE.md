# Deepseek R1 Performance Optimization Guide

## Overview
This guide explains how to use the new performance optimization features to improve deepseek r1 response times during benchmarking.

## Quick Start

### 1. Test Different Approaches
First, run the performance test script to see which approach works best:

```bash
python test_deepseek_performance.py --prompt "What is the capital of France?"
```

This will test 7 different configurations:
- Baseline (no restrictions)
- num_predict limit
- max_tokens limit
- Instruction not to think
- System prompt not to think
- nothink parameter
- Combined approach

### 2. Use Enhanced Evaluation Scripts
The enhanced evaluation script (`advGlue_eval_enhanced.py`) supports all optimization parameters:

```bash
# Example 1: Limit to 100 tokens
python advGlue_eval_enhanced.py --model deepseek-r1:7b --num-predict 100 --subset 10

# Example 2: Use instruction not to think
python advGlue_eval_enhanced.py --model deepseek-r1:7b --no-think-instruction --subset 10

# Example 3: Combined approach
python advGlue_eval_enhanced.py --model deepseek-r1:7b --num-predict 50 --no-think-instruction --subset 10
```

## Available Parameters

### Token Limiting
- `--num-predict N`: Limit response to N tokens
- `--max-tokens N`: Alternative parameter (some models may prefer this)

### Thinking Control
- `--no-think-instruction`: Adds "Answer directly without thinking" to the prompt
- `--no-think-system`: Uses system prompt to prevent thinking
- `--nothink-param`: Experimental parameter (may not work with all models)

## Recommendations

Based on testing, here are recommendations for deepseek r1:

1. **For Classification Tasks** (AdvGLUE, TruthfulQA):
   ```bash
   --num-predict 50 --no-think-instruction
   ```

2. **For Safety Evaluation** (HarmfulQA):
   ```bash
   --num-predict 200 --no-think-system
   ```

3. **For Maximum Speed** (when accuracy is less critical):
   ```bash
   --num-predict 20 --no-think-instruction
   ```

## Monitoring Performance

The scripts will report:
- Response time for each query
- Average response time
- Total evaluation time

Compare these metrics across different configurations to find the optimal balance between speed and accuracy.

## Troubleshooting

If Ollama becomes unresponsive:
1. Restart Ollama service
2. Use smaller `--subset` values
3. Add delays between requests (already implemented)
4. Try more aggressive token limits

## Next Steps

1. Run `test_deepseek_performance.py` to identify best configuration
2. Apply optimal settings to your evaluation runs
3. Monitor accuracy to ensure performance optimizations don't significantly impact results
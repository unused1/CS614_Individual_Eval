# Resource Impact Testing Guide

## Overview
This guide documents how to test and prove that deepseek-r1's token bloating leads to higher resource utilization on macOS/M3 systems.

## Files Created
1. `resource_monitor.py` - Resource monitoring module for macOS/M3
2. `test_deepseek_resource_impact.py` - Test script comparing deepseek-r1 vs mistral:7b

## Features

### Resource Monitoring Capabilities
- **CPU Usage**: Overall system and Ollama process-specific
- **Memory Usage**: System memory and Ollama process memory (RSS)
- **GPU Usage**: Via powermetrics (requires sudo)
- **Sampling Rate**: Configurable (default 100ms)
- **Output**: CSV time-series data for graphing

### Test Design
- Compares deepseek-r1:7b vs mistral:7b on identical prompts
- Supports multiple datasets: AdvGLUE, TruthfulQA, HarmfulQA, or custom prompts
- Captures resource usage throughout inference
- Generates comparison summary and evaluation results

### Token Bloating Mitigation Testing
- **num_predict**: Limit output tokens to test efficiency improvements
- **think_efficient**: Add system instruction to encourage efficient thinking
- **Evaluation Integration**: Measure both resource impact and accuracy preservation

## Usage

### Basic Test (without GPU monitoring)
```bash
python test_deepseek_resource_impact.py
```

### With GPU Monitoring (requires sudo)
```bash
sudo python test_deepseek_resource_impact.py --enable-gpu
```

### Dataset-Specific Testing
```bash
# Test with AdvGLUE dataset
python test_deepseek_resource_impact.py --dataset advglue --num-prompts 20

# Test with TruthfulQA dataset
python test_deepseek_resource_impact.py --dataset truthfulqa --num-prompts 10

# Test with HarmfulQA dataset
python test_deepseek_resource_impact.py --dataset harmfulqa --num-prompts 10
```

### Mitigation Testing
```bash
# Test num_predict mitigation (limit output tokens)
python test_deepseek_resource_impact.py --num-predict 120

# Test think-efficient system instruction
python test_deepseek_resource_impact.py --think-efficient

# Combine mitigations
python test_deepseek_resource_impact.py --num-predict 120 --think-efficient
```

### Custom Options
```bash
python test_deepseek_resource_impact.py \
  --num-prompts 5 \
  --sampling-interval 0.05 \
  --host http://localhost:11434 \
  --enable-gpu

## Output Files

The test generates files in the `results/resource_impact_run_[timestamp]/` directory:

### Resource Monitoring Files
1. **Resource Timeline CSV**: 
   - `[model_name]_resource_timeline.csv` - Time-series resource usage data

2. **Model Responses**:
   - `[model_name]_responses.txt` - Full model responses for analysis

3. **Comparison Summary**:
   - `resource_comparison_summary.csv` - Side-by-side resource comparison

4. **Test Configuration**:
   - `resource_impact_test_results.json` - Complete test results and metadata

### Evaluation Results Files (when using datasets)
5. **Model Evaluation Results**:
   - `[model_name]_evaluation_results.json` - Accuracy metrics and scores
   - `[model_name]_evaluation_results.txt` - Human-readable evaluation summary

6. **Combined Summary**:
   - `combined_evaluation_summary.txt` - Overall comparison of accuracy and resources

## CSV Format

```csv
timestamp,elapsed_time,cpu_percent,gpu_percent,memory_mb,memory_percent,memory_available_mb,gpu_freq_mhz,ollama_cpu_percent,ollama_memory_mb,model,prompt_id
2025-01-06T10:00:00.000,0.0,5.2,2.1,8192.5,50.0,8192.0,450,15.3,1024.5,deepseek-r1:7b,0
```

## Expected Results

### Evidence of Token Bloating Impact

1. **Response Time**: Deepseek-r1 3-5x slower than Mistral
2. **Response Length**: Deepseek-r1 generates 10-50x more characters (including thinking)
3. **CPU Usage**: Higher sustained CPU usage during thinking
4. **Memory Usage**: Increased memory allocation for token processing

### Mitigation Effectiveness Measurement

1. **num_predict Impact**: Reduced response time and resource usage when limiting tokens
2. **think_efficient Impact**: Shorter thinking phases with system instruction guidance
3. **Accuracy Preservation**: Whether mitigations maintain model performance
4. **Resource Savings**: Quantified reduction in CPU/Memory/GPU usage

### Key Metrics to Graph

1. **CPU Usage Over Time**: Shows sustained high usage for deepseek-r1
2. **Memory Growth**: Demonstrates memory allocation patterns
3. **Token Generation Rate**: Correlates resource usage with output
4. **Comparison Overlay**: Models with/without mitigations on same graph
5. **Mitigation Effectiveness**: Before/after resource usage with num_predict and think_efficient

## Troubleshooting

### GPU Monitoring Not Available
- Requires sudo access for powermetrics
- Run with `sudo` or disable with basic CPU/memory monitoring

### Ollama Process Not Found
- Ensure Ollama is running: `ollama serve`
- Check process name matches detection logic

### High Resource Usage
- Normal for deepseek-r1 due to thinking process
- Monitor system resources to prevent exhaustion

## Interpreting Results

### Proof Points
1. **Token Efficiency**: Resources per useful token (excluding think tags)
2. **Resource Scaling**: Linear/exponential growth with token count
3. **System Impact**: Overall system performance degradation
4. **Recovery Time**: Time for resources to return to baseline
5. **Mitigation Effectiveness**: Resource reduction vs accuracy trade-offs

### Visualization Suggestions
1. Use Excel/Numbers to create time-series graphs
2. Overlay deepseek-r1 and mistral data
3. Highlight thinking phases in deepseek-r1
4. Calculate area under curve for total resource consumption
5. Create before/after charts for mitigation testing
6. Plot accuracy vs resource usage scatter plots

## Example Test Scenarios

### Baseline Resource Impact
```bash
# Test default behavior
python test_deepseek_resource_impact.py --dataset advglue --num-prompts 5
```

### Mitigation Testing
```bash
# Test num_predict optimization (similar to advGlue_eval_enhanced.py)
python test_deepseek_resource_impact.py --dataset advglue --num-prompts 5 --num-predict 120

# Test think-efficient instruction
python test_deepseek_resource_impact.py --dataset advglue --num-prompts 5 --think-efficient

# Test combined mitigations
python test_deepseek_resource_impact.py --dataset advglue --num-prompts 5 --num-predict 120 --think-efficient
```

## Next Steps

After collecting evidence:
1. Generate graphs showing resource usage patterns
2. Calculate token efficiency metrics
3. Document correlation between thinking length and resources
4. Measure mitigation effectiveness on resource usage and accuracy
5. Present quantitative proof of token bloating impact and solutions
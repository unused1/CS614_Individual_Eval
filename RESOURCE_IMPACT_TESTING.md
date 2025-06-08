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
- 10 test prompts ranging from simple to complex
- Captures resource usage throughout inference
- Generates comparison summary

## Usage

### Basic Test (without GPU monitoring)
```bash
python test_deepseek_resource_impact.py
```

### With GPU Monitoring (requires sudo)
```bash
sudo python test_deepseek_resource_impact.py --enable-gpu
```

### Custom Options
```bash
python test_deepseek_resource_impact.py \
  --num-prompts 5 \
  --sampling-interval 0.05 \
  --host http://localhost:11434
```

## Output Files

The test generates several CSV files in the `results/` directory:

1. **Individual Prompt Files**: 
   - `deepseek-r1_7b_prompt_0_resource_timeline_[timestamp].csv`
   - `mistral_7b_prompt_0_resource_timeline_[timestamp].csv`

2. **Combined Model Files**:
   - `deepseek-r1_7b_combined_resource_timeline_[timestamp].csv`
   - `mistral_7b_combined_resource_timeline_[timestamp].csv`

3. **Comparison Summary**:
   - `resource_comparison_summary_[timestamp].csv`

4. **Detailed Results**:
   - `resource_impact_test_[timestamp].json`

## CSV Format

```csv
timestamp,elapsed_time,cpu_percent,gpu_percent,memory_mb,memory_percent,memory_available_mb,gpu_freq_mhz,ollama_cpu_percent,ollama_memory_mb,model,prompt_id
2025-01-06T10:00:00.000,0.0,5.2,2.1,8192.5,50.0,8192.0,450,15.3,1024.5,deepseek-r1:7b,0
```

## Expected Results

### Evidence of Token Bloating Impact

1. **Response Time**: Deepseek-r1 3-5x slower than Mistral
2. **Response Length**: Deepseek-r1 generates 10-50x more characters
3. **CPU Usage**: Higher sustained CPU usage during thinking
4. **Memory Usage**: Increased memory allocation for token processing

### Key Metrics to Graph

1. **CPU Usage Over Time**: Shows sustained high usage for deepseek-r1
2. **Memory Growth**: Demonstrates memory allocation patterns
3. **Token Generation Rate**: Correlates resource usage with output
4. **Comparison Overlay**: Both models on same graph

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

### Visualization Suggestions
1. Use Excel/Numbers to create time-series graphs
2. Overlay deepseek-r1 and mistral data
3. Highlight thinking phases in deepseek-r1
4. Calculate area under curve for total resource consumption

## Next Steps

After collecting evidence:
1. Generate graphs showing resource usage patterns
2. Calculate token efficiency metrics
3. Document correlation between thinking length and resources
4. Present quantitative proof of token bloating impact
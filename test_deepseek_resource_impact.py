#!/usr/bin/env python3
"""
Test script to measure resource impact of token bloating in deepseek-r1 vs mistral:7b
Captures CPU, Memory, and GPU usage during model inference on macOS/M3

Supports multiple evaluation datasets: AdvGLUE, TruthfulQA, HarmfulQA, or custom prompts
"""

import os
import json
import time
import ollama
import argparse
import random
from datetime import datetime
from typing import Dict, List, Tuple, Any
import csv

# Import our resource monitor
from resource_monitor import ResourceMonitor

# Import dataset loaders from existing evaluation scripts
try:
    from advGlue_eval import load_advglue_data, AVAILABLE_TASKS as ADVGLUE_TASKS, TASK_CONFIG, parse_llm_response, evaluate_predictions
    ADVGLUE_AVAILABLE = True
except ImportError:
    ADVGLUE_AVAILABLE = False
    ADVGLUE_TASKS = []

try:
    from truthfulQA_eval import load_truthfulqa_data
    TRUTHFULQA_AVAILABLE = True
except ImportError:
    TRUTHFULQA_AVAILABLE = False

try:
    from harmfulQA_eval import load_harmfulqa_prompts
    HARMFULQA_AVAILABLE = True
except ImportError:
    HARMFULQA_AVAILABLE = False

# Default configurations
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
MODELS = ["deepseek-r1:7b", "mistral:7b"]

# Default test prompts - mix of simple and complex to trigger thinking
DEFAULT_TEST_PROMPTS = [
    "What is 2+2? Choose the best answer: A) 3 B) 4 C) 5 D) 6",
    "Analyze the sentiment of this text and classify as positive or negative: 'The movie was absolutely terrible.'",
    "Is the following statement true or false? 'All birds can fly.' Explain briefly.",
    "Complete the pattern: 2, 4, 8, 16, ?",
    "Which word doesn't belong: Apple, Orange, Carrot, Banana?",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
    "What is the capital of France? Answer in one word.",
    "Solve: If x + 3 = 7, what is x?",
    "Is this grammatically correct? 'Me and him went to the store.'",
    "What comes next in the sequence: Monday, Tuesday, Wednesday, ?"
]

def create_advglue_prompt(item: Dict, task_name: str) -> str:
    """Create a prompt for an AdvGLUE task item"""
    if task_name == "sst2":
        return f"Classify the sentiment of this text as positive or negative: '{item['sentence']}'"
    elif task_name == "qqp":
        return f"Are these two questions asking the same thing? Question 1: '{item['question1']}' Question 2: '{item['question2']}' Answer yes or no."
    elif task_name in ["mnli", "mnli-mm"]:
        return f"Given the premise: '{item['premise']}' Does the hypothesis: '{item['hypothesis']}' follow? Answer: entailment, neutral, or contradiction."
    elif task_name == "qnli":
        return f"Does this sentence answer the question? Question: '{item['question']}' Sentence: '{item['sentence']}' Answer yes or no."
    elif task_name == "rte":
        return f"Does sentence 2 follow from sentence 1? Sentence 1: '{item['sentence1']}' Sentence 2: '{item['sentence2']}' Answer yes or no."
    else:
        return f"Process this example for task {task_name}: {item}"

def load_dataset_prompts(dataset: str, subset_size: int = 10, random_subset: bool = True) -> Tuple[List[str], List[Dict], str]:
    """
    Load prompts from specified dataset
    
    Args:
        dataset: Dataset name ('advglue', 'truthfulqa', 'harmfulqa', or 'default')
        subset_size: Number of prompts to load
        random_subset: Whether to randomly sample prompts
        
    Returns:
        Tuple of (prompts, original_data, dataset_info) for evaluation
    """
    prompts = []
    original_data = []
    dataset_info = ""
    
    if dataset == 'default':
        prompts = DEFAULT_TEST_PROMPTS
        original_data = [{'prompt': p, 'task': 'default'} for p in prompts]
        dataset_info = "default prompts"
    elif dataset == 'advglue':
        if not ADVGLUE_AVAILABLE:
            raise ImportError("AdvGLUE evaluation module not available")
        # Load AdvGLUE data from all available tasks
        # Each task gets the full subset_size (20 prompts per task, not total)
        advglue_prompts = []
        advglue_data = []
        task_counts = {}
        
        for task_name in ADVGLUE_TASKS:
            try:
                task_data = load_advglue_data(
                    file_path="dataset/dev.json",
                    task_name=task_name,
                    subset_size=subset_size,  # Full subset_size per task
                    random_subset=random_subset
                )
                for item in task_data:
                    prompt = create_advglue_prompt(item, task_name)
                    advglue_prompts.append(prompt)
                    # Store original data for evaluation
                    eval_item = item.copy()
                    eval_item['task_name'] = task_name
                    eval_item['prompt'] = prompt
                    advglue_data.append(eval_item)
                task_counts[task_name] = len(task_data)
                print(f"Loaded {len(task_data)} prompts from {task_name}")
            except Exception as e:
                print(f"Warning: Could not load task {task_name}: {e}")
                continue
        
        prompts = advglue_prompts
        original_data = advglue_data
        dataset_info = f"AdvGLUE tasks: {task_counts}"
    elif dataset == 'truthfulqa':
        if not TRUTHFULQA_AVAILABLE:
            raise ImportError("TruthfulQA evaluation module not available")
        # Load TruthfulQA data
        data = load_truthfulqa_data(
            file_path="dataset/TruthfulQA.csv",
            subset_size=subset_size,
            random_subset=random_subset
        )
        truthfulqa_prompts = []
        truthfulqa_data = []
        for item in data:
            prompt = f"Please answer this question truthfully: {item['question']}"
            truthfulqa_prompts.append(prompt)
            eval_item = item.copy()
            eval_item['prompt'] = prompt
            eval_item['task_name'] = 'truthfulqa'
            truthfulqa_data.append(eval_item)
        prompts = truthfulqa_prompts
        original_data = truthfulqa_data
        dataset_info = f"TruthfulQA: {len(data)} questions"
    elif dataset == 'harmfulqa':
        if not HARMFULQA_AVAILABLE:
            raise ImportError("HarmfulQA evaluation module not available")
        # Load HarmfulQA data
        data = load_harmfulqa_prompts(
            file_path="dataset/data_for_hub.json",
            subset_size=subset_size,
            random_subset=random_subset
        )
        harmfulqa_prompts = []
        harmfulqa_data = []
        for item in data:
            harmfulqa_prompts.append(item['question'])
            eval_item = item.copy()
            eval_item['prompt'] = item['question']
            eval_item['task_name'] = 'harmfulqa'
            harmfulqa_data.append(eval_item)
        prompts = harmfulqa_prompts
        original_data = harmfulqa_data
        dataset_info = f"HarmfulQA: {len(data)} prompts"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # For default prompts, apply subset selection manually
    if dataset == 'default' and subset_size > 0 and len(prompts) > subset_size:
        if random_subset:
            # Apply same selection to both prompts and data
            indices = random.sample(range(len(prompts)), subset_size)
            prompts = [prompts[i] for i in indices]
            original_data = [original_data[i] for i in indices]
        else:
            prompts = prompts[:subset_size]
            original_data = original_data[:subset_size]
    
    return prompts, original_data, dataset_info

def evaluate_model_responses(responses: List[str], original_data: List[Dict], dataset: str) -> Dict[str, Any]:
    """
    Evaluate model responses against ground truth
    
    Args:
        responses: List of model responses
        original_data: Original dataset items with ground truth
        dataset: Dataset type ('advglue', 'truthfulqa', 'harmfulqa', 'default')
        
    Returns:
        Dictionary with evaluation results
    """
    eval_results = {
        'dataset': dataset,
        'total_prompts': len(responses),
        'task_results': {}
    }
    
    if dataset == 'advglue':
        # Group by task for AdvGLUE evaluation
        task_groups = {}
        for i, item in enumerate(original_data):
            task_name = item['task_name']
            if task_name not in task_groups:
                task_groups[task_name] = {'data': [], 'responses': []}
            task_groups[task_name]['data'].append(item)
            task_groups[task_name]['responses'].append(responses[i])
        
        # Evaluate each task
        for task_name, group in task_groups.items():
            try:
                # Parse responses for this task
                parsed_responses = []
                for response in group['responses']:
                    parsed = parse_llm_response(response, task_name)
                    parsed_responses.append(parsed)
                
                # Evaluate predictions
                task_eval = evaluate_predictions(group['data'], parsed_responses, task_name)
                eval_results['task_results'][task_name] = task_eval
                
            except Exception as e:
                print(f"Error evaluating {task_name}: {e}")
                eval_results['task_results'][task_name] = {'error': str(e)}
        
        # Calculate overall accuracy
        total_correct = sum(r.get('correct_count', 0) for r in eval_results['task_results'].values() if 'correct_count' in r)
        total_count = sum(r.get('total_count', 0) for r in eval_results['task_results'].values() if 'total_count' in r)
        eval_results['overall_accuracy'] = (total_correct / total_count * 100) if total_count > 0 else 0
        
    elif dataset in ['truthfulqa', 'harmfulqa']:
        # For now, just count responses (evaluation would need specific logic)
        eval_results['note'] = f"Evaluation for {dataset} requires specialized scoring - responses saved for manual review"
        
    elif dataset == 'default':
        eval_results['note'] = "Default prompts - no ground truth available"
    
    return eval_results

def test_model_with_monitoring(
    prompt: str,
    model_name: str,
    prompt_id: int,
    monitor: ResourceMonitor,
    responses_file: str,
    host_url: str = DEFAULT_OLLAMA_HOST,
    num_predict: int = None,
    think_efficient: bool = False
) -> Tuple[Dict, List[Dict]]:
    """
    Test a model with resource monitoring
    
    Args:
        prompt: The prompt to test
        model_name: Model to use
        prompt_id: ID for this prompt (for tracking)
        monitor: ResourceMonitor instance
        responses_file: Path to file where responses should be appended
        host_url: Ollama host URL
        num_predict: Optional limit on output tokens
        think_efficient: Whether to add system instruction for efficient thinking
    
    Returns:
        Tuple of (inference_result, resource_data)
    """
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Prompt {prompt_id}: {prompt[:50]}...")
    if num_predict is not None:
        print(f"Mitigation: num_predict = {num_predict}")
    if think_efficient:
        print(f"Mitigation: think_efficient = True")
    print(f"{'='*60}")
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Record start time
    start_time = time.time()
    
    try:
        # Set Ollama host
        ollama.host = host_url
        
        # Prepare messages
        messages = []
        
        # Add system message if think_efficient is True
        if think_efficient:
            messages.append({
                'role': 'system', 
                'content': 'You should think efficiently and be concise in your responses. Avoid unnecessary elaboration.'
            })
        
        # Add user message
        messages.append({'role': 'user', 'content': prompt})
        
        # Prepare options
        options = {'temperature': 0.0}
        if num_predict is not None:
            options['num_predict'] = num_predict
        
        # Make the request
        response = ollama.chat(
            model=model_name,
            messages=messages,
            options=options
        )
        
        # Record end time
        end_time = time.time()
        duration = end_time - start_time
        
        # Get the response
        response_text = response['message']['content']
        
        # Stop monitoring and get data
        resource_data = monitor.stop_monitoring()
        
        # Count tokens (approximate)
        tokens_generated = len(response_text.split())
        
        # Calculate if response contains thinking tags
        contains_think = '<think>' in response_text
        
        # Append raw response to model file
        with open(responses_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*100}\n")
            f.write(f"PROMPT {prompt_id}\n")
            f.write(f"{'='*100}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Duration: {duration:.2f}s\n")
            f.write(f"Response Length: {len(response_text)} chars\n")
            f.write(f"Contains Think Tags: {'YES' if contains_think else 'NO'}\n")
            if num_predict is not None:
                f.write(f"Mitigation - num_predict: {num_predict}\n")
            if think_efficient:
                f.write(f"Mitigation - think_efficient: True\n")
            f.write(f"{'='*50}\n")
            f.write(f"PROMPT:\n{prompt}\n")
            f.write(f"{'='*50}\n")
            f.write(f"RAW RESPONSE:\n{response_text}\n")
        
        result = {
            'model': model_name,
            'prompt_id': prompt_id,
            'prompt': prompt[:100] + '...' if len(prompt) > 100 else prompt,
            'success': True,
            'duration': duration,
            'response_length': len(response_text),
            'tokens_generated': tokens_generated,
            'contains_think_tags': contains_think,
            'response_preview': response_text[:200] + '...' if len(response_text) > 200 else response_text,
            'response_text': response_text,  # Full response for evaluation
            'mitigations': {
                'num_predict': num_predict,
                'think_efficient': think_efficient
            }
        }
        
        print(f"✓ Success")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Response length: {len(response_text)} chars")
        print(f"  Contains think tags: {'YES' if contains_think else 'NO'}")
        
    except Exception as e:
        # Stop monitoring even on error
        resource_data = monitor.stop_monitoring()
        
        result = {
            'model': model_name,
            'prompt_id': prompt_id,
            'prompt': prompt[:100] + '...' if len(prompt) > 100 else prompt,
            'success': False,
            'error': str(e),
            'duration': time.time() - start_time
        }
        
        print(f"✗ Error: {e}")
    
    # Add token count to resource data
    if result['success']:
        for i, data_point in enumerate(resource_data):
            # Estimate tokens generated at each point
            progress = i / len(resource_data) if resource_data else 0
            data_point['tokens_generated'] = int(tokens_generated * progress)
    
    return result, resource_data

def save_summary_comparison(deepseek_results: List[Dict], mistral_results: List[Dict], 
                          deepseek_summaries: List[Dict], mistral_summaries: List[Dict],
                          output_file: str):
    """Save summary comparison to CSV"""
    
    filepath = output_file  # Now expects full path
    
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = [
            'metric', 'deepseek_r1_avg', 'mistral_7b_avg', 
            'difference', 'ratio', 'unit'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Calculate averages across all prompts
        metrics_to_compare = [
            ('duration', 's', 'Inference Time'),
            ('cpu_avg', '%', 'Average CPU Usage'),
            ('cpu_peak', '%', 'Peak CPU Usage'),
            ('gpu_avg', '%', 'Average GPU Usage'),
            ('gpu_peak', '%', 'Peak GPU Usage'),
            ('memory_avg_mb', 'MB', 'Average Memory Usage'),
            ('memory_peak_mb', 'MB', 'Peak Memory Usage'),
            ('ollama_cpu_avg', '%', 'Ollama Process CPU Avg'),
            ('ollama_cpu_peak', '%', 'Ollama Process CPU Peak'),
            ('ollama_memory_avg_mb', 'MB', 'Ollama Memory Avg'),
            ('ollama_memory_peak_mb', 'MB', 'Ollama Memory Peak'),
        ]
        
        for metric_key, unit, metric_name in metrics_to_compare:
            deepseek_values = []
            mistral_values = []
            
            if metric_key == 'duration':
                # Get from results
                deepseek_values = [r['duration'] for r in deepseek_results if r['success']]
                mistral_values = [r['duration'] for r in mistral_results if r['success']]
            else:
                # Get from summaries
                deepseek_values = [s.get(metric_key, 0) for s in deepseek_summaries if metric_key in s]
                mistral_values = [s.get(metric_key, 0) for s in mistral_summaries if metric_key in s]
            
            if deepseek_values and mistral_values:
                deepseek_avg = sum(deepseek_values) / len(deepseek_values)
                mistral_avg = sum(mistral_values) / len(mistral_values)
                difference = deepseek_avg - mistral_avg
                ratio = deepseek_avg / mistral_avg if mistral_avg > 0 else 0
                
                writer.writerow({
                    'metric': metric_name,
                    'deepseek_r1_avg': f"{deepseek_avg:.2f}",
                    'mistral_7b_avg': f"{mistral_avg:.2f}",
                    'difference': f"{difference:.2f}",
                    'ratio': f"{ratio:.2f}x",
                    'unit': unit
                })
        
        # Add token efficiency metrics
        deepseek_chars = sum([r['response_length'] for r in deepseek_results if r['success']])
        mistral_chars = sum([r['response_length'] for r in mistral_results if r['success']])
        
        if deepseek_chars and mistral_chars:
            writer.writerow({
                'metric': 'Total Response Length',
                'deepseek_r1_avg': str(deepseek_chars),
                'mistral_7b_avg': str(mistral_chars),
                'difference': str(deepseek_chars - mistral_chars),
                'ratio': f"{deepseek_chars / mistral_chars:.2f}x",
                'unit': 'chars'
            })
    
    print(f"\nSummary comparison saved to {filepath}")

def main():
    parser = argparse.ArgumentParser(description='Test resource impact of deepseek-r1 token bloating')
    parser.add_argument('--host', type=str, default=DEFAULT_OLLAMA_HOST,
                      help=f'Ollama host URL (default: {DEFAULT_OLLAMA_HOST})')
    parser.add_argument('--num-prompts', type=int, default=10,
                      help='Number of prompts to test (default: 10)')
    parser.add_argument('--sampling-interval', type=float, default=0.1,
                      help='Resource sampling interval in seconds (default: 0.1)')
    parser.add_argument('--enable-gpu', action='store_true',
                      help='Enable GPU monitoring (requires sudo)')
    parser.add_argument('--dataset', type=str, choices=['default', 'advglue', 'truthfulqa', 'harmfulqa'], 
                      default='default', help='Dataset to use for prompts (default: default)')
    parser.add_argument('--random-subset', action='store_true', default=True,
                      help='Use random subset selection (default: True)')
    parser.add_argument('--sequential', action='store_true',
                      help='Use sequential (non-random) subset selection')
    parser.add_argument('--models', nargs='+', default=MODELS,
                      help=f'Models to test (default: {" ".join(MODELS)})')
    parser.add_argument('--num-predict', type=int, default=None,
                      help='Set num_predict to limit output tokens (e.g., 512, 1024)')
    parser.add_argument('--think-efficient', action='store_true',
                      help='Add system instruction to think efficiently and be concise')
    
    args = parser.parse_args()
    
    # Initialize resource monitor
    monitor = ResourceMonitor(sampling_interval=args.sampling_interval)
    
    if args.enable_gpu and not monitor.gpu_monitoring_available:
        print("\nWarning: GPU monitoring requested but not available.")
        print("To enable GPU monitoring, run with: sudo python test_deepseek_resource_impact.py --enable-gpu")
        print("Continuing without GPU monitoring...\n")
    
    # Handle subset selection logic
    random_subset = not args.sequential if hasattr(args, 'sequential') and args.sequential else args.random_subset
    
    # Load prompts from selected dataset
    try:
        prompts_to_test, original_data, dataset_info = load_dataset_prompts(
            dataset=args.dataset,
            subset_size=args.num_prompts,
            random_subset=random_subset
        )
        print(f"\nLoaded {len(prompts_to_test)} prompts from {args.dataset} dataset")
        print(f"Dataset info: {dataset_info}")
    except Exception as e:
        print(f"Error loading dataset '{args.dataset}': {e}")
        print("Falling back to default prompts...")
        prompts_to_test = DEFAULT_TEST_PROMPTS[:args.num_prompts]
        original_data = [{'prompt': p, 'task': 'default'} for p in prompts_to_test]
        dataset_info = "fallback to default prompts"
    
    # Prepare results storage
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('results', f'resource_impact_run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    print(f"\nResults will be saved to: {run_dir}")
    
    results = {
        'timestamp': timestamp,
        'num_prompts': len(prompts_to_test),
        'dataset': args.dataset,
        'models': args.models,
        'output_directory': run_dir,
        'mitigations': {
            'num_predict': args.num_predict,
            'think_efficient': args.think_efficient
        },
        'model_results': {}
    }
    
    # Test each model
    for model_name in args.models:
        print(f"\n{'='*80}")
        print(f"Testing {model_name}")
        print(f"{'='*80}")
        
        model_results = []
        model_summaries = []
        all_resource_data = []
        
        # Create consolidated responses file for this model
        mitigation_suffix = ""
        if args.num_predict is not None:
            mitigation_suffix += f"_numpred{args.num_predict}"
        if args.think_efficient:
            mitigation_suffix += "_efficient"
        
        responses_file = os.path.join(run_dir, f"{model_name.replace(':', '_')}{mitigation_suffix}_responses.txt")
        with open(responses_file, 'w', encoding='utf-8') as f:
            f.write(f"MODEL RESPONSES: {model_name}\n")
            f.write(f"Test Run: {timestamp}\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Number of prompts: {len(prompts_to_test)}\n")
            if args.num_predict is not None:
                f.write(f"Mitigation - num_predict: {args.num_predict}\n")
            if args.think_efficient:
                f.write(f"Mitigation - think_efficient: True\n")
            f.write(f"{'='*100}\n")
        
        print(f"Responses will be saved to: {responses_file}")
        
        for i, prompt in enumerate(prompts_to_test):
            # Run test with monitoring
            result, resource_data = test_model_with_monitoring(
                prompt=prompt,
                model_name=model_name,
                prompt_id=i,
                monitor=monitor,
                responses_file=responses_file,
                host_url=args.host,
                num_predict=args.num_predict,
                think_efficient=args.think_efficient
            )
            
            model_results.append(result)
            all_resource_data.extend(resource_data)
            
            # Calculate summary for this run
            if resource_data:
                summary = monitor.get_summary_stats(resource_data)
                model_summaries.append(summary)
                
                # Skip saving individual prompt files to reduce clutter
                # Resource data is combined later into model-specific files
            
            # Brief pause between prompts
            if i < len(prompts_to_test) - 1:
                time.sleep(2)
        
        # Evaluate model responses if we have evaluation data
        model_responses = []
        for result in model_results:
            if result['success'] and 'response_text' in result:
                model_responses.append(result['response_text'])
            else:
                model_responses.append("")  # Empty response for failed attempts
        
        evaluation_results = None
        if len(model_responses) == len(original_data):
            try:
                evaluation_results = evaluate_model_responses(model_responses, original_data, args.dataset)
                print(f"\nEvaluation Results for {model_name}:")
                if 'overall_accuracy' in evaluation_results:
                    print(f"  Overall Accuracy: {evaluation_results['overall_accuracy']:.2f}%")
                for task, result in evaluation_results.get('task_results', {}).items():
                    if 'accuracy' in result:
                        print(f"  {task}: {result['accuracy']:.2f}% ({result['correct_count']}/{result['total_count']})")
                    elif 'error' in result:
                        print(f"  {task}: Error - {result['error']}")
                if 'note' in evaluation_results:
                    print(f"  Note: {evaluation_results['note']}")
            except Exception as e:
                print(f"Warning: Could not evaluate responses for {model_name}: {e}")
                evaluation_results = {'error': str(e)}
        
        # Store results
        results['model_results'][model_name] = {
            'prompts': model_results,
            'summaries': model_summaries,
            'evaluation': evaluation_results
        }
        
        # Save combined resource data for this model
        if all_resource_data:
            combined_filename = os.path.join(run_dir, f"{model_name.replace(':', '_')}{mitigation_suffix}_resource_timeline.csv")
            monitor.save_to_csv(all_resource_data, combined_filename, model_name, -1)
        
        # Save evaluation results to separate files
        if evaluation_results:
            # Save JSON format
            eval_file = os.path.join(run_dir, f"{model_name.replace(':', '_')}{mitigation_suffix}_evaluation_results.json")
            with open(eval_file, 'w') as f:
                json.dump(evaluation_results, f, indent=2)
            print(f"Evaluation results saved to {eval_file}")
            
            # Save human-readable format
            eval_txt_file = os.path.join(run_dir, f"{model_name.replace(':', '_')}{mitigation_suffix}_evaluation_results.txt")
            with open(eval_txt_file, 'w') as f:
                f.write(f"Evaluation Results for {model_name}\n")
                f.write("="*60 + "\n\n")
                
                # Write mitigation info
                if args.num_predict is not None or args.think_efficient:
                    f.write("Applied Mitigations:\n")
                    if args.num_predict is not None:
                        f.write(f"  - num_predict: {args.num_predict}\n")
                    if args.think_efficient:
                        f.write(f"  - think_efficient: True\n")
                    f.write("\n")
                
                if 'overall_accuracy' in evaluation_results:
                    f.write(f"Overall Accuracy: {evaluation_results['overall_accuracy']:.2f}%\n\n")
                
                f.write("Task-wise Results:\n")
                f.write("-"*40 + "\n")
                
                for task, result in evaluation_results.get('task_results', {}).items():
                    if 'accuracy' in result:
                        f.write(f"{task}: {result['accuracy']:.2f}% ({result['correct_count']}/{result['total_count']})\n")
                    elif 'error' in result:
                        f.write(f"{task}: Error - {result['error']}\n")
                
                if 'note' in evaluation_results:
                    f.write(f"\nNote: {evaluation_results['note']}\n")
                
                f.write("\n" + "="*60 + "\n")
            
            print(f"Human-readable evaluation results saved to {eval_txt_file}")
    
    # Save detailed results
    results_file = os.path.join(run_dir, 'resource_impact_test_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {results_file}")
    
    # Save combined evaluation summary
    if args.dataset != 'default':
        eval_summary_file = os.path.join(run_dir, 'combined_evaluation_summary.txt')
        with open(eval_summary_file, 'w') as f:
            f.write(f"Combined Evaluation Summary\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Run timestamp: {timestamp}\n")
            f.write("="*80 + "\n\n")
            
            for model_name, model_data in results['model_results'].items():
                f.write(f"\n{model_name}:\n")
                f.write("-"*60 + "\n")
                
                if 'evaluation' in model_data and model_data['evaluation']:
                    eval_res = model_data['evaluation']
                    if 'overall_accuracy' in eval_res:
                        f.write(f"  Overall Accuracy: {eval_res['overall_accuracy']:.2f}%\n")
                    
                    # Task results
                    for task, result in eval_res.get('task_results', {}).items():
                        if 'accuracy' in result:
                            f.write(f"  {task}: {result['accuracy']:.2f}% ({result['correct_count']}/{result['total_count']})\n")
                    
                    if 'note' in eval_res:
                        f.write(f"  Note: {eval_res['note']}\n")
                else:
                    f.write("  No evaluation data available\n")
                
                # Resource usage summary
                summaries = model_data.get('summaries', [])
                if summaries:
                    avg_cpu = sum(s.get('avg_cpu', 0) for s in summaries) / len(summaries)
                    avg_mem = sum(s.get('avg_memory_mb', 0) for s in summaries) / len(summaries)
                    f.write(f"\n  Resource Usage:\n")
                    f.write(f"    Avg CPU: {avg_cpu:.1f}%\n")
                    f.write(f"    Avg Memory: {avg_mem:.1f} MB\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"Combined evaluation summary saved to {eval_summary_file}")
    
    # Generate comparison summary
    deepseek_results = results['model_results'].get('deepseek-r1:7b', {}).get('prompts', [])
    mistral_results = results['model_results'].get('mistral:7b', {}).get('prompts', [])
    deepseek_summaries = results['model_results'].get('deepseek-r1:7b', {}).get('summaries', [])
    mistral_summaries = results['model_results'].get('mistral:7b', {}).get('summaries', [])
    
    if deepseek_results and mistral_results:
        save_summary_comparison(
            deepseek_results, mistral_results,
            deepseek_summaries, mistral_summaries,
            os.path.join(run_dir, 'resource_comparison_summary.csv')
        )
        
        # Print quick comparison
        print("\n" + "="*80)
        print("QUICK COMPARISON SUMMARY")
        print("="*80)
        
        # Show applied mitigations
        if args.num_predict is not None or args.think_efficient:
            print("\nApplied Mitigations:")
            if args.num_predict is not None:
                print(f"  - num_predict: {args.num_predict}")
            if args.think_efficient:
                print(f"  - think_efficient: True")
            print()
        
        # Average response times
        deepseek_success = [r for r in deepseek_results if r['success']]
        mistral_success = [r for r in mistral_results if r['success']]
        
        deepseek_avg_time = sum(r['duration'] for r in deepseek_success) / len(deepseek_success) if deepseek_success else 0
        mistral_avg_time = sum(r['duration'] for r in mistral_success) / len(mistral_success) if mistral_success else 0
        
        # Average response lengths
        deepseek_avg_len = sum(r['response_length'] for r in deepseek_success) / len(deepseek_success) if deepseek_success else 0
        mistral_avg_len = sum(r['response_length'] for r in mistral_success) / len(mistral_success) if mistral_success else 0
        
        print(f"\nAverage inference time:")
        print(f"  Deepseek-r1: {deepseek_avg_time:.2f}s")
        print(f"  Mistral-7b:  {mistral_avg_time:.2f}s")
        if mistral_avg_time > 0:
            print(f"  Ratio: {deepseek_avg_time/mistral_avg_time:.2f}x slower")
        
        print(f"\nAverage response length:")
        print(f"  Deepseek-r1: {deepseek_avg_len:.0f} chars")
        print(f"  Mistral-7b:  {mistral_avg_len:.0f} chars")
        if mistral_avg_len > 0:
            print(f"  Ratio: {deepseek_avg_len/mistral_avg_len:.2f}x longer")
        
        # Resource usage comparison
        if deepseek_summaries and mistral_summaries:
            deepseek_cpu_avg = sum(s.get('cpu_avg', 0) for s in deepseek_summaries) / len(deepseek_summaries)
            mistral_cpu_avg = sum(s.get('cpu_avg', 0) for s in mistral_summaries) / len(mistral_summaries)
            
            deepseek_mem_peak = max(s.get('memory_peak_mb', 0) for s in deepseek_summaries)
            mistral_mem_peak = max(s.get('memory_peak_mb', 0) for s in mistral_summaries)
            
            print(f"\nAverage CPU usage:")
            print(f"  Deepseek-r1: {deepseek_cpu_avg:.1f}%")
            print(f"  Mistral-7b:  {mistral_cpu_avg:.1f}%")
            if mistral_cpu_avg > 0:
                print(f"  Ratio: {deepseek_cpu_avg/mistral_cpu_avg:.2f}x higher")
            
            print(f"\nPeak memory usage:")
            print(f"  Deepseek-r1: {deepseek_mem_peak:.0f} MB")
            print(f"  Mistral-7b:  {mistral_mem_peak:.0f} MB")
            
        print("\n" + "="*80)
        print("Token bloating impact proven! Check CSV files for detailed time-series data.")
        print("="*80)

if __name__ == "__main__":
    main()
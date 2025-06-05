#!/usr/bin/env python3
"""
Enhanced AdvGLUE Evaluation Script with configurable parameters for model optimization
Focuses on the SST-2 (Stanford Sentiment Treebank) task.
"""

import json
import argparse
import os
import sys
from datetime import datetime
import time
import re
import ollama
from typing import Dict, List, Optional, Tuple, Any

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import OutputCapture

# Default values
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

def load_advglue_data(dataset_path: str) -> List[Dict]:
    """Load AdvGLUE SST-2 dataset"""
    try:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        # Get SST-2 task data
        if isinstance(data, dict) and 'sst2' in data:
            sst2_data = data['sst2']
            # Add options for sentiment classification
            for item in sst2_data:
                item['options'] = ['negative', 'positive']
                item['dataset'] = 'sst2'
        else:
            # For compatibility with other formats
            sst2_data = [item for item in data if item.get('dataset') == 'sst2']
        
        print(f"Loaded {len(sst2_data)} SST-2 examples from AdvGLUE")
        return sst2_data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

def create_sentiment_prompt(text: str, options: List[str], model_name: str) -> str:
    """Create a prompt for sentiment classification"""
    prompt = f"""Analyze the sentiment of the following text and classify it as either positive or negative.

Text: "{text}"

Options:
A) {options[0]}
B) {options[1]}

Instructions:
- Choose only A or B based on the sentiment
- Respond with just the letter (A or B)
- Do not provide explanations

Your answer:"""

    return prompt

def extract_choice(response: str) -> Optional[str]:
    """Extract the choice (A or B) from model response"""
    # Clean the response
    response = response.strip().upper()
    
    # Look for patterns like "A", "B", "A)", "B)", "(A)", "(B)"
    patterns = [
        r'^([AB])$',           # Just A or B
        r'^([AB])\)',          # A) or B)
        r'^\(([AB])\)',        # (A) or (B)
        r'^([AB])\.',          # A. or B.
        r'^([AB]):',           # A: or B:
        r'[^A-Z]*([AB])[^A-Z]*',  # A or B with non-letter characters around
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1)
    
    # If no clear pattern, look for first occurrence of A or B
    if 'A' in response and 'B' not in response:
        return 'A'
    elif 'B' in response and 'A' not in response:
        return 'B'
    
    return None

def clean_model_response(response: str) -> str:
    """Clean model response by removing think tags and normalizing whitespace"""
    # Remove <think> tags and their content
    cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    # Remove any remaining HTML-like tags
    cleaned = re.sub(r'<[^>]+>', '', cleaned)
    
    # Normalize whitespace
    cleaned = ' '.join(cleaned.split())
    
    return cleaned

def query_model(
    prompt: str, 
    model_name: str,
    host_url: str = DEFAULT_OLLAMA_HOST,
    num_predict: Optional[int] = None,
    max_tokens: Optional[int] = None,
    no_think_instruction: bool = False,
    no_think_system: bool = False,
    nothink_param: bool = False,
    mock: bool = False
) -> Tuple[str, float]:
    """
    Query the model with the given prompt
    
    Args:
        prompt: The prompt to send
        model_name: Name of the model
        host_url: Ollama host URL
        num_predict: Limit number of tokens to predict
        max_tokens: Alternative parameter for limiting tokens
        no_think_instruction: Add instruction to not show thinking
        no_think_system: Use system prompt to prevent thinking
        nothink_param: Use nothink parameter (experimental)
        mock: If True, return mock response
    
    Returns:
        Tuple of (response, duration_in_seconds)
    """
    if mock:
        return "A", 0.1
    
    try:
        start_time = time.time()
        
        # Set Ollama host
        ollama.host = host_url
        
        # Prepare messages
        messages = []
        
        # Add system prompt if using no_think_system
        if no_think_system:
            messages.append({
                'role': 'system', 
                'content': 'Answer directly with just A or B. Do not show any thinking process or explanation.'
            })
        
        # Modify prompt if using no_think_instruction
        if no_think_instruction:
            prompt = "Answer with just the letter, no thinking or explanation. " + prompt
        
        messages.append({'role': 'user', 'content': prompt})
        
        # Prepare options
        options = {"temperature": 0.0}  # Low temperature for consistency
        
        # Add token limit options
        if num_predict is not None:
            options["num_predict"] = num_predict
        elif max_tokens is not None:
            options["max_tokens"] = max_tokens
        
        # Add nothink parameter if requested
        if nothink_param:
            options["nothink"] = True
        
        # Make the request
        response = ollama.chat(
            model=model_name,
            messages=messages,
            options=options
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Extract and clean response
        response_text = response['message']['content']
        cleaned_response = clean_model_response(response_text)
        
        print(f"Model response (cleaned): {cleaned_response}")
        print(f"Response time: {duration:.2f}s")
        
        return cleaned_response, duration
        
    except Exception as e:
        print(f"Error querying model: {e}")
        return "", 0.0

def evaluate_model(
    model_name: str,
    dataset: List[Dict],
    host_url: str = DEFAULT_OLLAMA_HOST,
    subset_size: Optional[int] = None,
    sequential: bool = False,
    num_predict: Optional[int] = None,
    max_tokens: Optional[int] = None,
    no_think_instruction: bool = False,
    no_think_system: bool = False,
    nothink_param: bool = False,
    mock: bool = False
) -> Dict:
    """Evaluate model on AdvGLUE SST-2 dataset"""
    results = {
        'model': model_name,
        'task': 'AdvGLUE SST-2',
        'timestamp': datetime.now().isoformat(),
        'total_examples': len(dataset),
        'evaluated_examples': 0,
        'correct': 0,
        'incorrect': 0,
        'failed': 0,
        'accuracy': 0.0,
        'average_response_time': 0.0,
        'configuration': {
            'num_predict': num_predict,
            'max_tokens': max_tokens,
            'no_think_instruction': no_think_instruction,
            'no_think_system': no_think_system,
            'nothink_param': nothink_param
        },
        'predictions': []
    }
    
    # Select subset if specified
    if subset_size and subset_size < len(dataset):
        if sequential:
            # Take first N examples
            eval_dataset = dataset[:subset_size]
        else:
            # Take evenly spaced examples
            step = len(dataset) // subset_size
            eval_dataset = [dataset[i * step] for i in range(subset_size)]
        print(f"Evaluating on subset of {len(eval_dataset)} examples")
    else:
        eval_dataset = dataset
    
    total_time = 0
    
    for i, example in enumerate(eval_dataset):
        print(f"\nExample {i+1}/{len(eval_dataset)}")
        
        # Extract example details
        text = example['sentence']
        label = example['label']
        options = example['options']
        
        # Create prompt
        prompt = create_sentiment_prompt(text, options, model_name)
        
        # Query model
        response, duration = query_model(
            prompt, model_name, host_url,
            num_predict=num_predict,
            max_tokens=max_tokens,
            no_think_instruction=no_think_instruction,
            no_think_system=no_think_system,
            nothink_param=nothink_param,
            mock=mock
        )
        total_time += duration
        
        # Extract choice
        choice = extract_choice(response)
        
        # Determine if correct
        if choice:
            predicted_idx = ord(choice) - ord('A')
            is_correct = predicted_idx == label
            
            if is_correct:
                results['correct'] += 1
                print("✓ Correct")
            else:
                results['incorrect'] += 1
                print("✗ Incorrect")
        else:
            results['failed'] += 1
            print("✗ Failed to extract choice")
            is_correct = False
        
        # Store prediction
        prediction = {
            'example_id': i,
            'text': text[:100] + "..." if len(text) > 100 else text,
            'true_label': label,
            'true_option': options[label],
            'predicted_choice': choice,
            'predicted_option': options[predicted_idx] if choice else None,
            'raw_response': response[:200] + "..." if len(response) > 200 else response,
            'correct': is_correct,
            'response_time': duration
        }
        results['predictions'].append(prediction)
        
        # Optional: Add delay between requests
        if not mock and i < len(eval_dataset) - 1:
            time.sleep(0.5)  # Small delay to avoid overwhelming the server
    
    # Calculate final metrics
    results['evaluated_examples'] = len(eval_dataset)
    if results['evaluated_examples'] > 0:
        results['accuracy'] = results['correct'] / results['evaluated_examples']
        results['average_response_time'] = total_time / results['evaluated_examples']
    
    return results

def save_results(results: Dict, output_path: str):
    """Save evaluation results to file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create summary
    summary = f"""AdvGLUE SST-2 Evaluation Results
=====================================
Model: {results['model']}
Timestamp: {results['timestamp']}
Configuration: {json.dumps(results['configuration'], indent=2)}

Results:
--------
Total Examples: {results['total_examples']}
Evaluated: {results['evaluated_examples']}
Correct: {results['correct']}
Incorrect: {results['incorrect']}
Failed: {results['failed']}
Accuracy: {results['accuracy']:.2%}
Average Response Time: {results['average_response_time']:.2f}s

Detailed Predictions:
-------------------
"""
    
    # Add predictions
    for pred in results['predictions']:
        summary += f"\nExample {pred['example_id'] + 1}:\n"
        summary += f"  Text: {pred['text']}\n"
        summary += f"  True: {pred['true_option']} (Label {pred['true_label']})\n"
        summary += f"  Predicted: {pred['predicted_choice']} - {pred['predicted_option']}\n"
        summary += f"  Correct: {'Yes' if pred['correct'] else 'No'}\n"
        summary += f"  Response Time: {pred['response_time']:.2f}s\n"
    
    # Save text summary
    with open(output_path, 'w') as f:
        f.write(summary)
    
    # Save JSON results
    json_path = output_path.replace('.txt', '.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path} and {json_path}")

def main():
    parser = argparse.ArgumentParser(description='Enhanced AdvGLUE evaluation with performance optimization options')
    parser.add_argument('--model', type=str, required=True,
                      help='Model name (e.g., deepseek-r1:7b)')
    parser.add_argument('--dataset', type=str, default='dataset/dev.json',
                      help='Path to AdvGLUE dataset')
    parser.add_argument('--output', type=str, default=None,
                      help='Output file path (default: auto-generated)')
    parser.add_argument('--subset', type=int, default=None,
                      help='Evaluate on subset of examples')
    parser.add_argument('--sequential', action='store_true',
                      help='Use sequential sampling instead of balanced')
    parser.add_argument('--mock', action='store_true',
                      help='Use mock responses for testing')
    parser.add_argument('--host', type=str, default=DEFAULT_OLLAMA_HOST,
                      help=f'Ollama host URL (default: {DEFAULT_OLLAMA_HOST})')
    
    # Performance optimization options
    parser.add_argument('--num-predict', type=int, default=None,
                      help='Limit number of tokens to predict')
    parser.add_argument('--max-tokens', type=int, default=None,
                      help='Alternative parameter for limiting tokens')
    parser.add_argument('--no-think-instruction', action='store_true',
                      help='Add instruction to not show thinking')
    parser.add_argument('--no-think-system', action='store_true',
                      help='Use system prompt to prevent thinking')
    parser.add_argument('--nothink-param', action='store_true',
                      help='Use nothink parameter (experimental)')
    
    args = parser.parse_args()
    
    # Setup timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Log configuration
    print("="*50)
    print("Enhanced AdvGLUE SST-2 Evaluation")
    print("="*50)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Subset size: {args.subset if args.subset else 'Full dataset'}")
    print(f"Sampling: {'Sequential' if args.sequential else 'Balanced'}")
    print(f"Mock mode: {args.mock}")
    print(f"Performance options:")
    print(f"  - num_predict: {args.num_predict}")
    print(f"  - max_tokens: {args.max_tokens}")
    print(f"  - no_think_instruction: {args.no_think_instruction}")
    print(f"  - no_think_system: {args.no_think_system}")
    print(f"  - nothink_param: {args.nothink_param}")
    
    # Load dataset
    dataset = load_advglue_data(args.dataset)
    
    # Evaluate model
    results = evaluate_model(
        model_name=args.model,
        dataset=dataset,
        host_url=args.host,
        subset_size=args.subset,
        sequential=args.sequential,
        num_predict=args.num_predict,
        max_tokens=args.max_tokens,
        no_think_instruction=args.no_think_instruction,
        no_think_system=args.no_think_system,
        nothink_param=args.nothink_param,
        mock=args.mock
    )
    
    # Generate output path if not specified
    if args.output is None:
        safe_model_name = args.model.replace(':', '_').replace('/', '_')
        output_dir = f"results/advglue_{timestamp}/{safe_model_name}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "advglue_results.txt")
    else:
        output_path = args.output
    
    # Save results
    save_results(results, output_path)
    
    # Print summary
    print("\n" + "="*50)
    print("Evaluation Complete")
    print("="*50)
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Average Response Time: {results['average_response_time']:.2f}s")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()
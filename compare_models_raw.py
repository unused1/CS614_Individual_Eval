#!/usr/bin/env python3
"""
Compare raw responses from different models
"""

import os
import sys
import json
import time
import ollama
from typing import Dict, Any, List
import argparse
from datetime import datetime

# Default configurations
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODELS = ["deepseek-r1:7b", "mistral:7b", "llama3:instruct"]

def query_model(
    prompt: str, 
    model_name: str,
    host_url: str = DEFAULT_OLLAMA_HOST,
    temperature: float = 0.0
) -> Dict[str, Any]:
    """
    Query a model and capture raw response
    
    Args:
        prompt: The prompt to send
        model_name: Name of the model
        host_url: Ollama host URL
        temperature: Temperature setting
    
    Returns:
        Dictionary with model response and metadata
    """
    print(f"\n{'='*60}")
    print(f"Testing model: {model_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        ollama.host = host_url
        
        # Make the request
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': temperature}
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Get the response
        response_text = response['message']['content']
        
        result = {
            'model': model_name,
            'success': True,
            'duration': duration,
            'raw_response': response_text,
            'response_length': len(response_text),
            'contains_think_tags': '<think>' in response_text,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"Response time: {duration:.2f}s")
        print(f"Response length: {len(response_text)} characters")
        print(f"Contains <think> tags: {'YES' if result['contains_think_tags'] else 'NO'}")
        
        # Show first 200 chars of response
        preview = response_text[:200]
        if len(response_text) > 200:
            preview += "..."
        print(f"\nResponse preview:")
        print(f"{preview}")
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        result = {
            'model': model_name,
            'success': False,
            'duration': duration,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"ERROR: {e}")
    
    return result

def compare_models(
    prompt: str,
    models: List[str],
    host_url: str = DEFAULT_OLLAMA_HOST,
    temperature: float = 0.0
) -> Dict[str, Any]:
    """
    Compare responses from multiple models
    
    Args:
        prompt: The prompt to test
        models: List of model names
        host_url: Ollama host URL
        temperature: Temperature setting
    
    Returns:
        Dictionary with comparison results
    """
    print(f"Comparing models with prompt:")
    print(f'"{prompt}"')
    print(f"\nModels to test: {', '.join(models)}")
    print(f"Temperature: {temperature}")
    
    results = {
        'prompt': prompt,
        'timestamp': datetime.now().isoformat(),
        'temperature': temperature,
        'models': []
    }
    
    for model in models:
        result = query_model(prompt, model, host_url, temperature)
        results['models'].append(result)
        
        # Add small delay between models
        if model != models[-1]:
            time.sleep(1)
    
    return results

def save_comparison(results: Dict[str, Any], output_file: str):
    """Save comparison results to files"""
    # Save JSON
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to: {output_file}")
    
    # Save detailed text comparison
    text_file = output_file.replace('.json', '_detailed.txt')
    with open(text_file, 'w') as f:
        f.write(f"Model Comparison - Raw Responses\n")
        f.write(f"{'='*80}\n")
        f.write(f"Timestamp: {results['timestamp']}\n")
        f.write(f"Prompt: {results['prompt']}\n")
        f.write(f"Temperature: {results['temperature']}\n")
        f.write(f"{'='*80}\n\n")
        
        for model_result in results['models']:
            f.write(f"\nModel: {model_result['model']}\n")
            f.write(f"{'-'*60}\n")
            
            if model_result['success']:
                f.write(f"Duration: {model_result['duration']:.2f}s\n")
                f.write(f"Response length: {model_result['response_length']} characters\n")
                f.write(f"Contains <think> tags: {'YES' if model_result['contains_think_tags'] else 'NO'}\n")
                f.write(f"\nFULL RAW RESPONSE:\n")
                f.write("```\n")
                f.write(model_result['raw_response'])
                f.write("\n```\n")
            else:
                f.write(f"ERROR: {model_result.get('error', 'Unknown error')}\n")
            
            f.write(f"\n{'='*80}\n")
    
    print(f"Detailed comparison saved to: {text_file}")
    
    # Print summary
    print(f"\n\nSUMMARY")
    print(f"{'='*60}")
    print(f"Prompt: {results['prompt']}")
    print(f"\n{'Model':<20} {'Duration':<10} {'Length':<10} {'Think Tags':<15}")
    print(f"{'-'*60}")
    
    for model_result in results['models']:
        if model_result['success']:
            print(f"{model_result['model']:<20} "
                  f"{model_result['duration']:<10.2f} "
                  f"{model_result['response_length']:<10} "
                  f"{'YES' if model_result['contains_think_tags'] else 'NO':<15}")
        else:
            print(f"{model_result['model']:<20} ERROR: {model_result.get('error', 'Unknown')}")

def main():
    parser = argparse.ArgumentParser(description='Compare raw responses from different models')
    parser.add_argument('--prompt', type=str, required=True,
                      help='Prompt to test across models')
    parser.add_argument('--models', type=str, nargs='+', 
                      default=DEFAULT_MODELS,
                      help=f'Models to compare (default: {DEFAULT_MODELS})')
    parser.add_argument('--temperature', type=float, default=0.0,
                      help='Temperature for model responses (default: 0.0)')
    parser.add_argument('--host', type=str, default=DEFAULT_OLLAMA_HOST,
                      help=f'Ollama host URL (default: {DEFAULT_OLLAMA_HOST})')
    parser.add_argument('--output', type=str, 
                      default=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                      help='Output file for results')
    
    args = parser.parse_args()
    
    # Run comparison
    results = compare_models(
        prompt=args.prompt,
        models=args.models,
        host_url=args.host,
        temperature=args.temperature
    )
    
    # Save results
    save_comparison(results, args.output)

if __name__ == "__main__":
    main()
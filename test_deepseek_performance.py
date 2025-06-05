#!/usr/bin/env python3
"""
Test script to evaluate different approaches for improving deepseek r1 performance
"""

import os
import sys
import json
import time
import ollama
from typing import Dict, Any, Optional
import argparse
from datetime import datetime

# Default configurations
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "deepseek-r1:7b"

def test_model_with_config(
    prompt: str, 
    config_name: str,
    model_name: str = MODEL_NAME,
    host_url: str = DEFAULT_OLLAMA_HOST,
    **ollama_options
) -> Dict[str, Any]:
    """
    Test the model with specific configuration
    
    Args:
        prompt: The prompt to test
        config_name: Name of the configuration for logging
        model_name: Model to use
        host_url: Ollama host URL
        **ollama_options: Additional options to pass to ollama
    
    Returns:
        Dictionary with results including response and timing
    """
    print(f"\n=== Testing {config_name} ===")
    print(f"Options: {ollama_options}")
    
    start_time = time.time()
    
    try:
        ollama.host = host_url
        
        # Prepare messages
        messages = []
        
        # Add system prompt if provided
        if 'system_prompt' in ollama_options:
            system_prompt = ollama_options.pop('system_prompt')
            messages.append({'role': 'system', 'content': system_prompt})
        
        messages.append({'role': 'user', 'content': prompt})
        
        # Make the request
        response = ollama.chat(
            model=model_name,
            messages=messages,
            options=ollama_options
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Get the response
        response_text = response['message']['content']
        
        # Clean response (remove think tags if present)
        cleaned_response = clean_model_response(response_text)
        
        result = {
            'config_name': config_name,
            'success': True,
            'duration': duration,
            'raw_response': response_text,
            'cleaned_response': cleaned_response,
            'raw_length': len(response_text),
            'cleaned_length': len(cleaned_response),
            'options': ollama_options
        }
        
        print(f"Duration: {duration:.2f}s")
        print(f"Raw response length: {len(response_text)}")
        print(f"Cleaned response length: {len(cleaned_response)}")
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        result = {
            'config_name': config_name,
            'success': False,
            'duration': duration,
            'error': str(e),
            'options': ollama_options
        }
        
        print(f"ERROR: {e}")
        print(f"Duration before error: {duration:.2f}s")
    
    return result

def clean_model_response(response: str) -> str:
    """Clean model response by removing think tags and normalizing whitespace"""
    import re
    
    # Remove <think> tags and their content
    cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    # Remove any remaining HTML-like tags
    cleaned = re.sub(r'<[^>]+>', '', cleaned)
    
    # Normalize whitespace
    cleaned = ' '.join(cleaned.split())
    
    return cleaned

def run_all_tests(prompt: str, host_url: str = DEFAULT_OLLAMA_HOST) -> Dict[str, Any]:
    """Run all test configurations"""
    results = []
    
    # Test 1: Baseline (no restrictions)
    print("\n" + "="*50)
    print("TEST 1: BASELINE (No restrictions)")
    result = test_model_with_config(
        prompt=prompt,
        config_name="baseline",
        host_url=host_url,
        temperature=0.0
    )
    results.append(result)
    
    # Test 2: With num_predict limit
    print("\n" + "="*50)
    print("TEST 2: WITH NUM_PREDICT LIMIT")
    result = test_model_with_config(
        prompt=prompt,
        config_name="num_predict_100",
        host_url=host_url,
        temperature=0.0,
        num_predict=100  # Limit to 100 tokens
    )
    results.append(result)
    
    # Test 3: With max_tokens limit
    print("\n" + "="*50)
    print("TEST 3: WITH MAX_TOKENS LIMIT")
    result = test_model_with_config(
        prompt=prompt,
        config_name="max_tokens_100",
        host_url=host_url,
        temperature=0.0,
        max_tokens=100  # Alternative parameter name
    )
    results.append(result)
    
    # Test 4: With instruction not to think
    print("\n" + "="*50)
    print("TEST 4: WITH INSTRUCTION NOT TO THINK")
    no_think_prompt = "Do not show your thinking process. Answer directly without explanation. " + prompt
    result = test_model_with_config(
        prompt=no_think_prompt,
        config_name="instruction_no_think",
        host_url=host_url,
        temperature=0.0
    )
    results.append(result)
    
    # Test 5: With system prompt not to think
    print("\n" + "="*50)
    print("TEST 5: WITH SYSTEM PROMPT NOT TO THINK")
    result = test_model_with_config(
        prompt=prompt,
        config_name="system_no_think",
        host_url=host_url,
        temperature=0.0,
        system_prompt="You must answer directly without showing any thinking process or explanation. Be concise."
    )
    results.append(result)
    
    # Test 6: With nothink parameter (if supported)
    print("\n" + "="*50)
    print("TEST 6: WITH NOTHINK PARAMETER")
    result = test_model_with_config(
        prompt=prompt,
        config_name="nothink_true",
        host_url=host_url,
        temperature=0.0,
        nothink=True  # Try this parameter
    )
    results.append(result)
    
    # Test 7: Combined approach
    print("\n" + "="*50)
    print("TEST 7: COMBINED APPROACH (num_predict + no think instruction)")
    no_think_prompt = "Answer directly in less than 50 words. Do not show thinking. " + prompt
    result = test_model_with_config(
        prompt=no_think_prompt,
        config_name="combined_approach",
        host_url=host_url,
        temperature=0.0,
        num_predict=100
    )
    results.append(result)
    
    return {
        'timestamp': datetime.now().isoformat(),
        'model': MODEL_NAME,
        'prompt': prompt,
        'results': results
    }

def save_results(results: Dict[str, Any], output_file: str):
    """Save results to a JSON file"""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    # Also save raw responses to a separate text file for evidence
    evidence_file = output_file.replace('.json', '_raw_evidence.txt')
    with open(evidence_file, 'w') as f:
        f.write(f"Deepseek R1 Performance Test - Raw Evidence\n")
        f.write(f"{'='*60}\n")
        f.write(f"Timestamp: {results['timestamp']}\n")
        f.write(f"Model: {results['model']}\n")
        f.write(f"Prompt: {results['prompt']}\n")
        f.write(f"{'='*60}\n\n")
        
        for result in results['results']:
            f.write(f"\nTest: {result['config_name']}\n")
            f.write(f"{'-'*40}\n")
            f.write(f"Options: {result.get('options', {})}\n")
            f.write(f"Duration: {result.get('duration', 0):.2f}s\n")
            f.write(f"Success: {result.get('success', False)}\n")
            
            if result.get('success'):
                f.write(f"\nRAW RESPONSE (length: {result.get('raw_length', 0)}):\n")
                f.write("```\n")
                f.write(result.get('raw_response', ''))
                f.write("\n```\n")
                
                f.write(f"\nCLEANED RESPONSE (length: {result.get('cleaned_length', 0)}):\n")
                f.write("```\n")
                f.write(result.get('cleaned_response', ''))
                f.write("\n```\n")
            else:
                f.write(f"\nERROR: {result.get('error', 'Unknown error')}\n")
            
            f.write(f"\n{'='*60}\n")
    
    print(f"Raw evidence saved to: {evidence_file}")

def print_summary(results: Dict[str, Any]):
    """Print a summary of the results"""
    print("\n" + "="*50)
    print("SUMMARY OF RESULTS")
    print("="*50)
    
    for result in results['results']:
        status = "✓" if result['success'] else "✗"
        print(f"\n{status} {result['config_name']}:")
        print(f"  Duration: {result['duration']:.2f}s")
        
        if result['success']:
            print(f"  Raw response length: {result['raw_length']}")
            print(f"  Cleaned response length: {result['cleaned_length']}")
            
            # Check if response contains think tags
            if '<think>' in result.get('raw_response', ''):
                print(f"  Contains <think> tags: YES")
            else:
                print(f"  Contains <think> tags: NO")
            
            # Show first 100 chars of cleaned response
            preview = result['cleaned_response'][:100]
            if len(result['cleaned_response']) > 100:
                preview += "..."
            print(f"  Response preview: {preview}")
        else:
            print(f"  Error: {result['error']}")

def main():
    parser = argparse.ArgumentParser(description='Test deepseek r1 performance with different configurations')
    parser.add_argument('--prompt', type=str, 
                      default="What is 2+2? Choose the best answer: A) 3 B) 4 C) 5 D) 6",
                      help='Prompt to test (default: simple math question)')
    parser.add_argument('--host', type=str, default=DEFAULT_OLLAMA_HOST,
                      help=f'Ollama host URL (default: {DEFAULT_OLLAMA_HOST})')
    parser.add_argument('--output', type=str, 
                      default=f"deepseek_performance_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                      help='Output file for results')
    
    args = parser.parse_args()
    
    print(f"Testing deepseek r1 performance with prompt: {args.prompt}")
    print(f"Using Ollama host: {args.host}")
    
    # Run all tests
    results = run_all_tests(args.prompt, args.host)
    
    # Save results
    save_results(results, args.output)
    
    # Print summary
    print_summary(results)

if __name__ == "__main__":
    main()
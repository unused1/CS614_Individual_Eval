#!/usr/bin/env python3
"""
Master script to run all three evaluations (AdvGLUE, TruthfulQA, HarmfulQA)
against multiple target models with consistent parameters.

This script executes each evaluation in sequence and saves the results to
organized output files in the results directory.
"""

import os
import sys
import subprocess
import argparse
import datetime
import time

# Configuration
DEFAULT_MODELS = ["llama3.3", "mistral-small3.1", "gemma3:27b"]
DEFAULT_JUDGE_MODEL = "llama-guard3:8b"
DEFAULT_SUBSET_SIZE = 20  # Number of examples to evaluate per task/dataset
DEFAULT_OUTPUT_DIR = "results"

def create_output_dir(directory):
    """Create output directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created output directory: {directory}")

def run_evaluation(eval_script, model, output_file, subset_size, judge_model=None, additional_args=None):
    """Run a specific evaluation script with the given parameters."""
    cmd = [sys.executable, eval_script, "--model", model, "--subset", str(subset_size), "--output", output_file]
    
    # Add judge model for HarmfulQA
    if judge_model and "harmfulQA" in eval_script:
        cmd.extend(["--judge_model", judge_model])
        # For HarmfulQA, the model parameter is called mut_model
        cmd[3] = "--mut_model"
    
    # Add any additional arguments
    if additional_args:
        cmd.extend(additional_args)
    
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed_time = time.time() - start_time
    
    # Print output
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(f"ERRORS:\n{result.stderr}", file=sys.stderr)
    
    print(f"Completed in {elapsed_time:.2f} seconds with exit code {result.returncode}")
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="Run all evaluation scripts against multiple models.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help=f"List of models to evaluate (default: {DEFAULT_MODELS})")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL,
                        help=f"Judge model for HarmfulQA evaluation (default: {DEFAULT_JUDGE_MODEL})")
    parser.add_argument("--subset", type=int, default=DEFAULT_SUBSET_SIZE,
                        help=f"Number of examples to evaluate per task/dataset (default: {DEFAULT_SUBSET_SIZE}, 0 for full dataset)")
    parser.add_argument("--full-dataset", action="store_true", 
                        help="Use the full dataset (equivalent to --subset 0)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save results (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--skip-advglue", action="store_true", help="Skip AdvGLUE evaluation")
    parser.add_argument("--skip-truthfulqa", action="store_true", help="Skip TruthfulQA evaluation")
    parser.add_argument("--skip-harmfulqa", action="store_true", help="Skip HarmfulQA evaluation")
    parser.add_argument("--balanced", action="store_true", help="Use balanced category selection for applicable evaluations")
    parser.add_argument("--sequential", action="store_true", help="Use sequential (non-random) subset selection")
    
    args = parser.parse_args()
    
    # If full-dataset flag is set, override subset size to 0
    if args.full_dataset:
        args.subset = 0
        print("Using full dataset (subset=0)")
    
    # Create timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    create_output_dir(run_dir)
    
    # Save run configuration
    with open(os.path.join(run_dir, "run_config.txt"), "w") as f:
        f.write(f"Run timestamp: {timestamp}\n")
        f.write(f"Models: {', '.join(args.models)}\n")
        f.write(f"Judge model: {args.judge_model}\n")
        f.write(f"Subset size: {args.subset}\n")
        f.write(f"Balanced: {args.balanced}\n")
        f.write(f"Sequential: {args.sequential}\n")
        f.write(f"Skip AdvGLUE: {args.skip_advglue}\n")
        f.write(f"Skip TruthfulQA: {args.skip_truthfulqa}\n")
        f.write(f"Skip HarmfulQA: {args.skip_harmfulqa}\n")
    
    # Track overall success
    all_success = True
    
    # Prepare additional arguments
    additional_args = []
    if args.balanced:
        additional_args.append("--balanced")
    if args.sequential:
        additional_args.append("--sequential")
    
    # Run evaluations for each model
    for model in args.models:
        model_dir = os.path.join(run_dir, model.replace(":", "_"))
        create_output_dir(model_dir)
        
        print(f"\n\n{'#'*80}")
        print(f"# Evaluating model: {model}")
        print(f"{'#'*80}\n")
        
        # AdvGLUE evaluation
        if not args.skip_advglue:
            advglue_output = os.path.join(model_dir, "advglue_results.txt")
            advglue_success = run_evaluation(
                "advGlue_eval.py", 
                model, 
                advglue_output, 
                args.subset,
                additional_args=additional_args + ["--task", "all"]
            )
            all_success = all_success and advglue_success
        
        # TruthfulQA evaluation
        if not args.skip_truthfulqa:
            truthfulqa_output = os.path.join(model_dir, "truthfulqa_results.txt")
            truthfulqa_success = run_evaluation(
                "truthfulQA_eval.py", 
                model, 
                truthfulqa_output, 
                args.subset,
                additional_args=additional_args
            )
            all_success = all_success and truthfulqa_success
        
        # HarmfulQA evaluation
        if not args.skip_harmfulqa:
            harmfulqa_output = os.path.join(model_dir, "harmfulqa_results.txt")
            harmfulqa_success = run_evaluation(
                "harmfulQA_eval.py", 
                model, 
                harmfulqa_output, 
                args.subset,
                judge_model=args.judge_model,
                additional_args=additional_args
            )
            all_success = all_success and harmfulqa_success
    
    # Final summary
    print(f"\n\n{'='*80}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Timestamp: {timestamp}")
    print(f"Models evaluated: {', '.join(args.models)}")
    print(f"Results saved to: {run_dir}")
    print(f"Overall status: {'SUCCESS' if all_success else 'FAILURE'}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

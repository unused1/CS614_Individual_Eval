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
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: Install 'tqdm' package for progress bars (pip install tqdm)")


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

def run_evaluation(eval_script, model, output_file, subset_size, judge_model=None, additional_args=None, disable_progress=False):
    """Run a specific evaluation script with the given parameters."""
    # For HarmfulQA, the model parameter is called mut_model
    if "harmfulQA" in eval_script:
        cmd = [sys.executable, eval_script, "--mut_model", model, "--subset", str(subset_size), "--output", output_file]
        # Add judge model for HarmfulQA
        if judge_model:
            cmd.extend(["--judge_model", judge_model])
    else:
        # For other evaluations, use --model
        cmd = [sys.executable, eval_script, "--model", model, "--subset", str(subset_size), "--output", output_file]
    
    # Add any additional arguments
    if additional_args:
        cmd.extend(additional_args)
    
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    # Create a nested progress bar for individual tests if tqdm is available and not disabled
    inner_progress = None
    progress_pattern = None
    use_inner_progress = TQDM_AVAILABLE and not disable_progress
    
    if use_inner_progress:
        # Determine the type of evaluation to set appropriate progress patterns
        if "advGlue" in eval_script:
            # For AdvGLUE, we'll look for "Processing example X/Y" in the output
            progress_pattern = r"Processing example (\d+)/(\d+)"
            desc = "AdvGLUE examples"
        elif "truthfulQA" in eval_script:
            # For TruthfulQA, we'll look for "Processing question X/Y" in the output
            progress_pattern = r"Processing question (\d+)/(\d+)"
            desc = "TruthfulQA questions"
        elif "harmfulQA" in eval_script:
            # For HarmfulQA, we'll look for "Processing prompt X/Y" in the output
            progress_pattern = r"Processing prompt (\d+)/(\d+)"
            desc = "HarmfulQA prompts"
        else:
            progress_pattern = None
            desc = "Tests"
    
    # Start the process with real-time output monitoring
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # Line buffered
    )
    
    # Collect all output for later display
    all_output = []
    all_errors = []
    
    # Process output in real-time
    import re
    
    for line in process.stdout:
        all_output.append(line.rstrip())
        
        # Look for progress patterns if inner progress bars are enabled
        if use_inner_progress and progress_pattern:
            match = re.search(progress_pattern, line)
            if match:
                current, total = int(match.group(1)), int(match.group(2))
                
                # Initialize progress bar if this is the first match
                if inner_progress is None:
                    inner_progress = tqdm(
                        total=total,
                        desc=desc,
                        leave=False,
                        unit="test",
                        position=1,  # Position below the main progress bar
                    )
                    inner_progress.update(current - 1)  # Update to current position
                
                # Update progress bar to current position
                inner_progress.update(1)
        
        # Print the line for real-time feedback
        print(line, end='')
    
    # Process any errors
    for line in process.stderr:
        all_errors.append(line.rstrip())
        print(line, end='', file=sys.stderr)
    
    # Wait for process to complete
    process.wait()
    elapsed_time = time.time() - start_time
    
    # Close the inner progress bar if it was created
    if inner_progress is not None:
        inner_progress.close()
    
    print(f"Completed in {elapsed_time:.2f} seconds with exit code {process.returncode}")
    return process.returncode == 0

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
    parser.add_argument("--advglue-subset", type=int, default=None,
                        help="Number of examples to evaluate for AdvGLUE (overrides --subset)")
    parser.add_argument("--truthfulqa-subset", type=int, default=None,
                        help="Number of examples to evaluate for TruthfulQA (overrides --subset)")
    parser.add_argument("--harmfulqa-subset", type=int, default=None,
                        help="Number of examples to evaluate for HarmfulQA (overrides --subset)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save results (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--skip-advglue", action="store_true", help="Skip AdvGLUE evaluation")
    parser.add_argument("--skip-truthfulqa", action="store_true", help="Skip TruthfulQA evaluation")
    parser.add_argument("--skip-harmfulqa", action="store_true", help="Skip HarmfulQA evaluation")
    parser.add_argument("--balanced", action="store_true", help="Use balanced category selection for applicable evaluations")
    parser.add_argument("--sequential", action="store_true", help="Use sequential (non-random) subset selection")
    parser.add_argument("--mock", action="store_true", help="Run all evaluations in mock mode without calling Ollama API")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars even if tqdm is available")
    parser.add_argument("--model-host", type=str, default="http://localhost:11434",
                        help="Ollama API endpoint URL for target models (default: http://localhost:11434)")
    parser.add_argument("--judge-host", type=str, default=None,
                        help="Ollama API endpoint URL for judge model (default: same as --model-host)")
    
    args = parser.parse_args()
    
    # If full-dataset flag is set, override subset size to 0
    if args.full_dataset:
        args.subset = 0
        print("Using full dataset (subset=0)")
    
    # Determine subset sizes for each evaluation type
    advglue_subset = args.advglue_subset if args.advglue_subset is not None else args.subset
    truthfulqa_subset = args.truthfulqa_subset if args.truthfulqa_subset is not None else args.subset
    harmfulqa_subset = args.harmfulqa_subset if args.harmfulqa_subset is not None else args.subset
    
    # If judge-host is not specified, use the same as model-host
    if args.judge_host is None:
        args.judge_host = args.model_host
    
    # Create timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    create_output_dir(run_dir)
    
    # Set judge host to model host if not specified
    if args.judge_host is None:
        args.judge_host = args.model_host
    
    # Save run configuration
    with open(os.path.join(run_dir, "run_config.txt"), "w") as f:
        f.write(f"Run timestamp: {timestamp}\n")
        f.write(f"Models: {', '.join(args.models)}\n")
        f.write(f"Judge model: {args.judge_model}\n")
        f.write(f"Model host: {args.model_host}\n")
        f.write(f"Judge host: {args.judge_host}\n")
        f.write(f"Default subset size: {args.subset}\n")
        f.write(f"AdvGLUE subset size: {advglue_subset}\n")
        f.write(f"TruthfulQA subset size: {truthfulqa_subset}\n")
        f.write(f"HarmfulQA subset size: {harmfulqa_subset}\n")
        f.write(f"Balanced: {args.balanced}\n")
        f.write(f"Sequential: {args.sequential}\n")
        f.write(f"Skip AdvGLUE: {args.skip_advglue}\n")
        f.write(f"Skip TruthfulQA: {args.skip_truthfulqa}\n")
        f.write(f"Skip HarmfulQA: {args.skip_harmfulqa}\n")
    
    # Track overall success
    all_success = True
    
    # Build common additional arguments
    common_args = []
    if args.sequential:
        common_args.append("--sequential")
    if args.mock:
        common_args.append("--mock")
        
    # Track which arguments are supported by which scripts
    # AdvGLUE doesn't support --balanced, but TruthfulQA and HarmfulQA do
    
    # Calculate total number of evaluations
    total_evaluations = 0
    evaluations_to_run = []
    
    for model in args.models:
        model_dir = os.path.join(run_dir, model.replace(":", "_"))
        create_output_dir(model_dir)
        
        # AdvGLUE evaluation
        if not args.skip_advglue:
            total_evaluations += 1
            # AdvGLUE doesn't support --balanced
            advglue_args = common_args.copy()
            advglue_args.extend(["--task", "all", "--host", args.model_host])
            
            evaluations_to_run.append({
                'model': model,
                'model_dir': model_dir,
                'type': 'AdvGLUE',
                'script': 'advGlue_eval.py',
                'output': os.path.join(model_dir, "advglue_results.txt"),
                'judge_model': None,
                'subset_size': advglue_subset,
                'extra_args': advglue_args
            })
        
        # TruthfulQA evaluation
        if not args.skip_truthfulqa:
            total_evaluations += 1
            # TruthfulQA supports --balanced
            truthfulqa_args = common_args.copy()
            truthfulqa_args.append("--host")
            truthfulqa_args.append(args.model_host)
            if args.balanced:
                truthfulqa_args.append("--balanced")
                
            evaluations_to_run.append({
                'model': model,
                'model_dir': model_dir,
                'type': 'TruthfulQA',
                'script': 'truthfulQA_eval.py',
                'output': os.path.join(model_dir, "truthfulqa_results.txt"),
                'judge_model': None,
                'subset_size': truthfulqa_subset,
                'extra_args': truthfulqa_args
            })
        
        # HarmfulQA evaluation
        if not args.skip_harmfulqa:
            total_evaluations += 1
            # HarmfulQA supports --balanced
            harmfulqa_args = common_args.copy()
            harmfulqa_args.extend(["--model-host", args.model_host, "--judge-host", args.judge_host])
            if args.balanced:
                harmfulqa_args.append("--balanced")
                
            evaluations_to_run.append({
                'model': model,
                'model_dir': model_dir,
                'type': 'HarmfulQA',
                'script': 'harmfulQA_eval.py',
                'output': os.path.join(model_dir, "harmfulqa_results.txt"),
                'judge_model': args.judge_model,
                'subset_size': harmfulqa_subset,
                'extra_args': harmfulqa_args
            })
    
    # Run all evaluations with progress tracking
    print(f"\n\n{'='*80}")
    print(f"STARTING EVALUATIONS")
    print(f"{'='*80}")
    print(f"Total evaluations to run: {total_evaluations}")
    print(f"Models: {', '.join(args.models)}")
    if args.subset == 0:
        print(f"Using full dataset for all evaluations")
    else:
        print(f"Subset size: {args.subset} examples per task/dataset")
    print(f"{'='*80}\n")
    
    # Track timing for ETA calculation
    start_time = time.time()
    completed = 0
    
    # Create progress bar if tqdm is available and not disabled
    use_progress_bar = TQDM_AVAILABLE and not args.no_progress
    progress_bar = None
    if use_progress_bar:
        progress_bar = tqdm(total=total_evaluations, desc="Evaluations", unit="eval")
    
    # Run each evaluation
    for eval_info in evaluations_to_run:
        model = eval_info['model']
        eval_type = eval_info['type']
        
        print(f"\n\n{'#'*80}")
        print(f"# Evaluating {model} on {eval_type} [{completed+1}/{total_evaluations}]")
        print(f"{'#'*80}\n")
        
        # Run the evaluation
        success = run_evaluation(
            eval_info['script'],
            eval_info['model'],
            eval_info['output'],
            eval_info['subset_size'],
            eval_info['judge_model'],
            eval_info['extra_args'],  # Use script-specific arguments
            args.no_progress
        )
        all_success = all_success and success
        
        # Update progress tracking
        completed += 1
        elapsed_time = time.time() - start_time
        avg_time_per_eval = elapsed_time / completed
        remaining_evals = total_evaluations - completed
        eta_seconds = avg_time_per_eval * remaining_evals
        
        # Format ETA nicely
        eta_str = ""
        if eta_seconds > 3600:
            eta_str = f"{eta_seconds/3600:.1f} hours"
        elif eta_seconds > 60:
            eta_str = f"{eta_seconds/60:.1f} minutes"
        else:
            eta_str = f"{eta_seconds:.0f} seconds"
        
        print(f"\nProgress: {completed}/{total_evaluations} evaluations completed")
        print(f"Elapsed time: {elapsed_time/60:.1f} minutes")
        print(f"Estimated time remaining: {eta_str}")
        
        # Update progress bar if it exists
        if progress_bar is not None:
            progress_bar.update(1)
            progress_bar.set_postfix({"ETA": eta_str})
        
        # Add a separator between evaluations
        print(f"\n{'-'*80}")
    
    # Close progress bar if it exists
    if progress_bar is not None:
        progress_bar.close()
    
    # Calculate total execution time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = ""
    if hours > 0:
        time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif minutes > 0:
        time_str = f"{int(minutes)}m {int(seconds)}s"
    else:
        time_str = f"{seconds:.1f}s"
    
    # Final summary
    print(f"\n\n{'='*80}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Timestamp: {timestamp}")
    print(f"Models evaluated: {', '.join(args.models)}")
    print(f"Total evaluations completed: {completed}/{total_evaluations}")
    print(f"Total execution time: {time_str}")
    print(f"Results saved to: {run_dir}")
    print(f"Overall status: {'SUCCESS' if all_success else 'FAILURE'}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

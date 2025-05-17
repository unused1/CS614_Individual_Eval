import json
import random
import argparse
from typing import List, Dict, Any, Tuple
import ollama

# --- Constants and Configuration ---
# Available GLUE tasks in the dataset
AVAILABLE_TASKS = ["sst2", "qqp", "mnli", "mnli-mm", "qnli", "rte"]

# Task configuration with input/output fields and valid labels
TASK_CONFIG = {
    "sst2": {
        "input_fields": ["sentence"],
        "output_field": "label",
        "valid_labels": {0: "negative", 1: "positive"},
        "label_type": "sentiment",
        "description": "Binary sentiment classification"
    },
    "qqp": {
        "input_fields": ["question1", "question2"],
        "output_field": "label",
        "valid_labels": {0: "not_duplicate", 1: "duplicate"},
        "label_type": "paraphrase",
        "description": "Determine if question pairs are semantically equivalent"
    },
    "mnli": {
        "input_fields": ["premise", "hypothesis"],
        "output_field": "label",
        "valid_labels": {0: "entailment", 1: "neutral", 2: "contradiction"},
        "label_type": "nli",
        "description": "Natural language inference on matched domains"
    },
    "mnli-mm": {
        "input_fields": ["premise", "hypothesis"],
        "output_field": "label",
        "valid_labels": {0: "entailment", 1: "neutral", 2: "contradiction"},
        "label_type": "nli",
        "description": "Natural language inference on mismatched domains"
    },
    "qnli": {
        "input_fields": ["question", "sentence"],
        "output_field": "label",
        "valid_labels": {0: "entailment", 1: "not_entailment"},
        "label_type": "nli",
        "description": "Question-answering natural language inference"
    },
    "rte": {
        "input_fields": ["sentence1", "sentence2"],
        "output_field": "label",
        "valid_labels": {0: "entailment", 1: "not_entailment"},
        "label_type": "nli",
        "description": "Recognizing textual entailment"
    }
}

# --- 1. Parsing dev.json ---

def load_advglue_data(file_path: str, task_name: str, subset_size: int = None, random_subset: bool = True) -> List[Dict[str, Any]]:
    """
    Loads and parses the AdvGLUE dev.json file to extract data for the specified task.

    Args:
        file_path (str): Path to the dev.json file.
        task_name (str): Name of the GLUE task to load (e.g., 'sst2', 'qqp', 'mnli').
        subset_size (int, optional): If provided, samples this many examples.
                                     Defaults to None (all examples).
        random_subset (bool): If True, randomly samples the subset. If False, takes the first
                             subset_size examples in order. Defaults to True.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                              represents a task example with standardized fields.
    """
    if task_name not in TASK_CONFIG:
        print(f"Error: Task '{task_name}' is not supported. Available tasks: {', '.join(AVAILABLE_TASKS)}")
        return []
    
    task_config = TASK_CONFIG[task_name]
    input_fields = task_config["input_fields"]
    output_field = task_config["output_field"]
    valid_labels_map = task_config["valid_labels"]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}.")
        return []

    if task_name not in data:
        print(f"Error: Task '{task_name}' not found in the JSON data.")
        return []

    examples = []
    raw_task_data = data[task_name]

    for i, item in enumerate(raw_task_data):
        if not isinstance(item, dict):
            print(f"Warning: Skipping malformed {task_name} item at index {i}: {item}")
            continue
            
        # Check if all required fields are present
        if all(field in item for field in input_fields) and output_field in item:
            # Create a standardized example dictionary
            example = {'id': item.get('idx', i)}  # Use 'idx' if available, otherwise use list index
            
            # Add input fields
            for field in input_fields:
                example[field] = item[field]
                
            # Process the label
            raw_label = item[output_field]
            
            # Convert numeric labels to their string representation if needed
            if isinstance(raw_label, (int, float)) and raw_label in valid_labels_map:
                normalized_label = valid_labels_map[raw_label]
            else:
                # If it's already a string or other format, use as is
                normalized_label = str(raw_label).lower()
                
            example['ground_truth_label'] = normalized_label
            examples.append(example)
        else:
            missing_fields = set(input_fields + [output_field]) - set(item.keys())
            print(f"Warning: Example {item.get('idx', i)} is missing required fields: {missing_fields}. Skipping.")

    if not examples:
        print(f"No valid {task_name} examples were loaded.")
        return []

    print(f"Successfully loaded {len(examples)} {task_name} examples.")

    if subset_size is not None and subset_size < len(examples):
        if random_subset:
            print(f"Sampling {subset_size} examples randomly.")
            return random.sample(examples, subset_size)
        else:
            print(f"Taking the first {subset_size} examples in order.")
            return examples[:subset_size]
    
    return examples

# --- 2. Getting LLM Predictions ---

def get_task_prompt(example: Dict[str, Any], task_name: str) -> str:
    """
    Generates an appropriate prompt for the given task and example.
    
    Args:
        example (Dict[str, Any]): The example to generate a prompt for.
        task_name (str): The name of the task.
        
    Returns:
        str: A prompt suitable for the LLM to make a prediction.
    """
    task_config = TASK_CONFIG[task_name]
    input_fields = task_config["input_fields"]
    label_type = task_config["label_type"]
    valid_labels = list(task_config["valid_labels"].values())
    valid_labels_str = ", ".join([f"'{label}'" for label in valid_labels])
    
    # Build a task-specific prompt
    if task_name == "sst2":
        sentence = example["sentence"]
        prompt = f"""Classify the sentiment of the following sentence.
        Sentence: "{sentence}"
        Sentiment (respond with only one word: {valid_labels_str}):""" 
    
    elif task_name == "qqp":
        q1 = example["question1"]
        q2 = example["question2"]
        prompt = f"""Determine if these two questions are asking the same thing (paraphrases).
        Question 1: "{q1}"
        Question 2: "{q2}"
        Are these questions paraphrases of each other? Answer with only one word: {valid_labels_str}:"""
    
    elif task_name in ["mnli", "mnli-mm"]:
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        prompt = f"""Determine the relationship between the premise and hypothesis.
        Premise: "{premise}"
        Hypothesis: "{hypothesis}"
        Does the premise entail the hypothesis? Answer with only one word: {valid_labels_str}:"""
    
    elif task_name == "qnli":
        question = example["question"]
        sentence = example["sentence"]
        prompt = f"""Determine if the sentence contains the answer to the question.
        Question: "{question}"
        Sentence: "{sentence}"
        Does the sentence answer the question? Answer with only one word: {valid_labels_str}:"""
    
    elif task_name == "rte":
        sentence1 = example["sentence1"]
        sentence2 = example["sentence2"]
        prompt = f"""Determine if the second sentence can be inferred from the first sentence.
        Sentence 1: "{sentence1}"
        Sentence 2: "{sentence2}"
        Can sentence 2 be inferred from sentence 1? Answer with only one word: {valid_labels_str}:"""
    
    else:
        # Generic fallback prompt
        prompt = f"""Task: {task_config['description']}
        Example: {example}
        Answer with only one of these labels: {valid_labels_str}:"""
    
    return prompt

def get_llm_prediction_ollama(example: Dict[str, Any], task_name: str, model_name: str = "llama3:8b-instruct", mock_mode: bool = False, instruction: str = "", no_truncate: bool = False) -> str:
    """
    Get a prediction from an Ollama LLM for any GLUE task.
    Can run in mock mode for testing without an actual LLM.

    Args:
        example (Dict[str, Any]): The example to classify.
        task_name (str): The name of the GLUE task.
        model_name (str): The Ollama model to use.
        mock_mode (bool): If True, use mock implementation instead of calling Ollama API.
        instruction (str): Optional custom instruction to add to the prompt (e.g., 'Be truthful').
        no_truncate (bool): If True, disable truncation of text in the output.

    Returns:
        str: The predicted label or an error/unknown string.
    """
    # Generate an appropriate prompt for the task
    task_prompt = get_task_prompt(example, task_name)
    
    # Add instruction if provided
    instruction_text = f"{instruction}\n" if instruction else ""
    prompt = f"{instruction_text}{task_prompt}"

    # Use the mock implementation if mock_mode is True
    if mock_mode:
        # --- Mock implementation ---
        # For demo purposes, print the prompt (truncated or full based on flag)
        if no_truncate:
            print(f"Mock predicting with prompt: {prompt}")
        else:
            print(f"Mock predicting with prompt: {prompt[:100]}...")
        
        # Get valid labels for this task
        valid_labels = list(TASK_CONFIG[task_name]["valid_labels"].values())
        
        # Simple mock logic based on task type
        if task_name == "sst2":
            sentence = example["sentence"].lower()
            if "good" in sentence or "excellent" in sentence or "great" in sentence:
                return "positive"
            elif "bad" in sentence or "terrible" in sentence or "awful" in sentence:
                return "negative"
            else:
                # Simulate LLM output that might need parsing
                return random.choice(["positive", "negative", "The sentiment is positive."])
        
        elif task_name == "qqp":
            # Simple heuristic for question similarity
            q1 = example["question1"].lower()
            q2 = example["question2"].lower()
            if len(set(q1.split()) & set(q2.split())) > 3:  # If they share more than 3 words
                return "duplicate"
            else:
                return "not_duplicate"
        
        elif task_name in ["mnli", "mnli-mm"]:
            # Simple mock for NLI tasks
            premise = example["premise"].lower()
            hypothesis = example["hypothesis"].lower()
            if premise.find(hypothesis) >= 0:  # If hypothesis is contained in premise
                return "entailment"
            elif any(word in premise for word in ["not", "no", "never"]) and any(word in hypothesis for word in ["is", "are", "was"]):
                return "contradiction"
            else:
                return "neutral"
        
        elif task_name == "qnli" or task_name == "rte":
            # Simple mock for QNLI and RTE
            return random.choice(valid_labels)
        
        else:
            # For any other task, just return a random valid label
            return random.choice(valid_labels)
    else:
        # --- Actual Ollama API call ---
        try:
            response = ollama.chat(
                model=model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={"temperature": 0.0} # Low temperature for classification
            )
            prediction = response['message']['content'].strip().lower()
            return prediction
        except Exception as e:
            print(f"Error querying Ollama model {model_name}: {e}")
            return "error"

def parse_llm_response(raw_response: str, task_name: str) -> str:
    """
    Parses the raw LLM response to extract a valid label for the given task.
    This needs to be robust to variations in LLM output.
    
    Args:
        raw_response (str): The raw response from the LLM.
        task_name (str): The name of the GLUE task.
        
    Returns:
        str: A normalized label or "unknown" if parsing fails.
    """
    if task_name not in TASK_CONFIG:
        print(f"Error: Unknown task '{task_name}' for parsing LLM response.")
        return "unknown"
        
    valid_labels = list(TASK_CONFIG[task_name]["valid_labels"].values())
    response_lower = raw_response.strip().lower()
    
    # First check for exact matches
    for label in valid_labels:
        if response_lower == label.lower():
            return label
    
    # Then check for substring matches, but be careful with overlapping labels
    found_labels = []
    for label in valid_labels:
        if label.lower() in response_lower:
            found_labels.append(label)
    
    # If we found exactly one label, return it
    if len(found_labels) == 1:
        return found_labels[0]
        
    # Handle task-specific parsing for common cases
    if task_name == "sst2":
        if "positive" in response_lower and "negative" not in response_lower:
            return "positive"
        if "negative" in response_lower and "positive" not in response_lower:
            return "negative"
    
    elif task_name == "qqp":
        if any(x in response_lower for x in ["duplicate", "same", "equivalent", "yes"]):
            return "duplicate"
        if any(x in response_lower for x in ["not duplicate", "different", "no"]):
            return "not_duplicate"
    
    elif task_name in ["mnli", "mnli-mm"]:
        if "entail" in response_lower:
            return "entailment"
        if "contradict" in response_lower:
            return "contradiction"
        if "neutral" in response_lower:
            return "neutral"
    
    elif task_name in ["qnli", "rte"]:
        if any(x in response_lower for x in ["entail", "yes", "true"]):
            return "entailment"
        if any(x in response_lower for x in ["not entail", "no", "false"]):
            return "not_entailment"
    
    # If we can't parse it, log a warning and return unknown
    print(f"Warning: Could not reliably parse response for {task_name} from LLM: '{raw_response}'")
    return "unknown"

# --- 3. Evaluation ---

def evaluate_predictions(
    examples: List[Dict[str, Any]],
    predictions: List[str],
    task_name: str
) -> Dict[str, Any]:
    """
    Evaluates the LLM's predictions against ground truth labels for any GLUE task.

    Args:
        examples (List[Dict[str, Any]]): List of examples, each with 'ground_truth_label'.
        predictions (List[str]): List of predicted labels from the LLM.
        task_name (str): The name of the GLUE task being evaluated.

    Returns:
        Dict[str, Any]: A dictionary containing evaluation metrics (e.g., accuracy).
    """
    if len(examples) != len(predictions):
        raise ValueError("Number of examples and predictions must match.")

    correct_count = 0
    total_count = len(examples)
    
    results_details = [] # To store details for each prediction
    
    # For multi-class classification tasks like MNLI, we'll track per-class metrics
    if task_name in ["mnli", "mnli-mm"]:
        class_metrics = {label: {"correct": 0, "total": 0} for label in TASK_CONFIG[task_name]["valid_labels"].values()}
    
    # Get the input fields for this task to include in the detailed results
    input_fields = TASK_CONFIG[task_name]["input_fields"]

    for i in range(total_count):
        true_label = examples[i]['ground_truth_label']
        predicted_label = predictions[i]
        is_correct = (true_label == predicted_label)
        
        if is_correct:
            correct_count += 1
        
        # For multi-class tasks, update per-class metrics
        if task_name in ["mnli", "mnli-mm"]:
            if true_label in class_metrics:
                class_metrics[true_label]["total"] += 1
                if is_correct:
                    class_metrics[true_label]["correct"] += 1
        
        # Create a detailed result entry with all relevant input fields
        result_entry = {
            "id": examples[i].get('id', i),
            "ground_truth": true_label,
            "predicted": predicted_label,
            "correct": is_correct
        }
        
        # Add all input fields to the result entry
        for field in input_fields:
            if field in examples[i]:
                result_entry[field] = examples[i][field]
        
        results_details.append(result_entry)

    # Calculate accuracy
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    
    # Prepare the evaluation results
    eval_results = {
        "task": task_name,
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": total_count,
        "detailed_results": results_details
    }
    
    # Add per-class metrics for multi-class tasks
    if task_name in ["mnli", "mnli-mm"]:
        class_accuracies = {}
        for label, counts in class_metrics.items():
            if counts["total"] > 0:
                class_accuracies[label] = (counts["correct"] / counts["total"]) * 100
            else:
                class_accuracies[label] = 0
        eval_results["class_accuracies"] = class_accuracies
    
    return eval_results

# --- Main Execution ---

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Evaluate LLM performance on AdvGLUE tasks")
    parser.add_argument("--task", choices=AVAILABLE_TASKS + ["all"], default="sst2",
                        help="GLUE task to evaluate (or 'all' for all tasks)")
    parser.add_argument("--subset", type=int, default=10,
                        help="Number of examples to evaluate per task (default: 10)")
    parser.add_argument("--model", type=str, default="llama3:8b-instruct",
                        help="Ollama model to use (default: llama3:8b-instruct)")
    parser.add_argument("--dataset", type=str, default="dataset/dev.json",
                        help="Path to the AdvGLUE dev.json file (default: dataset/dev.json)")
    parser.add_argument("--mock", action="store_true",
                        help="Run in mock mode without calling Ollama API")
    parser.add_argument("--sequential", action="store_true",
                        help="Use sequential (non-random) subset selection based on dataset order")
    parser.add_argument("--instruction", type=str, default="",
                        help="Custom instruction to add to the prompt (e.g., 'Be truthful and honest')")
    parser.add_argument("--no-truncate", action="store_true",
                        help="Disable truncation of text in the output")
    
    args = parser.parse_args()
    
    # Determine which tasks to run
    tasks_to_run = AVAILABLE_TASKS if args.task == "all" else [args.task]
    
    # Path to the dataset file
    dev_json_path = args.dataset
    
    # Check if the dataset file exists
    try:
        with open(dev_json_path, 'r', encoding='utf-8') as f:
            pass  # Just checking if the file exists and can be opened
    except FileNotFoundError:
        print(f"Error: The dataset file {dev_json_path} was not found.")
        exit(1)
    
    # Store results for all tasks
    all_results = {}
    
    # Process each task
    for task_name in tasks_to_run:
        print(f"\n{'=' * 50}")
        print(f"Model: {args.model}")
        print(f"Processing task: {task_name} ({TASK_CONFIG[task_name]['description']})")
        if args.instruction:
            print(f"Custom instruction: \"{args.instruction}\"")
        if args.mock:
            print("Running in MOCK mode (no actual LLM calls)")
        print(f"{'=' * 50}")
        
        # Load data for this task
        task_data = load_advglue_data(
            dev_json_path, 
            task_name, 
            subset_size=args.subset,
            random_subset=not args.sequential
        )
        
        if not task_data:
            print(f"Could not run evaluation as no {task_name} data was loaded.")
            continue
        
        print(f"\n--- Processing {len(task_data)} {task_name} examples ---")
        
        llm_predictions_parsed = []
        
        for example in task_data:
            # Get the raw LLM prediction
            raw_llm_response = get_llm_prediction_ollama(
                example, 
                task_name, 
                model_name=args.model, 
                mock_mode=args.mock,
                instruction=args.instruction,
                no_truncate=args.no_truncate
            )
            
            # Parse the potentially messy response from the LLM
            parsed_prediction = parse_llm_response(raw_llm_response, task_name)
            llm_predictions_parsed.append(parsed_prediction)
            
            # Print progress with task-specific formatting
            print(f"\nExample ID: {example.get('id', 'unknown')}")
            
            # Print input fields based on task type
            for field in TASK_CONFIG[task_name]['input_fields']:
                if field in example:
                    # Truncate long text for display if not disabled
                    text = example[field]
                    if len(text) > 80 and not args.no_truncate:
                        text = text[:77] + "..."
                    print(f"  {field.capitalize()}: \"{text}\"")
            
            print(f"  Ground Truth: {example['ground_truth_label']}")
            print(f"  Raw LLM: \"{raw_llm_response}\"")
            print(f"  Parsed LLM: {parsed_prediction}")
            print(f"  Correct: {example['ground_truth_label'] == parsed_prediction}")
            print("-" * 40)
        
        # Evaluate predictions
        print("\n--- Evaluation Results ---")
        evaluation_results = evaluate_predictions(task_data, llm_predictions_parsed, task_name)
        
        # Store results for this task
        all_results[task_name] = evaluation_results
        
        # Print basic metrics
        print(f"Task: {task_name}")
        print(f"Total Examples: {evaluation_results['total_count']}")
        print(f"Correct Predictions: {evaluation_results['correct_count']}")
        print(f"Accuracy: {evaluation_results['accuracy']:.2f}%")
        
        # Print per-class metrics for multi-class tasks
        if "class_accuracies" in evaluation_results:
            print("\nPer-class accuracies:")
            for label, accuracy in evaluation_results["class_accuracies"].items():
                print(f"  {label}: {accuracy:.2f}%")
        
        # Optionally print detailed breakdown of incorrect predictions
        print("\n--- Incorrect Predictions ---")
        incorrect_count = 0
        for detail in evaluation_results['detailed_results']:
            if not detail['correct']:
                incorrect_count += 1
                print(f"ID: {detail['id']}")
                # Print the input fields
                for field in TASK_CONFIG[task_name]['input_fields']:
                    if field in detail:
                        # Truncate long text for display
                        text = detail[field]
                        if len(text) > 80:
                            text = text[:77] + "..."
                        print(f"  {field.capitalize()}: \"{text}\"")
                print(f"  GT: {detail['ground_truth']}, Pred: {detail['predicted']}")
                print("-" * 20)
                # Limit the number of incorrect examples shown
                if incorrect_count >= 5:  # Show at most 5 incorrect examples
                    remaining = sum(1 for d in evaluation_results['detailed_results'] if not d['correct']) - 5
                    if remaining > 0:
                        print(f"... and {remaining} more incorrect predictions.")
                    break
    
    # If multiple tasks were run, print a summary
    if len(tasks_to_run) > 1:
        print("\n" + "=" * 50)
        print("SUMMARY OF ALL TASKS")
        print("=" * 50)
        for task_name, results in all_results.items():
            print(f"{task_name}: Accuracy = {results['accuracy']:.2f}% ({results['correct_count']}/{results['total_count']})")
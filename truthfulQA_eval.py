import csv
import random
import argparse
import ollama
from typing import List, Dict, Any, Tuple, Optional
import re

# --- Constants ---
NUM_CHOICES = 4  # Total number of choices for MCQ (1 correct + NUM_CHOICES-1 incorrect)
DEFAULT_MODEL_NAME = "llama3.2:3b"

# --- 1. Parsing TruthfulQA.csv ---

def load_truthfulqa_data(file_path: str, subset_size: Optional[int] = None, random_subset: bool = True) -> List[Dict[str, Any]]:
    """
    Loads and parses the TruthfulQA.csv file.

    Args:
        file_path (str): Path to the TruthfulQA.csv file.
        subset_size (int, optional): If provided, samples this many questions.
                                     Defaults to None (all questions).
        random_subset (bool): If True, randomly samples the subset. If False, takes the first
                             subset_size questions in order. Defaults to True.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                              represents a question with its potential MCQ setup.
                              Each dict will have: 'question', 'best_answer',
                              'correct_answers_list', 'incorrect_answers_list'.
    """
    questions_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                # Clean up answers: strip whitespace and filter out empty strings
                correct_answers = [ans.strip() for ans in row.get('Correct Answers', '').split(';') if ans.strip()]
                incorrect_answers = [ans.strip() for ans in row.get('Incorrect Answers', '').split(';') if ans.strip()]
                best_answer = row.get('Best Answer', '').strip()

                if not row.get('Question') or not best_answer or not incorrect_answers:
                    # print(f"Warning: Skipping row {i+1} due to missing Question, Best Answer, or sufficient Incorrect Answers.")
                    continue
                
                # Ensure best_answer is in correct_answers if not already
                if best_answer and best_answer not in correct_answers:
                    correct_answers.append(best_answer)


                questions_data.append({
                    'id': i,
                    'question': row['Question'],
                    'best_answer': best_answer,
                    'correct_answers_list': correct_answers,
                    'incorrect_answers_list': incorrect_answers,
                    'category': row.get('Category', 'Unknown')
                })
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return []
    except Exception as e:
        print(f"Error reading or parsing CSV {file_path}: {e}")
        return []

    if not questions_data:
        print("No valid questions were loaded from TruthfulQA.")
        return []

    print(f"Successfully loaded {len(questions_data)} questions from TruthfulQA.")

    if subset_size is not None and subset_size > 0 and subset_size < len(questions_data):
        if random_subset:
            print(f"Sampling {subset_size} questions randomly.")
            return random.sample(questions_data, subset_size)
        else:
            print(f"Taking the first {subset_size} questions in order.")
            return questions_data[:subset_size]
    
    return questions_data


def create_mcq_options(question_item: Dict[str, Any], num_total_choices: int) -> Optional[Tuple[List[str], str]]:
    """
    Creates a list of MCQ options (one correct, rest incorrect) and identifies the correct answer string.
    Ensures no duplicate options.
    """
    best_answer = question_item['best_answer']
    incorrect_answers = question_item['incorrect_answers_list']

    if not best_answer:
        return None # Cannot create MCQ without a best answer

    options = {best_answer} # Use a set to keep options unique initially
    
    # Add incorrect answers, ensuring they are different from the best answer and each other
    potential_distractors = [ans for ans in incorrect_answers if ans != best_answer]
    random.shuffle(potential_distractors)

    for distractor in potential_distractors:
        if len(options) < num_total_choices:
            options.add(distractor)
        else:
            break
    
    # If not enough unique distractors, it's problematic for a fair MCQ.
    # For this assignment, we'll proceed if we have at least 2 options (correct + 1 incorrect)
    if len(options) < 2:
        # print(f"Warning: Not enough unique options for question ID {question_item['id']}. Skipping MCQ generation for this item.")
        return None

    # Convert set to list for shuffling
    final_options_list = list(options)
    random.shuffle(final_options_list) # Shuffle the options
    
    return final_options_list, best_answer


# --- 2. Getting LLM Predictions (Placeholder & MCQ Prompting) ---

def get_truthfulqa_mcq_prediction_ollama(
    question: str, 
    options: List[str], 
    model_name: str = DEFAULT_MODEL_NAME,
    mock_mode: bool = False,
    instruction: str = ""
) -> str:
    """
    Gets LLM's choice for an MCQ from TruthfulQA using Ollama.
    Can run in mock mode for testing without an actual LLM.

    Args:
        question (str): The question.
        options (List[str]): A list of answer choices.
        model_name (str): The Ollama model to use.
        mock_mode (bool): If True, use mock implementation instead of calling Ollama API.
        instruction (str): Optional custom instruction to add to the prompt (e.g., 'Be truthful').

    Returns:
        str: The letter (A, B, C, ...) corresponding to the LLM's chosen answer, 
             or "error"/"unknown".
    """
    option_labels = [chr(ord('A') + i) for i in range(len(options))]
    
    formatted_options = ""
    for label, opt_text in zip(option_labels, options):
        formatted_options += f"{label}) {opt_text}\n"

    # Add instruction if provided
    instruction_text = f"{instruction}\n" if instruction else ""
    
    prompt = f"""Please answer the following multiple-choice question.
{instruction_text}Question: {question}

Options:
{formatted_options}
Respond with only the letter of the correct option (e.g., 'A', 'B', 'C', etc.)."""

    # Use the mock implementation if mock_mode is True
    if mock_mode:
        # --- Mock implementation ---
        print(f"Mock predicting for MCQ: {question[:50]}...")
        # Simulate LLM choosing randomly or a fixed option.
        # This mock is very basic. A real LLM might be more complex.
        # chosen_letter = random.choice(option_labels) 
        chosen_letter = option_labels[0] # Mock: always picks the first option after shuffling
        print(f"  Prompt sent to LLM (or mock):\n{prompt}")
        print(f"  Mock LLM chose: {chosen_letter}")
        return chosen_letter
    else:
        # --- Actual Ollama API call ---
        try:
            response = ollama.chat(
                model=model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={"temperature": 0.0} # Low temperature for deterministic choice
            )
            raw_prediction = response['message']['content'].strip().upper()
            
            # Validate if the prediction is one of the possible option letters
            if len(raw_prediction) == 1 and raw_prediction in option_labels:
                return raw_prediction
            else:
                # Attempt to find the option letter even if there's extra text
                match = re.search(r'\b([A-Z])\b', raw_prediction) # Look for a single capital letter
                if match and match.group(1) in option_labels:
                    return match.group(1)
                print(f"Warning: LLM returned an unparseable MCQ choice: '{raw_prediction}' for question: '{question[:50]}...'")
                return "unknown"
        except Exception as e:
            print(f"Error querying Ollama model {model_name}: {e}")
            return "error"


# --- 3. Evaluation ---

def evaluate_truthfulqa_mcq(
    mcq_attempts: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Evaluates the LLM's MCQ predictions.

    Args:
        mcq_attempts (List[Dict[str, Any]]): List of dicts, each containing
            'question_id', 'question_text', 'shuffled_options', 
            'correct_answer_text', 'llm_chosen_letter', 'correct_letter'.
    Returns:
        Dict[str, Any]: Evaluation metrics (e.g., accuracy).
    """
    correct_count = 0
    total_count = 0
    
    detailed_results = []

    for attempt in mcq_attempts:
        if attempt['llm_chosen_letter'] == "error" or attempt['llm_chosen_letter'] == "unknown":
            # Optionally handle errors/unknowns differently, or just count as incorrect
            is_correct = False
        else:
            is_correct = (attempt['llm_chosen_letter'] == attempt['correct_letter'])
        
        if is_correct:
            correct_count += 1
        total_count += 1
        
        detailed_results.append({
            "id": attempt['question_id'],
            "category": attempt['category'],
            "question": attempt['question_text'],
            "options_presented": attempt['shuffled_options'],
            "correct_answer_text": attempt['correct_answer_text'],
            "llm_chosen_letter": attempt['llm_chosen_letter'],
            "actual_correct_letter": attempt['correct_letter'],
            "is_correct": is_correct
        })

    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0

    return {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": total_count,
        "detailed_results": detailed_results
    }

# --- Main Execution ---

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Evaluate LLM performance on TruthfulQA")
    parser.add_argument("--subset", type=int, default=10,
                        help="Number of examples to evaluate (default: 10)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME,
                        help=f"Ollama model to use (default: {DEFAULT_MODEL_NAME})")
    parser.add_argument("--dataset", type=str, default="dataset/TruthfulQA.csv",
                        help="Path to the TruthfulQA CSV file (default: dataset/TruthfulQA.csv)")
    parser.add_argument("--num-choices", type=int, default=NUM_CHOICES,
                        help=f"Number of choices for MCQs (default: {NUM_CHOICES})")
    parser.add_argument("--mock", action="store_true",
                        help="Run in mock mode without calling Ollama API")
    parser.add_argument("--instruction", type=str, default="",
                        help="Custom instruction to add to the prompt (e.g., 'Be truthful and honest')")
    parser.add_argument("--sequential", action="store_true",
                        help="Use sequential (non-random) subset selection based on dataset order")
    
    
    args = parser.parse_args()
    
    # Path to the dataset file
    truthfulqa_csv_path = args.dataset
    
    # Check if the dataset file exists
    try:
        with open(truthfulqa_csv_path, 'r', encoding='utf-8') as f:
            pass  # Just checking if the file exists and can be opened
    except FileNotFoundError:
        print(f"Error: The dataset file {truthfulqa_csv_path} was not found.")
        exit(1)
    
    # Load TruthfulQA data
    all_truthfulqa_data = load_truthfulqa_data(
        truthfulqa_csv_path, 
        subset_size=args.subset,
        random_subset=not args.sequential
    )

    if all_truthfulqa_data:
        print(f"\n--- Preparing and Processing {len(all_truthfulqa_data)} TruthfulQA questions as MCQs ---")
        print(f"Model: {args.model}")
        if args.mock:
            print("Running in MOCK mode (no actual LLM calls)")
        if args.instruction:
            print(f"Custom instruction: \"{args.instruction}\"")
        
        
        mcq_evaluation_attempts = []

        for i, item in enumerate(all_truthfulqa_data):
            print(f"\nProcessing Question ID: {item['id']} ({i+1}/{len(all_truthfulqa_data)})")
            
            mcq_setup = create_mcq_options(item, args.num_choices)
            
            if mcq_setup is None:
                print(f"  Skipping question ID {item['id']} due to insufficient options for MCQ.")
                continue
                
            shuffled_options, correct_answer_text = mcq_setup
            
            # Determine the letter of the correct answer after shuffling
            correct_letter = ""
            option_letters = [chr(ord('A') + j) for j in range(len(shuffled_options))]
            for letter, opt_text in zip(option_letters, shuffled_options):
                if opt_text == correct_answer_text:
                    correct_letter = letter
                    break
            
            if not correct_letter:
                print(f"  Error: Could not find correct answer text '{correct_answer_text}' in shuffled options for Q_ID {item['id']}. Options: {shuffled_options}. Skipping.")
                continue

            # Get LLM's choice
            llm_chosen_letter = get_truthfulqa_mcq_prediction_ollama(
                item['question'], 
                shuffled_options,
                model_name=args.model,
                mock_mode=args.mock,
                instruction=args.instruction
            )
            
            mcq_evaluation_attempts.append({
                'question_id': item['id'],
                'category': item['category'],
                'question_text': item['question'],
                'shuffled_options': dict(zip(option_letters, shuffled_options)), # Store as dict for clarity
                'correct_answer_text': correct_answer_text,
                'llm_chosen_letter': llm_chosen_letter,
                'correct_letter': correct_letter
            })
            
            print(f"  Question: \"{item['question'][:80]}...\"")
            print("  Options presented:")
            for letter, option in zip(option_letters, shuffled_options):
                print(f"    {letter}: {option}")
            print(f"  Correct Answer Letter: {correct_letter} (Text: '{correct_answer_text}')")
            print(f"  LLM Chosen Letter: {llm_chosen_letter}")
            print("-" * 20)


        if mcq_evaluation_attempts:
            print("\n--- TruthfulQA MCQ Evaluation Results ---")
            truthfulqa_eval_metrics = evaluate_truthfulqa_mcq(mcq_evaluation_attempts)

            print(f"Total MCQs Attempted: {truthfulqa_eval_metrics['total_count']}")
            print(f"Correct LLM Choices: {truthfulqa_eval_metrics['correct_count']}")
            print(f"Accuracy: {truthfulqa_eval_metrics['accuracy']:.2f}%")

            # For more detailed review:
            # print("\n--- Detailed Breakdown (Incorrect Answers) ---")
            # for detail in truthfulqa_eval_metrics['detailed_results']:
            #     if not detail['is_correct']:
            #         print(f"Q_ID: {detail['id']} (Category: {detail['category']}) - Incorrect")
            #         # print(f"  Q: {detail['question']}")
            #         # print(f"  Options: {detail['options_presented']}")
            #         print(f"  LLM Chose: {detail['llm_chosen_letter']}, Correct was: {detail['actual_correct_letter']} ('{detail['correct_answer_text']}')")
        else:
            print("No MCQ attempts were made, cannot evaluate.")
            
    else:
        print("Could not run evaluation as no TruthfulQA data was loaded.")
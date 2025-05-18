#!/usr/bin/env python3
"""Script to analyze the HarmfulQA dataset and count unique topics and subtopics."""

import json
import os
import random
import argparse
from collections import Counter
import pprint

def display_prompt_by_id(data, prompt_id):
    """Display a specific prompt by its ID."""
    try:
        # Convert prompt_id to integer if it's a string
        prompt_id = int(prompt_id) if isinstance(prompt_id, str) else prompt_id
        
        for item in data:
            if item.get('id') == prompt_id:
                print(f"\nPrompt ID: {prompt_id}")
                print(f"Topic: {item.get('topic', 'N/A')}")
                print(f"Subtopic: {item.get('subtopic', 'N/A')}")
                print(f"Question: {item.get('question', 'N/A')}")
                
                # Check if there are conversations
                if 'blue_conversations' in item:
                    print(f"\nBlue Conversations: {len(item['blue_conversations'])}")
                if 'red_conversations' in item:
                    print(f"\nRed Conversations: {len(item['red_conversations'])}")
                    
                return True
        print(f"\nNo prompt found with ID: {prompt_id}")
        return False
    except Exception as e:
        print(f"\nError finding prompt with ID {prompt_id}: {e}")
        return False


def sample_prompts_from_topic(data, topic, sample_size=5):
    """Sample a specified number of prompts from a given topic."""
    try:
        # Find all prompts that match the topic
        matching_prompts = []
        for item in data:
            # Check both topic and subtopic fields
            item_topic = item.get('topic', '')
            item_subtopic = item.get('subtopic', '')
            
            if (isinstance(item_topic, str) and topic.lower() in item_topic.lower()) or \
               (isinstance(item_subtopic, str) and topic.lower() in item_subtopic.lower()):
                matching_prompts.append(item)
        
        if not matching_prompts:
            print(f"\nNo prompts found for topic: {topic}")
            return False
        
        # Adjust sample size if necessary
        actual_sample_size = min(sample_size, len(matching_prompts))
        samples = random.sample(matching_prompts, actual_sample_size)
        
        print(f"\nSampled {actual_sample_size} prompts from topic '{topic}' (out of {len(matching_prompts)} matching prompts):")
        for i, item in enumerate(samples, 1):
            print(f"\n{i}. Prompt ID: {item.get('id')}")
            print(f"   Topic: {item.get('topic', 'N/A')}")
            print(f"   Subtopic: {item.get('subtopic', 'N/A')}")
            print(f"   Question: {item.get('question', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"\nError sampling prompts for topic {topic}: {e}")
        return False


def main():
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description="Analyze the HarmfulQA dataset and display prompts.")
    parser.add_argument("--dataset", type=str, default="dataset/data_for_hub.json",
                        help="Path to the HarmfulQA dataset JSON file")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze the dataset structure and categories")
    parser.add_argument("--id", type=str,
                        help="Display a specific prompt by its ID")
    parser.add_argument("--topic", type=str,
                        help="Sample prompts from a specific topic")
    parser.add_argument("--sample-size", type=int, default=5,
                        help="Number of prompts to sample when using --topic (default: 5)")
    
    args = parser.parse_args()
    
    # Path to the HarmfulQA dataset
    dataset_path = args.dataset
    
    # Check if the file exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        return
    
    # Load the dataset
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    # Handle specific command-line options
    if args.id:
        # Display a specific prompt by ID
        display_prompt_by_id(data, args.id)
        return
    
    if args.topic:
        # Sample prompts from a specific topic
        sample_prompts_from_topic(data, args.topic, args.sample_size)
        return
    
    # If no specific command is given or --analyze is specified, perform analysis
    if not (args.id or args.topic) or args.analyze:
        print(f"Total prompts in dataset: {len(data)}")
        
        # First, let's examine the structure of the first item
        print("\nDataset structure sample:")
        if data:
            pprint.pprint(data[0])
    
    # Look for any field that might contain category/topic information
    print("\nExamining all fields in the dataset...")
    all_fields = set()
    for item in data:
        all_fields.update(item.keys())
    
    print(f"\nAll fields found in the dataset: {', '.join(sorted(all_fields))}")
    
    # Check if we have a category field
    if 'category' in all_fields:
        print("\nAnalyzing 'category' field...")
        categories = [item.get('category') for item in data if 'category' in item]
        unique_categories = set(categories)
        print(f"  Found {len(unique_categories)} unique categories")
        print(f"  Sample categories: {list(unique_categories)[:5]}")
        
        # Count categories
        category_counter = Counter(categories)
        print(f"\nTop 10 categories by frequency:")
        for category, count in category_counter.most_common(10):
            print(f"  {category}: {count} prompts")
        
        # Check if categories have a topic/subtopic structure (with '/')
        categories_with_slash = [cat for cat in categories if isinstance(cat, str) and '/' in cat]
        if categories_with_slash:
            print(f"\nFound {len(categories_with_slash)} categories with '/' separator")
            
            # Extract topics and subtopics
            topic_subtopic_pairs = [cat.split('/', 1) for cat in categories_with_slash]
            topics = [pair[0].strip() for pair in topic_subtopic_pairs]
            subtopics = [pair[1].strip() for pair in topic_subtopic_pairs]
            
            # Count unique topics and subtopics
            unique_topics = set(topics)
            unique_subtopics = set(subtopics)
            
            print(f"Number of unique topics: {len(unique_topics)}")
            print(f"Number of unique subtopics: {len(unique_subtopics)}")
            
            # Count frequency of each topic
            topic_counter = Counter(topics)
            print(f"\nAll topics and counts:")
            for topic, count in sorted(topic_counter.items()):
                print(f"  {topic}: {count} prompts")
            
            # Create a mapping of topics to their subtopics
            topic_to_subtopics = {}
            for topic, subtopic in topic_subtopic_pairs:
                topic = topic.strip()
                subtopic = subtopic.strip()
                
                if topic not in topic_to_subtopics:
                    topic_to_subtopics[topic] = set()
                
                topic_to_subtopics[topic].add(subtopic)
            
            print("\nTopics with their subtopics:")
            for topic, subtopics_set in sorted(topic_to_subtopics.items()):
                print(f"  {topic} ({len(subtopics_set)} subtopics):")
                
                # Count prompts for each subtopic within this topic
                subtopic_counts = Counter([pair[1].strip() for pair in topic_subtopic_pairs if pair[0].strip() == topic])
                
                for subtopic, count in sorted(subtopic_counts.items()):
                    print(f"    - {subtopic}: {count} prompts")
        else:
            print("No categories with topic/subtopic structure found.")
    else:
        print("No 'category' field found in the dataset.")
        
        # Try to find other fields that might contain topic information
        for field in ['topic', 'type', 'harm_category']:
            if field in all_fields:
                print(f"\nAnalyzing '{field}' field as alternative...")
                values = [item.get(field) for item in data if field in item and item[field]]
                unique_values = set(values)
                print(f"  Found {len(unique_values)} unique values")
                print(f"  Sample values: {list(unique_values)[:5]}")
                
                # Count frequency
                counter = Counter(values)
                print(f"\nTop 10 {field} values by frequency:")
                for value, count in counter.most_common(10):
                    print(f"  {value}: {count} prompts")
    
if __name__ == "__main__":
    main()

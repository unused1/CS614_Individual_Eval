#!/usr/bin/env python3
"""Script to analyze the HarmfulQA dataset and count unique topics and subtopics."""

import json
import os
from collections import Counter
import pprint

def main():
    # Path to the HarmfulQA dataset
    dataset_path = "dataset/data_for_hub.json"
    
    # Check if the file exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        return
    
    # Load the dataset
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
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

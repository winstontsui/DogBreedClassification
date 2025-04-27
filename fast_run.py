"""
Fast execution script for the dog breed classification project.
This script provides options to run a faster version of the pipeline.
"""

import os
import argparse
import time
from data_preparation import download_dataset, organize_dataset
from feature_extraction import extract_all_features
from feature_selection import FeatureSelector
from classifier import train_and_evaluate
import config

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Fast Dog Breed Classification")
    parser.add_argument("--skip-data-prep", action="store_true", help="Skip data preparation")
    parser.add_argument("--skip-feature-extraction", action="store_true", help="Skip feature extraction")
    parser.add_argument("--skip-feature-selection", action="store_true", help="Skip feature selection")
    parser.add_argument("--full-dataset", action="store_true", help="Use full dataset instead of limited samples")
    parser.add_argument("--all-models", action="store_true", help="Use all CNN models instead of just ResNet50")
    return parser.parse_args()

def main():
    """
    Run the optimized pipeline.
    """
    args = parse_args()
    start_time = time.time()
    
    print("=" * 50)
    print("Fast Dog Breed Classification Pipeline")
    print("Based on the paper by Ying Cui et al. (Optimized Version)")
    print("=" * 50)
    
    # Apply command line options to config
    if args.full_dataset:
        config.USE_SAMPLE_LIMIT = False
        print("Using FULL dataset (slower but more accurate)")
    else:
        print(f"Using LIMITED dataset ({config.MAX_SAMPLES_PER_CLASS} samples per class)")
    
    if args.all_models:
        config.MODELS = ["resnet50", "vgg16", "inception_v3", "densenet121"]
        print("Using ALL CNN models (slower but more accurate)")
    else:
        print(f"Using only {config.MODELS} for feature extraction")
    
    # Step 1: Data Preparation
    if not args.skip_data_prep:
        print("\n[Step 1/4] Data Preparation")
        print("-" * 50)
        download_dataset()
        organize_dataset()
    else:
        print("\n[Step 1/4] Skipping Data Preparation")
    
    # Step 2: Feature Extraction
    if not args.skip_feature_extraction:
        print("\n[Step 2/4] Feature Extraction")
        print("-" * 50)
        extract_all_features()
    else:
        print("\n[Step 2/4] Skipping Feature Extraction")
    
    # Step 3: Feature Selection
    if not args.skip_feature_selection:
        print("\n[Step 3/4] Feature Selection")
        print("-" * 50)
        selector = FeatureSelector()
        selector.select_features()
    else:
        print("\n[Step 3/4] Skipping Feature Selection")
    
    # Step 4: Classification
    print("\n[Step 4/4] Classification")
    print("-" * 50)
    accuracy = train_and_evaluate()
    
    # Print summary
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 50)
    print(f"Pipeline completed in {elapsed_time:.2f} seconds")
    print(f"Final accuracy: {accuracy * 100:.2f}%")
    print("=" * 50)
    
    # Print notes about speed vs. accuracy tradeoffs
    print("\nNotes:")
    print("- Using limited samples per class speeds up processing but may reduce accuracy")
    print("- Using fewer CNN models speeds up processing but may reduce accuracy")
    print("- The simplified feature selection is much faster but may be less optimal")
    print("- To get results closer to the paper, use --full-dataset --all-models")

if __name__ == "__main__":
    main()

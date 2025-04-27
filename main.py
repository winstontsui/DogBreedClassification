"""
Main script for the dog breed classification project.
Runs the entire pipeline from data preparation to classification.
1. Download and prepare the Stanford Dog Dataset
2. Extract features using four CNN models (ResNet50, VGG16, InceptionV3, DenseNet121)
3. Apply feature selection using PCA and GWO
4. Train an SVM classifier and evaluate its performance
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
    parser = argparse.ArgumentParser(description="Dog Breed Classification")
    parser.add_argument("--skip-data-prep", action="store_true", help="Skip data preparation")
    parser.add_argument("--skip-feature-extraction", action="store_true", help="Skip feature extraction")
    parser.add_argument("--skip-feature-selection", action="store_true", help="Skip feature selection")
    return parser.parse_args()

def main():
    """
    Run the entire pipeline.
    """
    args = parse_args()
    start_time = time.time()
    
    print("=" * 50)
    print("Dog Breed Classification Pipeline")
    print("Based on the paper by Ying Cui et al.")
    print("=" * 50)
    
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

if __name__ == "__main__":
    main()

"""
The system will download the Stanford Dog Dataset (~794MB) which contains images of 120 different dog breeds. After downloading, it will:
Extract the dataset
Organize it into training and testing sets (80/20 split)
What's Coming Next:

Feature Extraction (Step 2/4)
The system will use four pre-trained CNN models to extract features from the dog images:
ResNet50
VGG16
InceptionV3
DenseNet121
This is the most time-consuming part of the process and may take a while depending on your hardware

Feature Selection (Step 3/4)
The extracted features from all four models will be combined
Principal Component Analysis (PCA) will be applied to reduce dimensionality
Gray Wolf Optimization (GWO) will be used to select the most important features

Classification (Step 4/4)
A Support Vector Machine (SVM) classifier will be trained on the selected features
The model will be evaluated on the test set
Performance metrics including accuracy and a confusion matrix will be generated
Expected Results
Based on the paper by Ying Cui et al., we can expect classification accuracy of:

Around 95.24% for all 120 breeds
Up to 99.34% for a subset of 76 selected breeds
"""
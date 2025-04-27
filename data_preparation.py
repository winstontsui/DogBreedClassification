"""
Data preparation script for the dog breed classification project.
Downloads and organizes the Stanford Dog Dataset.
"""

import os
import shutil
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import config
from utils import download_file, extract_tar, set_seed

def download_dataset():
    """
    Download the Stanford Dog Dataset.
    """
    # Create data directory if it doesn't exist
    if not os.path.exists(config.DATA_DIR):
        os.makedirs(config.DATA_DIR)
    
    # Download images
    images_tar = os.path.join(config.DATA_DIR, "images.tar")
    download_file(config.IMAGES_URL, images_tar)
    
    # Download annotations
    annotations_tar = os.path.join(config.DATA_DIR, "annotation.tar")
    download_file(config.ANNOTATIONS_URL, annotations_tar)
    
    # Extract files
    extract_tar(images_tar, config.DATA_DIR)
    extract_tar(annotations_tar, config.DATA_DIR)
    
    print("Dataset downloaded and extracted successfully.")

def organize_dataset():
    """
    Organize the dataset into train and test sets.
    """
    # Set random seed for reproducibility
    set_seed()
    
    # Create train and test directories
    if not os.path.exists(config.TRAIN_DIR):
        os.makedirs(config.TRAIN_DIR)
    if not os.path.exists(config.TEST_DIR):
        os.makedirs(config.TEST_DIR)
    
    # Get list of breed directories
    images_dir = os.path.join(config.DATA_DIR, "Images")
    breed_dirs = [os.path.join(images_dir, d) for d in os.listdir(images_dir) 
                 if os.path.isdir(os.path.join(images_dir, d))]
    
    # Process each breed
    for breed_dir in tqdm(breed_dirs, desc="Processing breeds"):
        breed_name = os.path.basename(breed_dir)
        
        # Create breed directories in train and test
        train_breed_dir = os.path.join(config.TRAIN_DIR, breed_name)
        test_breed_dir = os.path.join(config.TEST_DIR, breed_name)
        
        if not os.path.exists(train_breed_dir):
            os.makedirs(train_breed_dir)
        if not os.path.exists(test_breed_dir):
            os.makedirs(test_breed_dir)
        
        # Get all images for the breed
        image_files = [f for f in os.listdir(breed_dir) if f.endswith('.jpg')]
        
        # Limit samples per class if enabled in config
        if config.USE_SAMPLE_LIMIT and len(image_files) > config.MAX_SAMPLES_PER_CLASS:
            # Randomly sample a subset of images
            random.seed(config.RANDOM_SEED)
            image_files = random.sample(image_files, config.MAX_SAMPLES_PER_CLASS)
            print(f"Limiting {breed_name} to {config.MAX_SAMPLES_PER_CLASS} samples")
        
        # Split into train and test
        train_files, test_files = train_test_split(
            image_files, 
            test_size=config.TEST_RATIO,
            random_state=config.RANDOM_SEED
        )
        
        # Copy files to train directory
        for file in train_files:
            src = os.path.join(breed_dir, file)
            dst = os.path.join(train_breed_dir, file)
            shutil.copy(src, dst)
        
        # Copy files to test directory
        for file in test_files:
            src = os.path.join(breed_dir, file)
            dst = os.path.join(test_breed_dir, file)
            shutil.copy(src, dst)
    
    print("Dataset organized into train and test sets.")
    
    # Print statistics
    train_breeds = os.listdir(config.TRAIN_DIR)
    test_breeds = os.listdir(config.TEST_DIR)
    
    print(f"Number of breeds: {len(train_breeds)}")
    print(f"Train set: {sum(len(os.listdir(os.path.join(config.TRAIN_DIR, b))) for b in train_breeds)} images")
    print(f"Test set: {sum(len(os.listdir(os.path.join(config.TEST_DIR, b))) for b in test_breeds)} images")

if __name__ == "__main__":
    print("Starting data preparation...")
    download_dataset()
    organize_dataset()
    print("Data preparation completed.")

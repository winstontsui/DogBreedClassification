"""
Utility functions for the dog breed classification project.
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from tqdm import tqdm
import tarfile
import requests
from config import RANDOM_SEED

def set_seed(seed=RANDOM_SEED):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def download_file(url, destination):
    """
    Download a file from a URL to a destination.
    """
    if os.path.exists(destination):
        print(f"File {destination} already exists. Skipping download.")
        return
    
    print(f"Downloading {url} to {destination}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(destination, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

def extract_tar(tar_path, extract_path):
    """
    Extract a tar file to a directory.
    """
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    
    print(f"Extracting {tar_path} to {extract_path}")
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=extract_path)

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def print_classification_report(y_true, y_pred, class_names):
    """
    Print classification report.
    """
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    
    return accuracy

def plot_feature_importance(feature_importance, feature_names, top_n=20, save_path=None):
    """
    Plot feature importance.
    """
    # Get the indices of the top N features
    top_indices = np.argsort(feature_importance)[-top_n:]
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.barh(range(top_n), feature_importance[top_indices])
    plt.yticks(range(top_n), [feature_names[i] for i in top_indices])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Feature Importance')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def save_features(features, labels, filename):
    """
    Save features and labels to a file.
    """
    np.savez(filename, features=features, labels=labels)

def load_features(filename):
    """
    Load features and labels from a file.
    """
    data = np.load(filename)
    return data['features'], data['labels']

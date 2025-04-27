"""
Classifier script for the dog breed classification project.
Implements Support Vector Machine (SVM) for classification as described in the paper.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import config
from utils import load_features, plot_confusion_matrix, print_classification_report, set_seed

class SVMClassifier:
    def __init__(self):
        """
        Initialize the SVM classifier.
        """
        # Set random seed for reproducibility
        set_seed()
        
        # Initialize SVM
        self.svm = SVC(
            C=config.SVM_C,
            kernel=config.SVM_KERNEL,
            gamma=config.SVM_GAMMA,
            probability=True,
            random_state=config.RANDOM_SEED
        )
    
    def train(self, X_train, y_train):
        """
        Train the SVM classifier.
        
        Args:
            X_train (numpy.ndarray): Training features.
            y_train (numpy.ndarray): Training labels.
        """
        print("Training SVM classifier...")
        self.svm.fit(X_train, y_train)
        print("Training completed.")
    
    def predict(self, X_test):
        """
        Make predictions using the trained SVM.
        
        Args:
            X_test (numpy.ndarray): Test features.
        
        Returns:
            numpy.ndarray: Predicted labels.
        """
        return self.svm.predict(X_test)
    
    def evaluate(self, X_test, y_test, class_names):
        """
        Evaluate the classifier on the test set.
        
        Args:
            X_test (numpy.ndarray): Test features.
            y_test (numpy.ndarray): Test labels.
            class_names (list): List of class names.
        
        Returns:
            float: Accuracy score.
        """
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test accuracy: {accuracy * 100:.2f}%")
        
        # Print classification report
        print("\nClassification Report:")
        print_classification_report(y_test, y_pred, class_names)
        
        # Plot confusion matrix
        os.makedirs("results", exist_ok=True)
        plot_confusion_matrix(
            y_test, 
            y_pred, 
            class_names,
            save_path="results/confusion_matrix.png"
        )
        
        return accuracy

def train_and_evaluate():
    """
    Train and evaluate the SVM classifier on the selected features.
    """
    # Load selected features
    train_features, train_labels = load_features(os.path.join(config.FEATURE_DIR, "selected_train.npz"))
    test_features, test_labels = load_features(os.path.join(config.FEATURE_DIR, "selected_test.npz"))
    
    # Load class names
    with open(os.path.join(config.FEATURE_DIR, "class_names.txt"), "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    
    # Initialize and train classifier
    classifier = SVMClassifier()
    classifier.train(train_features, train_labels)
    
    # Evaluate classifier
    accuracy = classifier.evaluate(test_features, test_labels, class_names)
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/accuracy.txt", "w") as f:
        f.write(f"Test accuracy: {accuracy * 100:.2f}%")
    
    return accuracy

if __name__ == "__main__":
    print("Starting classifier training and evaluation...")
    train_and_evaluate()
    print("Classifier evaluation completed.")

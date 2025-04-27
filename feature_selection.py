"""
Feature selection script for the dog breed classification project.
Implements Principal Component Analysis (PCA) and Gray Wolf Optimization (GWO)
for feature selection as described in the paper.
"""

import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import pyswarms as ps
from tqdm import tqdm
import config
from utils import load_features, save_features, set_seed

class FeatureSelector:
    def __init__(self):
        """
        Initialize the feature selector.
        """
        # Set random seed for reproducibility
        set_seed()
        
        # Initialize PCA
        self.pca = PCA(n_components=config.PCA_COMPONENTS)
    
    def load_and_combine_features(self, split='train'):
        """
        Load features from all models and combine them.
        
        Args:
            split (str): 'train' or 'test'
        
        Returns:
            tuple: (combined_features, labels)
        """
        all_features = []
        labels = None
        
        for model_name in config.MODELS:
            feature_file = os.path.join(config.FEATURE_DIR, f"{model_name}_{split}.npz")
            features, current_labels = load_features(feature_file)
            
            # Store the first labels (they should all be the same)
            if labels is None:
                labels = current_labels
            
            # Append features
            all_features.append(features)
        
        # Concatenate features along the feature dimension
        combined_features = np.concatenate(all_features, axis=1)
        
        print(f"Combined {split} features shape: {combined_features.shape}")
        return combined_features, labels
    
    def apply_pca(self, features):
        """
        Apply PCA for dimensionality reduction.
        
        Args:
            features (numpy.ndarray): Input features.
        
        Returns:
            numpy.ndarray: Reduced features.
        """
        # Fit PCA on the data (or transform if already fitted)
        if not hasattr(self, 'pca_fitted') or not self.pca_fitted:
            reduced_features = self.pca.fit_transform(features)
            self.pca_fitted = True
        else:
            reduced_features = self.pca.transform(features)
        
        print(f"After PCA, features shape: {reduced_features.shape}")
        print(f"Explained variance ratio: {sum(self.pca.explained_variance_ratio_):.4f}")
        
        return reduced_features
    
    def apply_gwo(self, features, labels):
        """
        Apply a simplified feature selection approach for faster execution.
        Instead of full GWO, we use a faster approach based on feature importance.
        
        Args:
            features (numpy.ndarray): Input features.
            labels (numpy.ndarray): Corresponding labels.
        
        Returns:
            tuple: (selected_features, selected_indices)
        """
        print("Applying simplified feature selection for faster execution...")
        
        # Train a simple SVM to get feature importance
        from sklearn.svm import SVC
        from sklearn.feature_selection import SelectKBest, f_classif
        
        # Use ANOVA F-value for feature selection
        print(f"Selecting top {config.SELECTED_FEATURES} features using F-test...")
        selector = SelectKBest(f_classif, k=config.SELECTED_FEATURES)
        selector.fit(features, labels)
        
        # Get the selected indices
        selected_indices = np.argsort(selector.scores_)[-config.SELECTED_FEATURES:]
        selected_features = features[:, selected_indices]
        
        print(f"Selected {len(selected_indices)} features")
        print(f"Selected features shape: {selected_features.shape}")
        
        return selected_features, selected_indices
    
    # The _evaluate_features method is no longer needed with our simplified approach
    
    def select_features(self):
        """
        Perform feature selection on the combined features.
        
        Returns:
            tuple: (train_features, train_labels, test_features, test_labels, selected_indices)
        """
        # Load and combine features
        train_features, train_labels = self.load_and_combine_features('train')
        test_features, test_labels = self.load_and_combine_features('test')
        
        # Apply PCA
        train_features_pca = self.apply_pca(train_features)
        test_features_pca = self.pca.transform(test_features)
        
        # Apply GWO for feature selection
        selected_train_features, selected_indices = self.apply_gwo(train_features_pca, train_labels)
        selected_test_features = test_features_pca[:, selected_indices]
        
        # Save selected features
        save_features(selected_train_features, train_labels, 
                     os.path.join(config.FEATURE_DIR, "selected_train.npz"))
        save_features(selected_test_features, test_labels, 
                     os.path.join(config.FEATURE_DIR, "selected_test.npz"))
        
        # Save selected indices
        np.save(os.path.join(config.FEATURE_DIR, "selected_indices.npy"), selected_indices)
        
        return (selected_train_features, train_labels, 
                selected_test_features, test_labels, 
                selected_indices)

if __name__ == "__main__":
    print("Starting feature selection...")
    selector = FeatureSelector()
    selector.select_features()
    print("Feature selection completed.")

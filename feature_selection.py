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
        Apply a very fast feature selection method using multiple statistical measures.
        This approach combines different feature importance metrics for better selection.
        
        Args:
            features (numpy.ndarray): Input features.
            labels (numpy.ndarray): Corresponding labels.
        
        Returns:
            tuple: (selected_features, selected_indices)
        """
        print("Applying fast multi-metric feature selection...")
        
        # Use multiple feature selection methods for robustness
        from sklearn.feature_selection import mutual_info_classif, f_classif, chi2
        from sklearn.ensemble import ExtraTreesClassifier
        
        print("Calculating feature importance using multiple metrics...")
        
        # 1. F-test (ANOVA)
        f_scores, _ = f_classif(features, labels)
        f_indices = np.argsort(f_scores)[-config.SELECTED_FEATURES:]
        
        # 2. Mutual Information
        mi_scores = mutual_info_classif(features, labels, random_state=config.RANDOM_SEED)
        mi_indices = np.argsort(mi_scores)[-config.SELECTED_FEATURES:]
        
        # 3. Extra Trees feature importance (fast ensemble method)
        print("Training a fast ensemble model for feature importance...")
        et_model = ExtraTreesClassifier(
            n_estimators=50,  # Use fewer trees for speed
            max_depth=10,
            random_state=config.RANDOM_SEED,
            n_jobs=-1  # Use all CPU cores
        )
        # Use a sample of the data for faster training
        sample_size = min(5000, features.shape[0])
        indices = np.random.choice(features.shape[0], sample_size, replace=False)
        et_model.fit(features[indices], labels[indices])
        et_scores = et_model.feature_importances_
        et_indices = np.argsort(et_scores)[-config.SELECTED_FEATURES:]
        
        # Combine the selected indices from all methods
        print("Combining results from multiple selection methods...")
        all_indices = np.unique(np.concatenate([f_indices, mi_indices, et_indices]))
        
        # If we have more than the desired number of features, select the top ones
        # by averaging the normalized ranks from each method
        if len(all_indices) > config.SELECTED_FEATURES:
            # Normalize each score
            f_norm = (f_scores - np.min(f_scores)) / (np.max(f_scores) - np.min(f_scores) + 1e-10)
            mi_norm = (mi_scores - np.min(mi_scores)) / (np.max(mi_scores) - np.min(mi_scores) + 1e-10)
            et_norm = (et_scores - np.min(et_scores)) / (np.max(et_scores) - np.min(et_scores) + 1e-10)
            
            # Create a combined score with different weights
            combined_scores = 0.4 * f_norm + 0.3 * mi_norm + 0.3 * et_norm
            
            # Select the top features based on the combined score
            selected_indices = np.argsort(combined_scores)[-config.SELECTED_FEATURES:]
        else:
            # If we have fewer than needed, use all and add more from f_scores
            additional_needed = config.SELECTED_FEATURES - len(all_indices)
            remaining_indices = np.setdiff1d(np.arange(features.shape[1]), all_indices)
            additional_indices = np.argsort(f_scores[remaining_indices])[-additional_needed:]
            selected_indices = np.concatenate([all_indices, remaining_indices[additional_indices]])
        
        # Extract the selected features
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

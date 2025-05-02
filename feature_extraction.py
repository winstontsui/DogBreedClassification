"""
Feature extraction script for the dog breed classification project.
Extracts features from four pre-trained CNN models:
1. ResNet50
2. VGG16
3. InceptionV3
4. DenseNet121
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from tqdm import tqdm
import config
from utils import set_seed, save_features

class FeatureExtractor:
    def __init__(self, model_name):
        """
        Initialize the feature extractor with a specific model.
        
        Args:
            model_name (str): Name of the model to use for feature extraction.
                              Must be one of: 'resnet50', 'vgg16', 'inception_v3', 'densenet121'
        """
        self.model_name = model_name
        # Use MPS (Metal Performance Shaders) for Mac Silicon GPU if available
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        
        # Set random seed for reproducibility
        set_seed()
        
        # Load pre-trained model
        self.model = self._load_model()
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define transformations
        self.transform = self._get_transforms()
    
    def _load_model(self):
        """
        Load the pre-trained model.
        """
        if self.model_name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the last FC layer
        
        elif self.model_name == "vgg16":
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the classifier
        
        elif self.model_name == "inception_v3":
            # Special handling for Inception v3
            model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=True)
            # We need to modify the model to only use it for feature extraction
            # Set aux_logits to False to avoid the auxiliary output
            model.aux_logits = False
            # Remove the final fully connected layer
            model.fc = torch.nn.Identity()
            # Create a wrapper to get the output before the final classifier
            class InceptionFeatureExtractor(torch.nn.Module):
                def __init__(self, inception_model):
                    super(InceptionFeatureExtractor, self).__init__()
                    self.model = inception_model
                
                def forward(self, x):
                    # Forward pass through the model but get features before classification
                    x = self.model.Conv2d_1a_3x3(x)
                    x = self.model.Conv2d_2a_3x3(x)
                    x = self.model.Conv2d_2b_3x3(x)
                    x = self.model.maxpool1(x)
                    x = self.model.Conv2d_3b_1x1(x)
                    x = self.model.Conv2d_4a_3x3(x)
                    x = self.model.maxpool2(x)
                    x = self.model.Mixed_5b(x)
                    x = self.model.Mixed_5c(x)
                    x = self.model.Mixed_5d(x)
                    x = self.model.Mixed_6a(x)
                    x = self.model.Mixed_6b(x)
                    x = self.model.Mixed_6c(x)
                    x = self.model.Mixed_6d(x)
                    x = self.model.Mixed_6e(x)
                    x = self.model.Mixed_7a(x)
                    x = self.model.Mixed_7b(x)
                    x = self.model.Mixed_7c(x)
                    # Global average pooling
                    x = self.model.avgpool(x)
                    # Flatten to get feature vector
                    x = torch.flatten(x, 1)
                    return x
            
            model = InceptionFeatureExtractor(model)
        
        elif self.model_name == "densenet121":
            model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the classifier
        
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        return model
    
    def _get_transforms(self):
        """
        Get the transformations for the model.
        """
        if self.model_name == "inception_v3":
            # Inception v3 expects 299x299 images
            transform = transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # Other models expect 224x224 images
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        return transform
    
    def extract_features(self, data_dir):
        """
        Extract features from the images in the data directory.
        
        Args:
            data_dir (str): Path to the directory containing the images.
        
        Returns:
            tuple: (features, labels, class_names)
                - features (numpy.ndarray): Extracted features.
                - labels (numpy.ndarray): Corresponding labels.
                - class_names (list): List of class names.
        """
        # Create dataset and dataloader
        dataset = datasets.ImageFolder(data_dir, transform=self.transform)
        dataloader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS
        )
        
        # Extract features
        features = []
        labels = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc=f"Extracting features with {self.model_name}"):
                inputs = inputs.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Reshape the output to a flat vector for each image
                outputs = outputs.reshape(outputs.size(0), -1)
                
                # Convert to numpy and store
                features.append(outputs.cpu().numpy())
                labels.append(targets.numpy())
        
        # Concatenate batches
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        
        # Get class names
        class_names = [c for c, _ in dataset.class_to_idx.items()]
        
        return features, labels, class_names

def extract_all_features():
    """
    Extract features from all models for both train and test sets.
    """
    # Create feature directory if it doesn't exist
    if not os.path.exists(config.FEATURE_DIR):
        os.makedirs(config.FEATURE_DIR)
    
    # Extract features for each model
    for model_name in config.MODELS:
        print(f"\nExtracting features using {model_name}...")
        extractor = FeatureExtractor(model_name)
        
        # Extract training features
        print("Processing training set...")
        train_features, train_labels, class_names = extractor.extract_features(config.TRAIN_DIR)
        
        # Extract test features
        print("Processing test set...")
        test_features, test_labels, _ = extractor.extract_features(config.TEST_DIR)
        
        # Save features
        train_file = os.path.join(config.FEATURE_DIR, f"{model_name}_train.npz")
        test_file = os.path.join(config.FEATURE_DIR, f"{model_name}_test.npz")
        
        save_features(train_features, train_labels, train_file)
        save_features(test_features, test_labels, test_file)
        
        print(f"Features saved to {train_file} and {test_file}")
        print(f"Train features shape: {train_features.shape}")
        print(f"Test features shape: {test_features.shape}")
    
    # Save class names
    with open(os.path.join(config.FEATURE_DIR, "class_names.txt"), "w") as f:
        for name in class_names:
            f.write(f"{name}\n")
    
    print("\nFeature extraction completed.")

if __name__ == "__main__":
    print("Starting feature extraction...")
    extract_all_features()

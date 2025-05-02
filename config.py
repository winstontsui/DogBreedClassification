"""
Configuration parameters for the dog breed classification project.
"""

# Dataset parameters
DATASET_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs/"
IMAGES_URL = DATASET_URL + "images.tar"
ANNOTATIONS_URL = DATASET_URL + "annotation.tar"
DATA_DIR = "./data"
TRAIN_DIR = f"{DATA_DIR}/train"
TEST_DIR = f"{DATA_DIR}/test"
TRAIN_RATIO = 0.8
TEST_RATIO = 0.2
IMAGE_SIZE = 224  # Standard input size for most CNN models

# Dataset sampling for faster processing
MAX_SAMPLES_PER_CLASS = 50  # Limit samples per class for faster processing
USE_SAMPLE_LIMIT = False  # Set to False to use full dataset

# Feature extraction parameters
BATCH_SIZE = 64  # Increased batch size for faster processing
NUM_WORKERS = 4  # Adjust based on your CPU cores
FEATURE_DIR = "./features"

# CNN models to use for feature extraction
# Comment out models to use fewer for faster processing
MODELS = [
    "resnet50",  # Fastest model
    "vgg16",     # Uncomment to use more models (slower)
    "inception_v3",
    "densenet121",
]

# Feature selection parameters
PCA_COMPONENTS = 512  # Increased number of components to keep after PCA
GWO_POPULATION = 10   # Increased population size for GWO
GWO_ITERATIONS = 20   # Increased number of iterations for GWO
SELECTED_FEATURES = 256  # Increased number of features to select after GWO

# SVM parameters
SVM_C = 100.0  # Increased regularization parameter for better fit
SVM_KERNEL = 'rbf'  # Kernel type (radial basis function works well for image features)
SVM_GAMMA = 'scale'  # Kernel coefficient

# Training parameters
RANDOM_SEED = 42

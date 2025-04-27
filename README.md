# Dog Breed Classification


1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the data preparation script to download and preprocess the dataset:
   ```
   python data_preparation.py
   ```

3. Run the main script to extract features, perform feature selection, and train the classifier:
   ```
   python main.py
   ```
* Note: The full pipeline may take a while to run, so you may want to use the fast_run.py script to run a faster version of the pipeline.  
   ```
   python3 fast_run.py --skip-data-prep
   ``` 

This project replicates the approach described in the research paper "Classification of Dog Breeds Using Convolutional Neural Network Models and Support Vector Machine" by Ying Cui, Bixia Tang, Gangao Wu, Lun Li, Xin Zhang, Zhenglin Du, and Wenming Zhao.

## Overview

The implementation follows the methodology outlined in the paper:

1. **Data Acquisition**: Uses the Stanford Dog Dataset (120 breeds)
2. **Feature Extraction**: Extracts features from four pre-trained CNN models:
   - ResNet50
   - VGG16
   - InceptionV3
   - DenseNet121
3. **Feature Fusion and Selection**:
   - Combines features from the four CNN models
   - Applies Principal Component Analysis (PCA) for dimensionality reduction
   - Uses Gray Wolf Optimization (GWO) for feature selection
4. **Classification**: Employs Support Vector Machine (SVM) for the final classification

## Project Structure

- `data_preparation.py`: Downloads and prepares the Stanford Dog Dataset
- `feature_extraction.py`: Extracts features from the four CNN models
- `feature_selection.py`: Implements PCA and GWO for feature selection
- `classifier.py`: Implements the SVM classifier
- `main.py`: Main script to run the entire pipeline
- `utils.py`: Utility functions
- `config.py`: Configuration parameters



## Results

The paper reports classification accuracy of:
- 95.24% for 120 breeds
- 99.34% for 76 selected breeds


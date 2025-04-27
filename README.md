# Dog Breed Classification


1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the data preparation script to download and preprocess the dataset:
   ```
   python data_preparation.py
   ```

3. The fast_run.py script runs a fast version of the pipeline. It extracts features, performs feature selection, and trains the classifier. 
   ```
   python3 fast_run.py --skip-data-prep
   ``` 
   
I have not tested the main script yet, this is for the future:
   ```
   python main.py
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


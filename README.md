# Bird Call Classification with Machine Learning

A machine learning project that classifies bird species from 3-second audio clips using MFCC (Mel-Frequency Cepstral Coefficients) feature extraction and k-Nearest Neighbors classification.

## Project Overview

This project implements an audio classification pipeline to identify bird species from short audio recordings. The system processes raw audio data, extracts meaningful features, and uses machine learning to make predictions on new audio samples.

## Technical Implementation

### Data Processing
- **Training Data**: 12,000 audio files (.wav format, 24,000 samples each)
- **Test Data**: 3,000 audio files for final predictions
- **Audio Format**: 44.1 kHz sampling rate, 3-second clips

### Feature Engineering
- **MFCC Feature Extraction**: Extracted 13 Mel-Frequency Cepstral Coefficients from each audio clip
- **Feature Standardization**: Applied StandardScaler to normalize features and improve model robustness
- **Dimensionality**: Reduced from 24,000 raw audio samples to 13 MFCC features per clip

### Machine Learning Pipeline
- **Algorithm**: k-Nearest Neighbors (k-NN) classifier
- **Hyperparameter Tuning**: Grid search with the following parameters:
  - `n_neighbors`: [3, 5, 10, 20]
  - `weights`: ['uniform', 'distance']
- **Cross-Validation**: 5-fold cross-validation for model evaluation
- **Train-Test Split**: 75% training, 25% validation with stratified sampling

### Model Performance
- **Best Parameters**: n_neighbors=5, weights='distance'
- **Training Accuracy**: 100.00%
- **Validation Accuracy**: 75.67%
- **Cross-Validation Accuracy**: 73.42%

## Technologies Used
- **Python**: Core programming language
- **Scikit-learn**: Machine learning algorithms and model evaluation
- **Librosa**: Audio processing and MFCC feature extraction
- **NumPy**: Numerical computations and array operations
- **Matplotlib**: Data visualization (imported for potential plotting)
- **SciPy**: Audio file I/O operations

## Key Features
- Robust audio preprocessing pipeline
- Feature extraction optimized for audio classification
- Hyperparameter optimization using grid search
- Cross-validation for reliable performance estimation
- Competition-ready submission format generation

## Output
The system generates predictions for the test dataset and exports results in CSV format suitable for competition submission, with columns for sample ID and predicted bird species label.

# Fake News Detection System

This project implements a fake news detection system using Natural Language Processing (NLP) and machine learning techniques. The system uses BERT embeddings and a Random Forest classifier to classify news articles as real or fake.

## Features

- Text preprocessing with NLTK
- Feature extraction using BERT embeddings
- Machine learning model for classification
- Evaluation metrics and visualization
- Easy-to-use API for predictions

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your dataset in a CSV format with two columns:
   - `text`: The news article text
   - `label`: 0 for real news, 1 for fake news

2. Run the main script:
```bash
python main.py
```

## Project Structure

- `preprocess.py`: Text preprocessing module
- `feature_extractor.py`: Feature extraction using BERT and TF-IDF
- `model.py`: Machine learning model implementation
- `main.py`: Main script to run the system
- `requirements.txt`: Project dependencies

## Customization

You can customize the system by:

1. Modifying the preprocessing steps in `preprocess.py`
2. Changing the feature extraction method in `feature_extractor.py`
3. Using a different classifier in `model.py`
4. Adding your own dataset in `main.py`

## Note

The current implementation uses a small sample dataset for demonstration. For production use, you should:

1. Use a larger, more diverse dataset
2. Fine-tune the BERT model on your specific domain
3. Implement cross-validation for better model evaluation
4. Add more sophisticated feature engineering
5. Consider using a more advanced model architecture

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
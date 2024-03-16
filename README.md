# financial-sentiment-analysis

## Overview
This project presents a comprehensive sentiment analysis framework leveraging JAX, Flax, TensorFlow, and various machine learning techniques. It includes a multi-layer perceptron (MLP) model for sentiment classification, data preprocessing, sentiment scoring, and sentiment-based stock grouping. The framework is designed to process and analyze text data for sentiment, specifically targeting stock descriptions for sentiment-driven investment strategies.

## Components

### MLP Model
- Implemented using Flax Linen API.
- Comprises two dense layers with ReLU activation and softmax output, initialized using Xavier uniform distribution.

### Data Preprocessing and Training
- Custom training logic with Adam optimization.
- Data preprocessing includes reading CSV files, translating descriptions to English, and generating sentiment scores.
- Utilizes batch sampling and training steps with custom loss calculation.

### Sentiment Analysis
- Employs Google Translator for language translation.
- Uses a sentiment intensity analyzer for scoring text sentiment.
- Groups stocks based on computed sentiment scores into categories such as negative, positive, and ambiguous.

### Tokenization and Embedding
- Tokenizes descriptions using Jieba (for Chinese text) and prepares them for sentiment analysis.
- Implements an autoencoder for embedding generation and applies KMeans clustering for grouping text data.

## Key Libraries
- **Flax and JAX:** For defining the MLP model and leveraging GPU acceleration in computations.
- **TensorFlow:** For tokenization, sequence padding, and the autoencoder model.
- **Pandas:** For data manipulation and CSV file processing.
- **Deep Translator:** For translating non-English text to English.
- **NLTK:** For sentiment analysis and tokenization.

## Functionality
- The framework reads stock descriptions from a CSV file, processes the text data through tokenization and translation, computes sentiment scores, and clusters stocks based on their sentiments.
- It showcases the application of deep learning models and natural language processing (NLP) techniques in financial sentiment analysis.

## Usage
- Users can initialize the framework with a CSV file path and other parameters such as learning rate and batch size for training.
- The framework provides functionality for sentiment analysis, tokenization, autoencoder-based embedding generation, and clustering.

## Conclusion
This project demonstrates the integration of advanced machine learning and NLP techniques for sentiment analysis, particularly in the context of financial decision-making. It leverages the power of modern deep learning libraries to process, analyze, and classify text data for actionable insights.


import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import torch
from scipy.sparse import hstack
import re

class MinimalFeatureExtractor:
    def __init__(self):
        # Load BERT model
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),
            stop_words='english'
        )
    
    def extract_bert_features(self, texts):
        """Extract BERT embeddings"""
        embeddings = []
        for text in texts:
            inputs = self.bert_tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            )
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # Use mean pooling of all token embeddings
                token_embeddings = outputs.last_hidden_state
                sentence_embedding = torch.mean(token_embeddings, dim=1)
                embeddings.append(sentence_embedding.numpy())
        return np.vstack(embeddings)
    
    def extract_text_features(self, texts):
        """Extract text-based features using TextBlob"""
        features = []
        for text in texts:
            # Basic text statistics
            words = text.split()
            sentences = text.split('.')
            
            # Create TextBlob object for sentiment analysis
            blob = TextBlob(text)
            sentiment = blob.sentiment
            
            # Calculate features
            feature_vector = [
                len(words),  # Total words
                len(sentences),  # Number of sentences
                len(set(words)),  # Unique words
                len(words) / len(sentences) if len(sentences) > 0 else 0,  # Average sentence length
                sentiment.polarity,  # Sentiment polarity
                sentiment.subjectivity,  # Sentiment subjectivity
                len([w for w in words if w[0].isupper()]),  # Proper nouns
                len(re.findall(r'[!?]', text)),  # Number of exclamation/question marks
                len(re.findall(r'\d+', text)),  # Number of digits
                len(re.findall(r'[A-Z][a-z]+', text))  # Number of capitalized words
            ]
            features.append(feature_vector)
        return np.array(features)
    
    def extract_tfidf_features(self, texts):
        """Extract TF-IDF features"""
        return self.tfidf_vectorizer.fit_transform(texts)
    
    def fit_transform(self, texts):
        """Extract all features and combine them"""
        # BERT features
        bert_features = self.extract_bert_features(texts)
        
        # Text features
        text_features = self.extract_text_features(texts)
        
        # TF-IDF features
        tfidf_features = self.extract_tfidf_features(texts)
        
        # Combine all features
        combined_features = hstack([
            tfidf_features,
            text_features
        ]).toarray()
        
        # Concatenate with BERT features
        final_features = np.hstack([bert_features, combined_features])
        
        return final_features
    
    def transform(self, texts):
        """Transform new texts using fitted vectorizers"""
        # BERT features
        bert_features = self.extract_bert_features(texts)
        
        # Text features
        text_features = self.extract_text_features(texts)
        
        # TF-IDF features
        tfidf_features = self.tfidf_vectorizer.transform(texts)
        
        # Combine all features
        combined_features = hstack([
            tfidf_features,
            text_features
        ]).toarray()
        
        # Concatenate with BERT features
        final_features = np.hstack([bert_features, combined_features])
        
        return final_features 
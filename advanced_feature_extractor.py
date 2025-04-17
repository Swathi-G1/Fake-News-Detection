import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import torch
from scipy.sparse import hstack
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

class AdvancedFeatureExtractor:
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
        
        # Technical terms to preserve
        self.technical_terms = {
            'isro', 'nasa', 'spacex', 'esa', 'roscosmos', 'jaxa',
            'gaganyaan', 'chandrayaan', 'mangalyaan', 'artemis',
            'perseverance', 'curiosity', 'falcon', 'starship',
            'iss', 'hubble', 'webb', 'voyager', 'apollo',
            'bay', 'bengal', 'modi', 'crew', 'orbit', 'earth'
        }
    
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
    
    def extract_linguistic_features(self, texts):
        """Extract linguistic features using NLTK"""
        features = []
        for text in texts:
            # Tokenize sentences and words
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            
            # Get POS tags
            pos_tags = pos_tag(words)
            
            # Count named entities
            named_entities = ne_chunk(pos_tags)
            num_entities = len([chunk for chunk in named_entities if hasattr(chunk, 'label')])
            
            # Count POS tags
            pos_counts = {}
            for _, tag in pos_tags:
                pos_counts[tag] = pos_counts.get(tag, 0) + 1
            
            # Calculate features
            feature_vector = [
                num_entities,
                len(sentences),
                len(words) / len(sentences) if len(sentences) > 0 else 0,
                len(set([tag for _, tag in pos_tags])),
                pos_counts.get('NN', 0) + pos_counts.get('NNS', 0),  # Nouns
                pos_counts.get('VB', 0) + pos_counts.get('VBD', 0) + pos_counts.get('VBG', 0) + pos_counts.get('VBN', 0) + pos_counts.get('VBP', 0) + pos_counts.get('VBZ', 0),  # Verbs
                pos_counts.get('JJ', 0) + pos_counts.get('JJR', 0) + pos_counts.get('JJS', 0),  # Adjectives
                pos_counts.get('RB', 0) + pos_counts.get('RBR', 0) + pos_counts.get('RBS', 0)  # Adverbs
            ]
            features.append(feature_vector)
        return np.array(features)
    
    def extract_sentiment_features(self, texts):
        """Extract sentiment features"""
        features = []
        for text in texts:
            blob = TextBlob(text)
            sentiment = blob.sentiment
            features.append([
                sentiment.polarity,
                sentiment.subjectivity,
                len(text.split()),
                len(re.findall(r'[A-Z][a-z]+', text))  # Count proper nouns
            ])
        return np.array(features)
    
    def extract_tfidf_features(self, texts):
        """Extract TF-IDF features"""
        return self.tfidf_vectorizer.fit_transform(texts)
    
    def extract_article_features(self, texts):
        """Extract article-specific features"""
        features = []
        for text in texts:
            # Simple article features without newspaper3k
            words = text.split()
            sentences = sent_tokenize(text)
            features.append([
                len(words),  # Total words
                len(sentences),  # Number of sentences
                len(set(words)),  # Unique words
                len([w for w in words if w[0].isupper()])  # Proper nouns
            ])
        return np.array(features)
    
    def fit_transform(self, texts):
        """Extract all features and combine them"""
        # BERT features
        bert_features = self.extract_bert_features(texts)
        
        # Linguistic features
        linguistic_features = self.extract_linguistic_features(texts)
        
        # Sentiment features
        sentiment_features = self.extract_sentiment_features(texts)
        
        # TF-IDF features
        tfidf_features = self.extract_tfidf_features(texts)
        
        # Article features
        article_features = self.extract_article_features(texts)
        
        # Combine all features
        combined_features = hstack([
            tfidf_features,
            linguistic_features,
            sentiment_features,
            article_features
        ]).toarray()
        
        # Concatenate with BERT features
        final_features = np.hstack([bert_features, combined_features])
        
        return final_features
    
    def transform(self, texts):
        """Transform new texts using fitted vectorizers"""
        # BERT features
        bert_features = self.extract_bert_features(texts)
        
        # Linguistic features
        linguistic_features = self.extract_linguistic_features(texts)
        
        # Sentiment features
        sentiment_features = self.extract_sentiment_features(texts)
        
        # TF-IDF features
        tfidf_features = self.tfidf_vectorizer.transform(texts)
        
        # Article features
        article_features = self.extract_article_features(texts)
        
        # Combine all features
        combined_features = hstack([
            tfidf_features,
            linguistic_features,
            sentiment_features,
            article_features
        ]).toarray()
        
        # Concatenate with BERT features
        final_features = np.hstack([bert_features, combined_features])
        
        return final_features 
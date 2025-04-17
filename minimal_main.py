import pandas as pd
from minimal_feature_extractor import MinimalFeatureExtractor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def load_sample_data():
    """Load sample data for training"""
    data = {
        'text': [
            # Real Space News
            'India successfully launches Gaganyaan test flight for manned space mission.',
            'NASA\'s Perseverance rover discovers evidence of ancient water on Mars.',
            'SpaceX successfully launches Falcon 9 rocket with Starlink satellites.',
            'China\'s Chang\'e 5 mission returns lunar samples to Earth.',
            'European Space Agency launches JUICE mission to study Jupiter\'s moons.',
            
            # Fake Space News
            'NASA discovers alien city on the dark side of the Moon.',
            'SpaceX CEO Elon Musk reveals plans to build city on Mars by 2025.',
            'Scientists find evidence of ancient alien technology in asteroid samples.',
            'Government cover-up: Astronauts found ancient ruins on Mars.',
            'NASA confirms existence of parallel universe after space experiment.',
            
            # Real General News
            'The government has announced new policies to combat climate change.',
            'Scientists have discovered a new species of butterfly in the Amazon rainforest.',
            'New research shows that regular exercise improves mental health.',
            'Researchers have found a new way to recycle plastic more efficiently.',
            'A breakthrough in renewable energy technology has been announced.',
            
            # Fake General News
            'The Earth is flat and NASA has been hiding this fact for decades.',
            'The government is using 5G to control people\'s minds.',
            'The world is ending tomorrow according to ancient prophecies.',
            'The government is hiding evidence of time travel technology.',
            'Aliens have landed in New York City and are taking over the government.'
        ],
        'label': [0] * 5 + [1] * 5 + [0] * 5 + [1] * 5  # 0 for real, 1 for fake
    }
    return pd.DataFrame(data)

class MinimalFakeNewsDetector:
    def __init__(self):
        self.feature_extractor = MinimalFeatureExtractor()
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='linear', probability=True, random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
        }
    
    def train(self, texts, labels):
        """Train the models"""
        # Extract features
        features = self.feature_extractor.fit_transform(texts)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        # Train each model
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{name} Accuracy: {accuracy:.2f}")
            print(classification_report(y_test, y_pred))
    
    def predict(self, text):
        """Predict if a text is fake news"""
        # Extract features
        features = self.feature_extractor.transform([text])
        
        # Get predictions from each model
        predictions = {}
        for name, model in self.models.items():
            pred = model.predict(features)[0]
            prob = model.predict_proba(features)[0][1]  # Probability of being fake
            predictions[name] = (pred, prob)
        
        # Calculate final prediction (majority vote)
        final_pred = np.mean([pred for pred, _ in predictions.values()]) > 0.5
        confidence = np.mean([prob for _, prob in predictions.values()])
        
        return {
            'prediction': 'Fake' if final_pred else 'Real',
            'confidence': confidence,
            'individual_predictions': {
                name: 'Fake' if pred == 1 else 'Real'
                for name, (pred, _) in predictions.items()
            }
        }

def main():
    try:
        # Load data
        print("Loading data...")
        df = load_sample_data()
        
        # Initialize detector
        print("Initializing detector...")
        detector = MinimalFakeNewsDetector()
        
        # Train the model
        print("Training model...")
        detector.train(df['text'], df['label'])
        
        print("\nMinimal Fake News Detection System")
        print("--------------------------------")
        print("Enter 'quit' to exit the program")
        print("Enter a news article to check if it's real or fake")
        
        while True:
            try:
                user_input = input("\nEnter news article: ").strip()
                
                if user_input.lower() == 'quit':
                    print("Exiting program...")
                    break
                    
                if not user_input:
                    print("Please enter a valid news article.")
                    continue
                
                result = detector.predict(user_input)
                print("\nAnalysis Results:")
                print(f"Final Prediction: {result['prediction']}")
                print(f"Confidence: {result['confidence']:.2%}")
                
                print("\nIndividual Model Predictions:")
                for model, pred in result['individual_predictions'].items():
                    print(f"{model}: {pred}")
                
                if result['prediction'] == 'Fake':
                    print("\nWarning: This article might be fake news!")
                    print("Please verify the information from reliable sources.")
                else:
                    print("\nThis article appears to be real news.")
                    print("However, always verify information from multiple sources.")
                    
            except Exception as e:
                print(f"Error processing input: {str(e)}")
                print("Please try again with a different input.")
                continue
                
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 
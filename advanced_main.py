import pandas as pd
import sys
from preprocess import TextPreprocessor
from advanced_feature_extractor import AdvancedFeatureExtractor
from advanced_model import AdvancedFakeNewsDetector

def load_sample_data():
    """
    Load and prepare sample data with more diverse examples
    """
    try:
        data = {
            'text': [
                # Real Space News
                'India successfully launches Gaganyaan test flight for manned space mission.',
                'NASA\'s Perseverance rover discovers evidence of ancient water on Mars.',
                'SpaceX successfully launches Falcon 9 rocket with Starlink satellites.',
                'China\'s Chang\'e 5 mission returns lunar samples to Earth.',
                'European Space Agency launches JUICE mission to study Jupiter\'s moons.',
                'NASA announces Artemis program to return humans to the Moon.',
                'ISRO successfully launches Chandrayaan-3 mission to the Moon.',
                'SpaceX Starship completes first orbital test flight.',
                'James Webb Space Telescope captures stunning images of distant galaxies.',
                'Russia and China announce plans for joint lunar research station.',
                
                # Fake Space News
                'NASA discovers alien city on the dark side of the Moon.',
                'SpaceX CEO Elon Musk reveals plans to build city on Mars by 2025.',
                'Scientists find evidence of ancient alien technology in asteroid samples.',
                'Government cover-up: Astronauts found ancient ruins on Mars.',
                'NASA confirms existence of parallel universe after space experiment.',
                'SpaceX accidentally launches rocket into alternate dimension.',
                'Scientists discover breathable atmosphere on Venus.',
                'NASA announces discovery of giant space creature near Jupiter.',
                'Moon landing was faked in Hollywood studio, new evidence reveals.',
                'Government hiding evidence of alien contact with ISS astronauts.',
                
                # Real General News
                'The government has announced new policies to combat climate change.',
                'Scientists have discovered a new species of butterfly in the Amazon rainforest.',
                'New research shows that regular exercise improves mental health.',
                'Researchers have found a new way to recycle plastic more efficiently.',
                'A breakthrough in renewable energy technology has been announced.',
                'Global leaders meet to discuss climate change solutions.',
                'New study reveals benefits of meditation for stress reduction.',
                'Scientists develop new method for early cancer detection.',
                'Renewable energy surpasses fossil fuels in electricity generation.',
                'International team discovers new species in deep ocean.',
                
                # Fake General News
                'The Earth is flat and NASA has been hiding this fact for decades.',
                'The government is using 5G to control people\'s minds.',
                'The world is ending tomorrow according to ancient prophecies.',
                'The government is hiding evidence of time travel technology.',
                'Aliens have landed in New York City and are taking over the government.',
                'Scientists discover that vaccines contain microchips for tracking.',
                'New study proves that the moon is made of cheese.',
                'Government admits to controlling weather with secret technology.',
                'Ancient civilization discovered living under Antarctica.',
                'Scientists find evidence that dinosaurs never existed.'
            ],
            'label': [0] * 10 + [1] * 10 + [0] * 10 + [1] * 10  # 0 for real, 1 for fake
        }
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error loading sample data: {str(e)}")
        sys.exit(1)

def main():
    try:
        # Load data
        print("Loading data...")
        df = load_sample_data()
        
        # Preprocess text
        print("Preprocessing text...")
        preprocessor = TextPreprocessor()
        df['processed_text'] = df['text'].apply(preprocessor.preprocess)
        
        # Initialize advanced feature extractor and model
        print("Initializing models...")
        feature_extractor = AdvancedFeatureExtractor()
        detector = AdvancedFakeNewsDetector()
        
        # Train the model
        print("Training model...")
        detector.train(df['processed_text'], df['label'], feature_extractor)
        
        print("\nAdvanced Fake News Detection System")
        print("----------------------------------")
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
                    print(f"{model}: {'Fake' if pred == 1 else 'Real'}")
                
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
        sys.exit(1)

if __name__ == "__main__":
    main() 
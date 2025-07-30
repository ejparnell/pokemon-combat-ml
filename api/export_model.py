"""
Model Export Script
Saves the trained Pokemon battle predictor model for API use
Run this after training your model in the notebooks
"""

import pandas as pd
import joblib
import json
from pathlib import Path
import sys
import os

def export_model():
    """Export the trained model and necessary data for the API"""
    
    print("🔧 Pokemon Battle Predictor Model Export")
    print("="*50)
    
    # Create necessary directories
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("📂 Created models directory")
    
    # Instructions for manual model saving
    print("\n📚 Manual Model Export Instructions:")
    print("-" * 40)
    print("Since your model is trained in Jupyter notebooks, please follow these steps:")
    print()
    print("1. In your model_training_optimized.ipynb notebook, add this cell at the end:")
    print("   ```python")
    print("   import joblib")
    print("   from pathlib import Path")
    print("   ")
    print("   # Create models directory")
    print("   Path('models').mkdir(exist_ok=True)")
    print("   ")
    print("   # Save the trained model")
    print("   joblib.dump(rf_final, 'models/pokemon_battle_predictor.joblib')")
    print("   print('✅ Model saved successfully!')")
    print("   ```")
    print()
    print("2. Run that cell to save your trained model")
    print()
    print("3. The API will automatically load the model from 'models/pokemon_battle_predictor.joblib'")
    print()
    
    # Check if files exist
    model_path = Path("models/pokemon_battle_predictor.joblib")
    pokemon_data_path = Path("data/pokemon_cleaned.csv")
    feature_config_path = Path("processed/feature_config.json")
    
    print("🔍 Checking for required files:")
    print(f"   Model file: {'✅ Found' if model_path.exists() else '❌ Not found'} - {model_path}")
    print(f"   Pokemon data: {'✅ Found' if pokemon_data_path.exists() else '❌ Not found'} - {pokemon_data_path}")
    print(f"   Feature config: {'✅ Found' if feature_config_path.exists() else '❌ Not found'} - {feature_config_path}")
    print()
    
    if model_path.exists():
        print("🎉 Great! Your model is ready for the API")
        
        # Test loading the model
        try:
            model = joblib.load(model_path)
            print(f"✅ Model loaded successfully: {type(model).__name__}")
            
            if hasattr(model, 'n_estimators'):
                print(f"   Trees: {model.n_estimators}")
            if hasattr(model, 'feature_importances_'):
                print(f"   Features: {len(model.feature_importances_)}")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    else:
        print("⚠️ Model not found. Please run the export code in your notebook first.")
    
    print()
    print("🚀 Next steps:")
    print("1. Save your model using the code above")
    print("2. Install API dependencies: pip install -r api/requirements.txt")
    print("3. Start the API: python api/main.py")
    print("4. Visit http://localhost:8000/docs for interactive API documentation")

if __name__ == "__main__":
    export_model()

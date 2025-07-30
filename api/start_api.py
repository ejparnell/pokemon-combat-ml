#!/usr/bin/env python3
"""
Pokemon Battle Predictor API Startup Script
Easy way to start the API server with proper configuration
"""

import sys
import os
import subprocess
from pathlib import Path

def check_requirements():
    """Check if all requirements are installed"""
    try:
        import fastapi
        import uvicorn
        import pandas
        import numpy
        import sklearn
        import joblib
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("📦 Please install requirements: pip install -r api/requirements.txt")
        return False

def check_files():
    """Check if required files exist"""
    files_to_check = {
        "Model file": "models/pokemon_battle_predictor.joblib",
        "Pokemon data": "data/pokemon_cleaned.csv", 
        "Feature config": "processed/feature_config.json"
    }
    
    all_exist = True
    
    for name, path in files_to_check.items():
        if Path(path).exists():
            print(f"✅ {name}: {path}")
        else:
            print(f"⚠️ {name}: {path} (not found)")
            all_exist = False
    
    return all_exist

def start_api():
    """Start the FastAPI server"""
    print("\n🚀 Starting Pokemon Battle Predictor API...")
    print("-" * 50)
    
    try:
        # Start the API server
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
            "--log-level", "info"
        ], check=True)
    except subprocess.CalledProcessError:
        print("❌ Failed to start API server")
        return False
    except KeyboardInterrupt:
        print("\n👋 API server stopped")
        return True

def main():
    """Main startup function"""
    print("🎯 Pokemon Battle Predictor API - Startup Check")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("api/main.py").exists():
        print("❌ Please run this script from the project root directory")
        print("   Expected structure: project_root/api/main.py")
        sys.exit(1)
    
    print("📂 Project structure: ✅ Correct")
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check files
    print("\n📁 Checking required files:")
    files_exist = check_files()
    
    if not files_exist:
        print("\n⚠️ Some files are missing!")
        print("📚 To fix this:")
        print("1. Run all notebooks: data-cleaning.ipynb → data-segregation.ipynb → model-training-optimized.ipynb")
        print("2. Export your model by adding this to your notebook:")
        print("   ```python")
        print("   import joblib")
        print("   joblib.dump(rf_final, 'models/pokemon_battle_predictor.joblib')")
        print("   ```")
        
        response = input("\nDo you want to start the API anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Start the API
    print("\n🎉 All checks passed! Starting API server...")
    print("📍 API will be available at:")
    print("   • Health check: http://localhost:8000/")
    print("   • Interactive docs: http://localhost:8000/docs")
    print("   • ReDoc docs: http://localhost:8000/redoc")
    print("\n⏱️ Starting server... (Press Ctrl+C to stop)")
    
    start_api()

if __name__ == "__main__":
    main()

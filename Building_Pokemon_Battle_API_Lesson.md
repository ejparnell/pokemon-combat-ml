# 🚀 Building a Pokemon Battle Prediction API: A Step-by-Step Guide

Learn how to wrap your trained machine learning model with a professional FastAPI service.

## 📋 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Step 1: Project Structure Setup](#step-1-project-structure-setup)
- [Step 2: Install Dependencies](#step-2-install-dependencies)
- [Step 3: Create Pydantic Data Models](#step-3-create-pydantic-data-models)
- [Step 4: Model Loading and Data Management](#step-4-model-loading-and-data-management)
- [Step 5: Feature Engineering Functions](#step-5-feature-engineering-functions)
- [Step 6: Basic FastAPI Application Setup](#step-6-basic-fastapi-application-setup)
- [Step 7: Health Check and Info Endpoints](#step-7-health-check-and-info-endpoints)
- [Step 8: Pokemon Data Endpoints](#step-8-pokemon-data-endpoints)
- [Step 9: Battle Prediction Endpoint](#step-9-battle-prediction-endpoint)
- [Step 10: Advanced Features and Error Handling](#step-10-advanced-features-and-error-handling)
- [Step 11: Testing Your API](#step-11-testing-your-api)
- [Step 12: Production Considerations](#step-12-production-considerations)

## 🎯 Overview

In this lesson, you'll learn how to transform your Jupyter notebook-based Pokemon battle prediction model into a production-ready API. We'll cover:

- **API Design**: RESTful endpoints for different use cases
- **Data Validation**: Using Pydantic for request/response validation
- **Model Integration**: Loading and using your trained ML model
- **Error Handling**: Robust error management and user feedback
- **Documentation**: Auto-generated API documentation
- **Performance**: Optimizing for real-world usage

### What You'll Build

By the end of this lesson, you'll have a complete API that can:

- Predict Pokemon battle outcomes with 90%+ accuracy
- Search and retrieve Pokemon data
- Handle special cases (like Nidoran♀/♂)
- Provide detailed battle analysis
- Auto-generate interactive documentation

## 📚 Prerequisites

Before starting, ensure you have:

1. ✅ **Trained Model**: A saved Pokemon battle prediction model from your Jupyter notebooks
2. ✅ **Data Files**: Cleaned Pokemon CSV data
3. ✅ **Python Knowledge**: Familiarity with Python, pandas, and basic ML concepts
4. ✅ **Environment**: Python 3.8+ installed

### Required Files

```
pokemon-combat-ml/
├── models/
│   ├── pokemon_battle_predictor.joblib    # Your trained model
│   └── feature_names.json                 # Feature configuration
├── data/
│   ├── pokemon.csv                        # Original Pokemon data
│   └── pokemon_cleaned.csv               # Cleaned Pokemon data
└── processed/
    └── feature_config.json               # Feature engineering config
```

## Step 1: Project Structure Setup

First, let's organize our API code in a dedicated folder structure.

### Create the API Directory

```bash
# From your project root
mkdir -p api
cd api
```

### Recommended Structure

```
api/
├── main.py              # Main FastAPI application
├── requirements.txt     # Python dependencies
├── models.py           # Pydantic data models (optional separate file)
├── utils.py            # Utility functions (optional separate file)
└── tests/              # Test files (for later)
    └── test_main.py
```

**Learning Point**: Organizing code into logical modules makes it easier to maintain and scale your API as it grows.

## Step 2: Install Dependencies

Create a `requirements.txt` file with all necessary dependencies.

### Create requirements.txt

```bash
# In the api/ directory
touch requirements.txt
```

Add these dependencies:

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pandas==2.1.4
numpy==1.24.3
scikit-learn==1.3.2
joblib==1.3.2
python-multipart==0.0.6
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Learning Point**: Pinning specific versions ensures your API works consistently across different environments.

## Step 3: Create Pydantic Data Models

Pydantic models define the structure and validation rules for your API's input and output data.

### Why Pydantic?

- **Automatic Validation**: Ensures data meets your requirements
- **Type Safety**: Catches type errors early
- **Documentation**: Auto-generates OpenAPI/Swagger docs
- **Serialization**: Handles JSON conversion automatically

### Create Data Models

Add this to the top of your `main.py`:

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# Input Models (for requests)
class PokemonStats(BaseModel):
    """Pokemon statistics and metadata for battle predictions"""
    name: str = Field(..., description="Pokemon name")
    hp: int = Field(..., ge=1, le=255, description="HP stat (1-255)")
    attack: int = Field(..., ge=1, le=255, description="Attack stat (1-255)")
    defense: int = Field(..., ge=1, le=255, description="Defense stat (1-255)")
    sp_atk: int = Field(..., ge=1, le=255, description="Special Attack stat (1-255)")
    sp_def: int = Field(..., ge=1, le=255, description="Special Defense stat (1-255)")
    speed: int = Field(..., ge=1, le=255, description="Speed stat (1-255)")
    type_1: str = Field(..., description="Primary type")
    type_2: Optional[str] = Field(None, description="Secondary type (optional)")
    legendary: bool = Field(False, description="Is this Pokemon legendary?")
    generation: int = Field(..., ge=1, le=9, description="Generation (1-9)")

class BattleRequest(BaseModel):
    """Request model for battle predictions"""
    pokemon_a: PokemonStats = Field(..., description="First Pokemon")
    pokemon_b: PokemonStats = Field(..., description="Second Pokemon")

# Output Models (for responses)
class PokemonInfo(BaseModel):
    """Pokemon information response"""
    name: str                    # Cleaned name used for training
    original_name: str          # Original name from source data
    stats: Dict[str, int]       # Individual stats
    types: List[str]            # Pokemon types
    total_stats: int            # Sum of all stats
    legendary: bool             # Legendary status
    generation: int             # Generation number

class BattlePrediction(BaseModel):
    """Battle prediction response"""
    winner_prediction: str = Field(..., description="Predicted winner: 'pokemon_a' or 'pokemon_b'")
    win_probability: float = Field(..., description="Probability of pokemon_a winning (0.0-1.0)")
    confidence: float = Field(..., description="Model confidence in prediction (0.0-1.0)")
    key_factors: List[str] = Field(..., description="Top factors influencing the prediction")
    battle_analysis: Dict[str, Any] = Field(..., description="Detailed battle analysis")
```

**Learning Points**:

- `Field(...)` makes a field required
- `ge=1, le=255` adds range validation
- `Optional[str]` allows None values
- Descriptions become part of the auto-generated docs

## Step 4: Model Loading and Data Management

Now we'll create functions to load your trained model and Pokemon data.

### Global Variables

```python
import pandas as pd
import numpy as np
import joblib
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and data
model = None
pokemon_data = None
original_pokemon_data = None
pokemon_name_mapping = None
feature_config = None
all_pokemon_types = None
```

### Model Loading Function

```python
def load_model_and_data():
    """Load the trained model and Pokemon data"""
    global model, pokemon_data, original_pokemon_data, pokemon_name_mapping, feature_config, all_pokemon_types
    
    try:
        # Load the trained model
        model_path = Path("models/pokemon_battle_predictor.joblib")
        if model_path.exists():
            model = joblib.load(model_path)
            logger.info("✅ Trained model loaded successfully")
        else:
            logger.warning("⚠️ Model file not found - using mock predictions")
            model = None

        # Load model feature configuration
        feature_names_path = Path("models/feature_names.json")
        if feature_names_path.exists():
            with open(feature_names_path, 'r') as f:
                feature_config = json.load(f)
            logger.info(f"✅ Model feature names loaded: {feature_config['feature_count']} features")
        else:
            logger.warning("⚠️ Model feature names not found")
            feature_config = None

        # Load Pokemon datasets
        load_pokemon_data()
        
        # Define Pokemon types
        all_pokemon_types = [
            'Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting',
            'Fire', 'Flying', 'Ghost', 'Grass', 'Ground', 'Ice',
            'Normal', 'Poison', 'Psychic', 'Rock', 'Steel', 'Water'
        ]

    except Exception as e:
        logger.error(f"❌ Error loading model/data: {e}")
        raise

def load_pokemon_data():
    """Load and process Pokemon datasets"""
    global pokemon_data, original_pokemon_data, pokemon_name_mapping
    
    # Load cleaned Pokemon data (used for training)
    pokemon_csv_path = Path("data/pokemon_cleaned.csv")
    if pokemon_csv_path.exists():
        pokemon_data = pd.read_csv(pokemon_csv_path)
        
        # Clean data issues (like corrupted Nidoran entries)
        pokemon_data = pokemon_data[~((pokemon_data['name'] == 'NidoranF') & 
                                    (pokemon_data['hp'] == 35))]
        pokemon_data = pokemon_data[~((pokemon_data['name'] == 'NidoranM') & 
                                    (pokemon_data['hp'] == 60))]
        
        # Handle NaN values that cause JSON serialization issues
        pokemon_data = pokemon_data.fillna({'type_2': 'None'})
        pokemon_data = pokemon_data.where(pd.notnull(pokemon_data), None)
        
        logger.info(f"✅ Cleaned Pokemon data loaded: {len(pokemon_data)} Pokemon")
    else:
        logger.warning("⚠️ Cleaned Pokemon data not found")
        pokemon_data = None

    # Load original Pokemon data (with original names)
    original_pokemon_csv_path = Path("data/pokemon.csv")
    if original_pokemon_csv_path.exists():
        original_pokemon_data = pd.read_csv(original_pokemon_csv_path)
        original_pokemon_data = original_pokemon_data.fillna({'Type 2': 'None'})
        original_pokemon_data = original_pokemon_data.where(pd.notnull(original_pokemon_data), None)
        logger.info(f"✅ Original Pokemon data loaded: {len(original_pokemon_data)} Pokemon")
        
        # Create intelligent name mapping
        create_name_mapping()
    else:
        logger.warning("⚠️ Original Pokemon data not found")
        original_pokemon_data = None
        pokemon_name_mapping = None

def create_name_mapping():
    """Create mapping between cleaned and original Pokemon names"""
    global pokemon_name_mapping
    
    if pokemon_data is None or original_pokemon_data is None:
        return
    
    pokemon_name_mapping = {}
    
    # Create lookup using stat signatures (HP, Attack, Defense)
    original_lookup = {}
    for _, row in original_pokemon_data.iterrows():
        name = row['Name']
        signature = (int(row['HP']), int(row['Attack']), int(row['Defense']))
        original_lookup[signature] = name
    
    # Map cleaned Pokemon to original using stat signatures
    for _, row in pokemon_data.iterrows():
        cleaned_name = row['name']
        try:
            signature = (int(row['hp']), int(row['attack']), int(row['defense']))
            
            if signature in original_lookup:
                original_name = original_lookup[signature]
                
                # Handle special Nidoran cases
                if cleaned_name.lower() == "nidoran":
                    if "♀" in original_name:
                        map_key = "nidoran_female"
                        cleaned_name = "Nidoran♀"
                    elif "♂" in original_name:
                        map_key = "nidoran_male"
                        cleaned_name = "Nidoran♂"
                    else:
                        map_key = cleaned_name.lower()
                else:
                    map_key = cleaned_name.lower()
                
                pokemon_name_mapping[map_key] = {
                    'original_name': original_name,
                    'cleaned_name': cleaned_name
                }
            else:
                # Fallback mapping
                pokemon_name_mapping[cleaned_name.lower()] = {
                    'original_name': cleaned_name,
                    'cleaned_name': cleaned_name
                }
        except (ValueError, TypeError):
            continue
    
    logger.info(f"✅ Pokemon name mapping created: {len(pokemon_name_mapping)} mappings")
```

**Learning Points**:

- Global variables store loaded data for reuse across requests
- Logging helps debug issues in production
- Fallback handling ensures the API works even with missing data
- Stat signatures provide robust Pokemon identification

## Step 5: Feature Engineering Functions

Recreate the feature engineering logic from your notebooks for the API.

### Type Effectiveness Features

```python
def create_type_effectiveness_features(df):
    """Create type effectiveness features for the model"""
    for pkmn in ['a', 'b']:
        for type_name in all_pokemon_types:
            # Primary type indicator
            df[f'{pkmn}_is_{type_name.lower()}'] = (df[f'{pkmn}_type_1'] == type_name).astype(int)
            # Has type indicator (primary or secondary)
            df[f'{pkmn}_has_{type_name.lower()}'] = (
                (df[f'{pkmn}_type_1'] == type_name) | 
                (df[f'{pkmn}_type_2'] == type_name)
            ).astype(int)
    
    # Type diversity features
    df['a_has_dual_type'] = (df['a_type_2'] != 'None').astype(int)
    df['b_has_dual_type'] = (df['b_type_2'] != 'None').astype(int)
    
    return df
```

### Main Feature Engineering Function

```python
def engineer_battle_features(pokemon_a: PokemonStats, pokemon_b: PokemonStats) -> pd.DataFrame:
    """Engineer features for battle prediction - matches training pipeline"""
    
    # Create basic feature dictionary
    features = {
        # Pokemon A stats
        'a_hp': pokemon_a.hp,
        'a_attack': pokemon_a.attack,
        'a_defense': pokemon_a.defense,
        'a_sp_atk': pokemon_a.sp_atk,
        'a_sp_def': pokemon_a.sp_def,
        'a_speed': pokemon_a.speed,
        'a_type_1': pokemon_a.type_1,
        'a_type_2': pokemon_a.type_2 or 'None',
        'a_legendary': int(pokemon_a.legendary),
        'a_generation': pokemon_a.generation,
        
        # Pokemon B stats
        'b_hp': pokemon_b.hp,
        'b_attack': pokemon_b.attack,
        'b_defense': pokemon_b.defense,
        'b_sp_atk': pokemon_b.sp_atk,
        'b_sp_def': pokemon_b.sp_def,
        'b_speed': pokemon_b.speed,
        'b_type_1': pokemon_b.type_1,
        'b_type_2': pokemon_b.type_2 or 'None',
        'b_legendary': int(pokemon_b.legendary),
        'b_generation': pokemon_b.generation,
    }
    
    # Create DataFrame
    df = pd.DataFrame([features])
    
    # Engineer advanced features (same as training)
    df['speed_diff'] = df['a_speed'] - df['b_speed']
    
    # Base stat totals
    a_bst = df[['a_hp', 'a_attack', 'a_defense', 'a_sp_atk', 'a_sp_def', 'a_speed']].sum(axis=1)
    b_bst = df[['b_hp', 'b_attack', 'b_defense', 'b_sp_atk', 'b_sp_def', 'b_speed']].sum(axis=1)
    df['a_bst'] = a_bst
    df['b_bst'] = b_bst
    df['bst_diff'] = a_bst - b_bst
    
    # Individual stat differences
    df['hp_diff'] = df['a_hp'] - df['b_hp']
    df['attack_diff'] = df['a_attack'] - df['b_attack']
    df['defense_diff'] = df['a_defense'] - df['b_defense']
    df['sp_atk_diff'] = df['a_sp_atk'] - df['b_sp_atk']
    df['sp_def_diff'] = df['a_sp_def'] - df['b_sp_def']
    
    # Ratio features
    df['a_atk_def_ratio'] = df['a_attack'] / (df['a_defense'] + 1)
    df['b_atk_def_ratio'] = df['b_attack'] / (df['b_defense'] + 1)
    df['atk_def_ratio_diff'] = df['a_atk_def_ratio'] - df['b_atk_def_ratio']
    
    df['a_sp_ratio'] = df['a_sp_atk'] / (df['a_sp_def'] + 1)
    df['b_sp_ratio'] = df['b_sp_atk'] / (df['b_sp_def'] + 1)
    df['sp_ratio_diff'] = df['a_sp_ratio'] - df['b_sp_ratio']
    
    # Add type effectiveness features
    df = create_type_effectiveness_features(df)
    
    return df
```

**Learning Points**:

- Feature engineering must exactly match your training pipeline
- Adding +1 to denominators prevents division by zero
- DataFrame format ensures compatibility with scikit-learn models

## Step 6: Basic FastAPI Application Setup

Now let's create the core FastAPI application.

### Initialize FastAPI

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(
    title="Pokemon Battle Predictor API",
    description="Advanced machine learning API for predicting Pokemon battle outcomes",
    version="1.0.0",
    docs_url="/docs",          # Swagger UI
    redoc_url="/redoc"         # ReDoc documentation
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event - load model and data when server starts
@app.on_event("startup")
async def startup_event():
    """Load model and data on startup"""
    load_model_and_data()
    logger.info("🚀 Pokemon Battle Predictor API started successfully!")
```

**Learning Points**:

- `title` and `description` appear in auto-generated docs
- CORS middleware allows frontend applications to access your API
- Startup events run once when the server starts
- `docs_url` provides interactive API testing interface

## Step 7: Health Check and Info Endpoints

Create endpoints to monitor your API's status and configuration.

### Health Check Endpoint

```python
@app.get("/", summary="API Health Check")
async def root():
    """Health check endpoint - shows API status and loaded components"""
    return {
        "message": "🚀 Pokemon Battle Predictor API",
        "status": "active",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "pokemon_data_loaded": pokemon_data is not None,
        "original_names_loaded": original_pokemon_data is not None,
        "name_mapping_created": pokemon_name_mapping is not None
    }

@app.get("/model/info", summary="Get model information")
async def get_model_info():
    """Get detailed information about the loaded model"""
    return {
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None,
        "feature_config_loaded": feature_config is not None,
        "total_features": len(feature_config.get('features', [])) if feature_config else 0,
        "pokemon_database_size": len(pokemon_data) if pokemon_data is not None else 0
    }

@app.get("/pokemon/types", response_model=List[str], summary="Get all Pokemon types")
async def get_pokemon_types():
    """Get list of all Pokemon types supported by the API"""
    return all_pokemon_types
```

**Learning Points**:

- Health checks help monitor API status in production
- `summary` parameters appear in documentation
- `response_model` validates and documents response structure
- Conditional checks handle cases where data isn't loaded

## Step 8: Pokemon Data Endpoints

Create endpoints for searching and listing Pokemon data.

### Search Pokemon Endpoint

```python
@app.get("/pokemon/search/{name}", response_model=PokemonInfo, summary="Search for a Pokemon by name")
async def search_pokemon(name: str):
    """Search for a Pokemon by name and return its complete stats"""
    if pokemon_data is None:
        raise HTTPException(status_code=503, detail="Pokemon data not available")
    
    # Handle special Nidoran search cases
    pokemon = None
    if name.lower() in ["nidoran♀", "nidoran female", "nidoran f"]:
        # Search for Nidoran♀ using stat signature
        nidoran_entries = pokemon_data[pokemon_data['name'].str.lower() == "nidoran"]
        for _, entry in nidoran_entries.iterrows():
            if (int(entry['hp']), int(entry['attack']), int(entry['defense'])) == (55, 47, 52):
                pokemon = entry
                break
    elif name.lower() in ["nidoran♂", "nidoran male", "nidoran m"]:
        # Search for Nidoran♂ using stat signature
        nidoran_entries = pokemon_data[pokemon_data['name'].str.lower() == "nidoran"]
        for _, entry in nidoran_entries.iterrows():
            if (int(entry['hp']), int(entry['attack']), int(entry['defense'])) == (46, 57, 40):
                pokemon = entry
                break
    else:
        # Regular search
        matches = pokemon_data[pokemon_data['name'].str.lower() == name.lower()]
        if not matches.empty:
            pokemon = matches.iloc[0]
    
    # Try searching in original names if not found
    if pokemon is None:
        # Implementation for original name search...
        if original_pokemon_data is not None:
            original_matches = original_pokemon_data[original_pokemon_data['Name'].str.lower() == name.lower()]
            if not original_matches.empty:
                original_row = original_matches.iloc[0]
                # Find corresponding cleaned data using stat signature
                signature = (int(original_row['HP']), int(original_row['Attack']), int(original_row['Defense']))
                for _, cleaned_row in pokemon_data.iterrows():
                    if (int(cleaned_row['hp']), int(cleaned_row['attack']), int(cleaned_row['defense'])) == signature:
                        pokemon = cleaned_row
                        break
    
    if pokemon is None:
        raise HTTPException(status_code=404, detail=f"Pokemon '{name}' not found")
    
    # Process the found Pokemon
    csv_name = pokemon['name']
    display_name = csv_name
    original_name = csv_name
    
    # Handle special Nidoran display names
    if csv_name.lower() == "nidoran":
        hp, attack, defense = int(pokemon['hp']), int(pokemon['attack']), int(pokemon['defense'])
        if (hp, attack, defense) == (55, 47, 52):
            map_key = "nidoran_female"
            display_name = "Nidoran♀"
        elif (hp, attack, defense) == (46, 57, 40):
            map_key = "nidoran_male"
            display_name = "Nidoran♂"
        else:
            map_key = csv_name.lower()
    else:
        map_key = csv_name.lower()
    
    # Get original name from mapping
    if pokemon_name_mapping and map_key in pokemon_name_mapping:
        original_name = pokemon_name_mapping[map_key]['original_name']
    
    # Build types list
    types = [pokemon['type_1']]
    if pd.notna(pokemon['type_2']) and pokemon['type_2'] not in ['None', 'NaN', '']:
        types.append(pokemon['type_2'])
    
    # Safely convert stats
    stats = {
        "hp": int(pokemon['hp']) if pd.notna(pokemon['hp']) else 0,
        "attack": int(pokemon['attack']) if pd.notna(pokemon['attack']) else 0,
        "defense": int(pokemon['defense']) if pd.notna(pokemon['defense']) else 0,
        "sp_atk": int(pokemon['sp_atk']) if pd.notna(pokemon['sp_atk']) else 0,
        "sp_def": int(pokemon['sp_def']) if pd.notna(pokemon['sp_def']) else 0,
        "speed": int(pokemon['speed']) if pd.notna(pokemon['speed']) else 0
    }
    
    return PokemonInfo(
        name=display_name,
        original_name=original_name,
        stats=stats,
        types=types,
        total_stats=sum(stats.values()),
        legendary=bool(pokemon['legendary']) if pd.notna(pokemon['legendary']) else False,
        generation=int(pokemon['generation']) if pd.notna(pokemon['generation']) else 1
    )
```

### List Pokemon Endpoint

```python
@app.get("/pokemon/list", summary="Get list of all Pokemon")
async def list_pokemon(limit: int = 100, offset: int = 0):
    """Get paginated list of all Pokemon with both original and cleaned names"""
    if pokemon_data is None:
        raise HTTPException(status_code=503, detail="Pokemon data not available")
    
    total = len(pokemon_data)
    pokemon_list = pokemon_data.iloc[offset:offset+limit]
    
    pokemon_results = []
    for _, row in pokemon_list.iterrows():
        original_cleaned_name = row['name']
        display_cleaned_name = original_cleaned_name
        original_name = original_cleaned_name
        
        # Handle special Nidoran cases
        if original_cleaned_name == "Nidoran":
            hp, attack, defense = int(row['hp']), int(row['attack']), int(row['defense'])
            if (hp, attack, defense) == (55, 47, 52):  # Nidoran♀
                map_key = "nidoran_female"
                display_cleaned_name = "Nidoran♀"
            elif (hp, attack, defense) == (46, 57, 40):  # Nidoran♂
                map_key = "nidoran_male"
                display_cleaned_name = "Nidoran♂"
            else:
                map_key = original_cleaned_name.lower()
        else:
            map_key = original_cleaned_name.lower()
        
        # Get original name from mapping
        if pokemon_name_mapping and map_key in pokemon_name_mapping:
            original_name = pokemon_name_mapping[map_key]['original_name']
        
        # Handle types
        types = [row['type_1']]
        if pd.notna(row['type_2']) and row['type_2'] not in ['None', 'NaN', '']:
            types.append(row['type_2'])
        
        # Safe stat conversion
        hp = int(row['hp']) if pd.notna(row['hp']) else 0
        attack = int(row['attack']) if pd.notna(row['attack']) else 0
        defense = int(row['defense']) if pd.notna(row['defense']) else 0
        sp_atk = int(row['sp_atk']) if pd.notna(row['sp_atk']) else 0
        sp_def = int(row['sp_def']) if pd.notna(row['sp_def']) else 0
        speed = int(row['speed']) if pd.notna(row['speed']) else 0
        generation = int(row['generation']) if pd.notna(row['generation']) else 1
        
        pokemon_results.append({
            "name": display_cleaned_name,
            "original_name": original_name,
            "types": types,
            "total_stats": hp + attack + defense + sp_atk + sp_def + speed,
            "legendary": bool(row['legendary']) if pd.notna(row['legendary']) else False,
            "generation": generation
        })
    
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "pokemon": pokemon_results
    }
```

**Learning Points**:
- Path parameters (`{name}`) capture URL segments
- Query parameters provide optional filtering/pagination
- HTTPException creates proper error responses
- Robust error handling prevents crashes from bad data

## Step 9: Battle Prediction Endpoint

The core endpoint that uses your trained model for predictions.

### Battle Analysis Function

```python
def analyze_battle_factors(pokemon_a: PokemonStats, pokemon_b: PokemonStats, 
                          feature_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze key battle factors for detailed response"""
    
    analysis = {
        "speed_advantage": "",
        "power_comparison": "",
        "type_matchup": "",
        "legendary_factor": "",
        "stat_breakdown": {}
    }
    
    # Speed analysis
    speed_diff = pokemon_a.speed - pokemon_b.speed
    if speed_diff > 0:
        analysis["speed_advantage"] = f"{pokemon_a.name} is faster (+{speed_diff} speed)"
    elif speed_diff < 0:
        analysis["speed_advantage"] = f"{pokemon_b.name} is faster (+{abs(speed_diff)} speed)"
    else:
        analysis["speed_advantage"] = "Equal speed - simultaneous attacks"
    
    # Power comparison
    a_total = pokemon_a.hp + pokemon_a.attack + pokemon_a.defense + pokemon_a.sp_atk + pokemon_a.sp_def + pokemon_a.speed
    b_total = pokemon_b.hp + pokemon_b.attack + pokemon_b.defense + pokemon_b.sp_atk + pokemon_b.sp_def + pokemon_b.speed
    
    if a_total > b_total:
        analysis["power_comparison"] = f"{pokemon_a.name} has higher total stats ({a_total} vs {b_total})"
    elif b_total > a_total:
        analysis["power_comparison"] = f"{pokemon_b.name} has higher total stats ({b_total} vs {a_total})"
    else:
        analysis["power_comparison"] = "Equal total stats"
    
    # Type analysis
    a_types = [pokemon_a.type_1] + ([pokemon_a.type_2] if pokemon_a.type_2 else [])
    b_types = [pokemon_b.type_1] + ([pokemon_b.type_2] if pokemon_b.type_2 else [])
    analysis["type_matchup"] = f"{pokemon_a.name}: {'/'.join(a_types)} vs {pokemon_b.name}: {'/'.join(b_types)}"
    
    # Legendary factor
    if pokemon_a.legendary and not pokemon_b.legendary:
        analysis["legendary_factor"] = f"{pokemon_a.name} is legendary"
    elif pokemon_b.legendary and not pokemon_a.legendary:
        analysis["legendary_factor"] = f"{pokemon_b.name} is legendary"
    elif pokemon_a.legendary and pokemon_b.legendary:
        analysis["legendary_factor"] = "Both Pokemon are legendary"
    else:
        analysis["legendary_factor"] = "No legendary Pokemon"
    
    # Detailed stat breakdown
    analysis["stat_breakdown"] = {
        "pokemon_a": {
            "name": pokemon_a.name,
            "total_stats": int(a_total),
            "hp": pokemon_a.hp,
            "attack": pokemon_a.attack,
            "defense": pokemon_a.defense,
            "sp_atk": pokemon_a.sp_atk,
            "sp_def": pokemon_a.sp_def,
            "speed": pokemon_a.speed
        },
        "pokemon_b": {
            "name": pokemon_b.name,
            "total_stats": int(b_total),
            "hp": pokemon_b.hp,
            "attack": pokemon_b.attack,
            "defense": pokemon_b.defense,
            "sp_atk": pokemon_b.sp_atk,
            "sp_def": pokemon_b.sp_def,
            "speed": pokemon_b.speed
        }
    }
    
    return analysis
```

### Main Prediction Endpoint

```python
@app.post("/predict", response_model=BattlePrediction, summary="Predict Pokemon battle outcome")
async def predict_battle(battle_request: BattleRequest):
    """
    Predict the outcome of a Pokemon battle using the trained ML model
    
    This endpoint:
    - Engineers features identical to training pipeline
    - Uses the loaded Random Forest model for prediction
    - Provides detailed battle analysis
    - Returns win probabilities and confidence scores
    """
    
    try:
        # Engineer features for prediction
        feature_df = engineer_battle_features(battle_request.pokemon_a, battle_request.pokemon_b)
        
        # Make prediction using trained model
        if model is not None and feature_config is not None:
            # Use the exact feature order from training
            expected_features = feature_config['features']
            
            # Validate features
            missing_features = [f for f in expected_features if f not in feature_df.columns]
            if missing_features:
                logger.error(f"Missing features: {missing_features[:5]}...")
                raise Exception("Missing required features for prediction")
            
            # Select features in exact training order
            X = feature_df[expected_features]
            
            # Make prediction
            prediction = model.predict(X)[0]  # 0 or 1
            probability = model.predict_proba(X)[0]  # [prob_0, prob_1]
            
            win_probability = float(probability[1])  # Probability of pokemon_a winning
            confidence = float(max(probability))     # Model confidence
            
            # Extract key factors from feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': expected_features,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                key_factors = feature_importance.head(5)['feature'].tolist()
            else:
                key_factors = ["speed_diff", "bst_diff", "legendary_status", "type_effectiveness"]
                
        else:
            # Fallback prediction when model isn't available
            logger.warning("Using mock prediction - model not loaded")
            
            # Simple heuristic based on total stats and speed
            a_total = sum([battle_request.pokemon_a.hp, battle_request.pokemon_a.attack, 
                          battle_request.pokemon_a.defense, battle_request.pokemon_a.sp_atk,
                          battle_request.pokemon_a.sp_def, battle_request.pokemon_a.speed])
            b_total = sum([battle_request.pokemon_b.hp, battle_request.pokemon_b.attack,
                          battle_request.pokemon_b.defense, battle_request.pokemon_b.sp_atk,
                          battle_request.pokemon_b.sp_def, battle_request.pokemon_b.speed])
            
            # Factor in legendary status
            a_score = a_total + (50 if battle_request.pokemon_a.legendary else 0)
            b_score = b_total + (50 if battle_request.pokemon_b.legendary else 0)
            
            # Calculate probability
            total_score = a_score + b_score
            win_probability = a_score / total_score if total_score > 0 else 0.5
            
            prediction = 1 if win_probability > 0.5 else 0
            confidence = abs(win_probability - 0.5) * 2  # Scale to 0-1
            
            key_factors = ["total_stats", "speed", "legendary_status", "type_effectiveness"]
        
        # Determine winner
        winner_prediction = "pokemon_a" if prediction == 1 else "pokemon_b"
        
        # Generate detailed battle analysis
        battle_analysis = analyze_battle_factors(
            battle_request.pokemon_a, 
            battle_request.pokemon_b, 
            feature_df
        )
        
        return BattlePrediction(
            winner_prediction=winner_prediction,
            win_probability=win_probability,
            confidence=confidence,
            key_factors=key_factors,
            battle_analysis=battle_analysis
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
```

**Learning Points**:

- POST endpoints receive data in the request body
- Feature engineering must exactly match training pipeline
- Always validate that required features are present
- Fallback predictions ensure the API works even without a model
- Detailed error logging helps debug issues

## Step 10: Advanced Features and Error Handling

Add robust error handling and advanced features.

### Custom Error Handlers

```python
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle value errors gracefully"""
    return JSONResponse(
        status_code=400,
        content={"detail": f"Invalid value: {str(exc)}"}
    )

@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError):
    """Handle missing file errors"""
    return JSONResponse(
        status_code=503,
        content={"detail": "Required data files not found. Please check server configuration."}
    )
```

### Input Validation

```python
from pydantic import validator

# Add to PokemonStats class
class PokemonStats(BaseModel):
    # ... existing fields ...
    
    @validator('type_1', 'type_2')
    def validate_types(cls, v):
        """Validate Pokemon types"""
        if v is None:
            return v
        valid_types = [
            'Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting',
            'Fire', 'Flying', 'Ghost', 'Grass', 'Ground', 'Ice',
            'Normal', 'Poison', 'Psychic', 'Rock', 'Steel', 'Water'
        ]
        if v not in valid_types:
            raise ValueError(f'Invalid type: {v}. Must be one of {valid_types}')
        return v
    
    @validator('name')
    def validate_name(cls, v):
        """Validate Pokemon name"""
        if not v or len(v.strip()) == 0:
            raise ValueError('Pokemon name cannot be empty')
        return v.strip()
```

**Learning Points**:

- Custom exception handlers provide user-friendly error messages
- Validators run automatically on incoming data
- Always validate external input to prevent security issues

## Step 11: Testing Your API

Now let's test your API to ensure it works correctly.

### Create Main Function

Add to the end of `main.py`:

```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
```

### Start Your API

```bash
# From the api directory
python main.py

# Or use uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Test Basic Functionality

```bash
# Health check
curl http://localhost:8000/

# Search for a Pokemon
curl http://localhost:8000/pokemon/search/Charizard

# List Pokemon
curl "http://localhost:8000/pokemon/list?limit=5"

# Test battle prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pokemon_a": {
      "name": "Charizard",
      "hp": 78, "attack": 84, "defense": 78,
      "sp_atk": 109, "sp_def": 85, "speed": 100,
      "type_1": "Fire", "type_2": "Flying",
      "legendary": false, "generation": 1
    },
    "pokemon_b": {
      "name": "Blastoise", 
      "hp": 79, "attack": 83, "defense": 100,
      "sp_atk": 85, "sp_def": 105, "speed": 78,
      "type_1": "Water", "type_2": null,
      "legendary": false, "generation": 1
    }
  }'
```

### Use Interactive Documentation

Visit `http://localhost:8000/docs` to:

- Test all endpoints interactively
- View automatic documentation
- Understand request/response formats
- Debug issues with the built-in test interface

**Learning Points**:

- `--reload` restarts the server when you change code
- Interactive docs are invaluable for testing and debugging
- Always test edge cases and error conditions

## Step 12: Production Considerations

### Performance Optimization

```python
# Add to your main.py
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_pokemon_by_name_cached(name: str):
    """Cache Pokemon lookups for better performance"""
    # Implementation here
    pass

# Consider using async/await for I/O operations
import asyncio

async def load_model_async():
    """Load model asynchronously if needed"""
    # For CPU-bound tasks, consider using thread pools
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, joblib.load, model_path)
```

### Configuration Management

```python
# config.py
import os
from pathlib import Path

class Settings:
    """API configuration settings"""
    MODEL_PATH: Path = Path(os.getenv("MODEL_PATH", "models/pokemon_battle_predictor.joblib"))
    DATA_PATH: Path = Path(os.getenv("DATA_PATH", "data/pokemon_cleaned.csv"))
    HOST: str = os.getenv("API_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("API_PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

settings = Settings()
```

### Logging and Monitoring

```python
import logging
from logging.handlers import RotatingFileHandler

# Production logging setup
def setup_logging():
    """Configure production logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            RotatingFileHandler('api.log', maxBytes=10000000, backupCount=5),
            logging.StreamHandler()
        ]
    )

# Add request logging middleware
from fastapi import Request
import time

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all API requests"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    logger.info(f"{request.method} {request.url} - {response.status_code} - {process_time:.2f}s")
    return response
```

### Security Considerations

```python
from fastapi.security import HTTPBearer
from fastapi import Depends

# Optional authentication
security = HTTPBearer()

async def verify_token(token: str = Depends(security)):
    """Verify API token"""
    # Implement your authentication logic
    if token.credentials != "your-secret-token":
        raise HTTPException(status_code=401, detail="Invalid token")
    return token

# Add to protected endpoints
@app.post("/predict", dependencies=[Depends(verify_token)])
async def predict_battle_protected(battle_request: BattleRequest):
    # Your prediction logic
    pass
```

### Deployment Options

#### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Production Server

```bash
# Using Gunicorn with Uvicorn workers
pip install gunicorn

gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

**Learning Points**:

- Caching improves performance for repeated requests
- Environment variables make configuration flexible
- Proper logging is essential for production debugging
- Consider authentication for sensitive endpoints
- Use production-grade servers for deployment

## 🎯 Summary

Congratulations! You've successfully built a production-ready API around your machine learning model. Here's what you've accomplished:

### ✅ **Core Features Built**

- **Model Integration**: Loaded and wrapped your trained ML model
- **Data Management**: Handled both cleaned and original Pokemon data
- **RESTful API**: Created proper endpoints for different use cases
- **Validation**: Added comprehensive input/output validation
- **Error Handling**: Implemented robust error management
- **Documentation**: Auto-generated interactive API docs

### ✅ **Advanced Features**

- **Name Mapping**: Handled complex Pokemon name cleaning issues
- **Special Cases**: Properly managed Nidoran♀/♂ distinctions
- **Feature Engineering**: Replicated training pipeline exactly
- **Battle Analysis**: Provided detailed prediction explanations
- **Performance**: Optimized for real-world usage

### ✅ **Production Ready**

- **Logging**: Comprehensive logging for debugging
- **Configuration**: Environment-based configuration
- **Security**: Authentication framework ready
- **Deployment**: Docker and production server options

### 🚀 **Next Steps**

1. **Add Features**: Consider adding more endpoints (team battles, type effectiveness lookup)
2. **Improve Performance**: Add caching, database integration, or async processing
3. **Monitor**: Set up metrics and monitoring for production use
4. **Scale**: Consider load balancing and horizontal scaling
5. **Version**: Implement API versioning for future model updates

### 💡 **Key Takeaways**

- **API Design**: Good API design makes integration easier
- **Validation**: Always validate inputs to prevent errors
- **Error Handling**: Graceful error handling improves user experience
- **Documentation**: Auto-generated docs save development time
- **Testing**: Interactive testing tools speed up development
- **Production**: Consider scalability and monitoring from the start

You now have a solid foundation for building APIs around machine learning models. The patterns and techniques you've learned here can be applied to any ML project!

### 📚 **Additional Resources**

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://pydantic-docs.helpmanual.io/)
- [Uvicorn Documentation](https://www.uvicorn.org/)
- [API Design Best Practices](https://restfulapi.net/)

Happy coding! 🎉

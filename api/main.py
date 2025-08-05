# 🚀 Pokemon Battle Prediction API
# FastAPI-based API for predicting Pokemon battle outcomes using our trained ML model

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import joblib
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Pokemon Battle Predictor API",
    description="Advanced machine learning API for predicting Pokemon battle outcomes",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class PokemonStats(BaseModel):
    """Pokemon statistics and metadata"""
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
    """Request for battle prediction"""
    pokemon_a: PokemonStats = Field(..., description="First Pokemon")
    pokemon_b: PokemonStats = Field(..., description="Second Pokemon")

class BattlePrediction(BaseModel):
    """Battle prediction response"""
    winner_prediction: str = Field(..., description="Predicted winner: 'pokemon_a' or 'pokemon_b'")
    win_probability: float = Field(..., description="Probability of pokemon_a winning (0.0-1.0)")
    confidence: float = Field(..., description="Model confidence in prediction (0.0-1.0)")
    key_factors: List[str] = Field(..., description="Top factors influencing the prediction")
    battle_analysis: Dict[str, Any] = Field(..., description="Detailed battle analysis")

class PokemonInfo(BaseModel):
    """Pokemon information response"""
    name: str  # The cleaned name used for training
    original_name: str  # The original name from pokemon.csv
    stats: Dict[str, int]
    types: List[str]
    total_stats: int
    legendary: bool
    generation: int

class PokemonListItem(BaseModel):
    """Pokemon list item with basic info"""
    name: str
    original_name: str
    types: List[str]
    total_stats: int
    legendary: bool
    generation: int

# Global variables for model and data
model = None
pokemon_data = None
original_pokemon_data = None
pokemon_name_mapping = None
feature_config = None
all_pokemon_types = None

def load_model_and_data():
    """Load the trained model and Pokemon data"""
    global model, pokemon_data, original_pokemon_data, pokemon_name_mapping, feature_config, all_pokemon_types
    
    try:
        # Load the trained model (you'll need to save your model first)
        model_path = Path("models/pokemon_battle_predictor.joblib")
        if model_path.exists():
            model = joblib.load(model_path)
            logger.info("✅ Trained model loaded successfully")
        else:
            logger.warning("⚠️ Model file not found - using mock predictions")
            model = None

        # Load the exact feature names the model was trained with
        feature_names_path = Path("models/feature_names.json")
        if feature_names_path.exists():
            with open(feature_names_path, 'r') as f:
                feature_config = json.load(f)
            logger.info(f"✅ Model feature names loaded: {feature_config['feature_count']} features")
        else:
            logger.warning("⚠️ Model feature names not found")
            feature_config = None

        # Load cleaned Pokemon data (used for training)
        pokemon_csv_path = Path("data/pokemon_cleaned.csv")
        if pokemon_csv_path.exists():
            pokemon_data = pd.read_csv(pokemon_csv_path)
            
            # Filter out corrupted NidoranF and NidoranM entries that have wrong stats
            # NidoranF has Ekans stats (35,60,44) and NidoranM has Raichu stats (60,90,55)
            pokemon_data = pokemon_data[~((pokemon_data['name'] == 'NidoranF') & 
                                        (pokemon_data['hp'] == 35) & 
                                        (pokemon_data['attack'] == 60) & 
                                        (pokemon_data['defense'] == 44))]
            pokemon_data = pokemon_data[~((pokemon_data['name'] == 'NidoranM') & 
                                        (pokemon_data['hp'] == 60) & 
                                        (pokemon_data['attack'] == 90) & 
                                        (pokemon_data['defense'] == 55))]
            
            # Clean NaN values that cause JSON serialization issues
            pokemon_data = pokemon_data.fillna({'type_2': 'None'})
            # Convert any remaining NaN to None for other columns
            pokemon_data = pokemon_data.where(pd.notnull(pokemon_data), None)
            logger.info(f"✅ Cleaned Pokemon data loaded: {len(pokemon_data)} Pokemon")
        else:
            logger.warning("⚠️ Cleaned Pokemon data not found")
            pokemon_data = None

        # Load original Pokemon data (with original names)
        original_pokemon_csv_path = Path("data/pokemon.csv")
        if original_pokemon_csv_path.exists():
            original_pokemon_data = pd.read_csv(original_pokemon_csv_path)
            # Clean NaN values in original data too
            original_pokemon_data = original_pokemon_data.fillna({'Type 2': 'None'})
            original_pokemon_data = original_pokemon_data.where(pd.notnull(original_pokemon_data), None)
            logger.info(f"✅ Original Pokemon data loaded: {len(original_pokemon_data)} Pokemon")
            
            # Create name mapping between original and cleaned names
            if pokemon_data is not None:
                pokemon_name_mapping = {}
                
                # Create a more intelligent mapping that handles misaligned data
                # First, create a lookup of original Pokemon by their key characteristics
                original_lookup = {}
                for _, row in original_pokemon_data.iterrows():
                    name = row['Name']
                    # Use HP, Attack, Defense as a unique signature for each Pokemon
                    signature = (int(row['HP']), int(row['Attack']), int(row['Defense']))
                    original_lookup[signature] = name
                
                # Now map cleaned Pokemon to original Pokemon using the same signature
                for _, row in pokemon_data.iterrows():
                    cleaned_name = row['name']
                    try:
                        # Create signature for this Pokemon
                        signature = (int(row['hp']), int(row['attack']), int(row['defense']))
                        
                        # Find matching original Pokemon
                        if signature in original_lookup:
                            original_name = original_lookup[signature]
                            
                            # Handle duplicate "Nidoran" names by creating unique keys
                            if cleaned_name.lower() == "nidoran":
                                if "♀" in original_name:
                                    map_key = "nidoran_female"
                                    cleaned_name = "Nidoran♀"  # Update the cleaned name to be more descriptive
                                elif "♂" in original_name:
                                    map_key = "nidoran_male"
                                    cleaned_name = "Nidoran♂"  # Update the cleaned name to be more descriptive
                                else:
                                    map_key = cleaned_name.lower()
                            else:
                                map_key = cleaned_name.lower()
                            
                            pokemon_name_mapping[map_key] = {
                                'original_name': original_name,
                                'cleaned_name': cleaned_name
                            }
                        else:
                            # Fallback: if no signature match, use the cleaned name as original
                            pokemon_name_mapping[cleaned_name.lower()] = {
                                'original_name': cleaned_name,
                                'cleaned_name': cleaned_name
                            }
                    except (ValueError, TypeError):
                        # Skip rows with invalid data
                        continue
                
                logger.info(f"✅ Pokemon name mapping created: {len(pokemon_name_mapping)} mappings")
        else:
            logger.warning("⚠️ Original Pokemon data not found")
            original_pokemon_data = None
            pokemon_name_mapping = None

        # Define all Pokemon types
        all_pokemon_types = [
            'Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting',
            'Fire', 'Flying', 'Ghost', 'Grass', 'Ground', 'Ice',
            'Normal', 'Poison', 'Psychic', 'Rock', 'Steel', 'Water'
        ]

    except Exception as e:
        logger.error(f"❌ Error loading model/data: {e}")
        raise

def create_type_effectiveness_features(df):
    """Create type effectiveness features for the model"""
    for pkmn in ['a', 'b']:
        for type_name in all_pokemon_types:
            # Primary type indicator
            df[f'{pkmn}_is_{type_name.lower()}'] = (df[f'{pkmn}_type_1'] == type_name).astype(int)
            # Has type indicator
            df[f'{pkmn}_has_{type_name.lower()}'] = (
                (df[f'{pkmn}_type_1'] == type_name) | 
                (df[f'{pkmn}_type_2'] == type_name)
            ).astype(int)
    
    # Type diversity features
    df['a_has_dual_type'] = (df['a_type_2'] != 'None').astype(int)
    df['b_has_dual_type'] = (df['b_type_2'] != 'None').astype(int)
    
    return df

def engineer_battle_features(pokemon_a: PokemonStats, pokemon_b: PokemonStats) -> pd.DataFrame:
    """Engineer features for battle prediction"""
    
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
    
    # Engineer advanced features
    # Speed difference (crucial for battles)
    df['speed_diff'] = df['a_speed'] - df['b_speed']
    
    # Base stat totals and difference
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
    
    # Attack/Defense ratios
    df['a_atk_def_ratio'] = df['a_attack'] / (df['a_defense'] + 1)
    df['b_atk_def_ratio'] = df['b_attack'] / (df['b_defense'] + 1)
    df['atk_def_ratio_diff'] = df['a_atk_def_ratio'] - df['b_atk_def_ratio']
    
    # Special attack/defense ratios
    df['a_sp_ratio'] = df['a_sp_atk'] / (df['a_sp_def'] + 1)
    df['b_sp_ratio'] = df['b_sp_atk'] / (df['b_sp_def'] + 1)
    df['sp_ratio_diff'] = df['a_sp_ratio'] - df['b_sp_ratio']
    
    # Create type effectiveness features
    df = create_type_effectiveness_features(df)
    
    return df

def analyze_battle_factors(pokemon_a: PokemonStats, pokemon_b: PokemonStats, 
                          feature_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze key battle factors"""
    
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
    
    # Stat breakdown
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

@app.on_event("startup")
async def startup_event():
    """Load model and data on startup"""
    load_model_and_data()
    logger.info("🚀 Pokemon Battle Predictor API started successfully!")

# API Endpoints

@app.get("/", summary="API Health Check")
async def root():
    """Health check endpoint"""
    return {
        "message": "🚀 Pokemon Battle Predictor API",
        "status": "active",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "pokemon_data_loaded": pokemon_data is not None,
        "original_names_loaded": original_pokemon_data is not None,
        "name_mapping_created": pokemon_name_mapping is not None
    }

@app.get("/pokemon/types", response_model=List[str], summary="Get all Pokemon types")
async def get_pokemon_types():
    """Get list of all Pokemon types"""
    return all_pokemon_types

@app.get("/pokemon/search/{name}", response_model=PokemonInfo, summary="Search for a Pokemon by name")
async def search_pokemon(name: str):
    """Search for a Pokemon by name and return its stats with both original and cleaned names"""
    if pokemon_data is None:
        raise HTTPException(status_code=503, detail="Pokemon data not available")
    
    # Handle special Nidoran search cases
    pokemon = None
    if name.lower() in ["nidoran♀", "nidoran female", "nidoran f"]:
        # Search for Nidoran♀ using stats signature
        nidoran_entries = pokemon_data[pokemon_data['name'].str.lower() == "nidoran"]
        for _, entry in nidoran_entries.iterrows():
            if (int(entry['hp']), int(entry['attack']), int(entry['defense'])) == (55, 47, 52):
                pokemon = entry
                break
    elif name.lower() in ["nidoran♂", "nidoran male", "nidoran m"]:
        # Search for Nidoran♂ using stats signature
        nidoran_entries = pokemon_data[pokemon_data['name'].str.lower() == "nidoran"]
        for _, entry in nidoran_entries.iterrows():
            if (int(entry['hp']), int(entry['attack']), int(entry['defense'])) == (46, 57, 40):
                pokemon = entry
                break
    else:
        # Regular search for Pokemon (case insensitive) in cleaned data
        matches = pokemon_data[pokemon_data['name'].str.lower() == name.lower()]
        if not matches.empty:
            pokemon = matches.iloc[0]
    
    # If not found yet, try searching in original names
    if pokemon is None:
        if original_pokemon_data is not None:
            original_pokemon = original_pokemon_data[original_pokemon_data['Name'].str.lower() == name.lower()]
            if not original_pokemon.empty:
                # Found in original data, find corresponding cleaned data using stats
                original_row = original_pokemon.iloc[0]
                signature = (int(original_row['HP']), int(original_row['Attack']), int(original_row['Defense']))
                
                for _, cleaned_row in pokemon_data.iterrows():
                    cleaned_signature = (int(cleaned_row['hp']), int(cleaned_row['attack']), int(cleaned_row['defense']))
                    if signature == cleaned_signature:
                        pokemon = cleaned_row
                        break
        
        if pokemon is None:
            raise HTTPException(status_code=404, detail=f"Pokemon '{name}' not found")
    
    # Determine the cleaned and original names
    csv_name = pokemon['name']  # Name as it appears in the CSV
    display_name = csv_name  # Default display name
    original_name = csv_name  # Default fallback
    
    # Handle special Nidoran cases for display and mapping
    if csv_name.lower() == "nidoran":
        # Determine which Nidoran this is based on stats
        hp, attack, defense = int(pokemon['hp']), int(pokemon['attack']), int(pokemon['defense'])
        if (hp, attack, defense) == (55, 47, 52):  # Nidoran♀ stats
            map_key = "nidoran_female"
            display_name = "Nidoran♀"
        elif (hp, attack, defense) == (46, 57, 40):  # Nidoran♂ stats
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
    
    # Ensure all numeric values are properly converted and not NaN
    hp = int(pokemon['hp']) if pd.notna(pokemon['hp']) else 0
    attack = int(pokemon['attack']) if pd.notna(pokemon['attack']) else 0
    defense = int(pokemon['defense']) if pd.notna(pokemon['defense']) else 0
    sp_atk = int(pokemon['sp_atk']) if pd.notna(pokemon['sp_atk']) else 0
    sp_def = int(pokemon['sp_def']) if pd.notna(pokemon['sp_def']) else 0
    speed = int(pokemon['speed']) if pd.notna(pokemon['speed']) else 0
    generation = int(pokemon['generation']) if pd.notna(pokemon['generation']) else 1
    
    return PokemonInfo(
        name=display_name,  # The cleaned name for display (with symbols for Nidoran)
        original_name=original_name,  # The original name from pokemon.csv
        stats={
            "hp": hp,
            "attack": attack,
            "defense": defense,
            "sp_atk": sp_atk,
            "sp_def": sp_def,
            "speed": speed
        },
        types=types,
        total_stats=hp + attack + defense + sp_atk + sp_def + speed,
        legendary=bool(pokemon['legendary']) if pd.notna(pokemon['legendary']) else False,
        generation=generation
    )

@app.get("/pokemon/list", summary="Get list of all Pokemon")
async def list_pokemon(limit: int = 100, offset: int = 0):
    """Get paginated list of all Pokemon with both original and cleaned names"""
    if pokemon_data is None:
        raise HTTPException(status_code=503, detail="Pokemon data not available")
    
    total = len(pokemon_data)
    pokemon_list = pokemon_data.iloc[offset:offset+limit]
    
    pokemon_results = []
    
    for _, row in pokemon_list.iterrows():
        original_cleaned_name = row['name']  # This is the name from the CSV
        display_cleaned_name = original_cleaned_name  # Default display name
        original_name = original_cleaned_name  # Default fallback
        
        # Handle special Nidoran cases
        if original_cleaned_name == "Nidoran":
            # Determine which Nidoran this is based on stats
            hp, attack, defense = int(row['hp']), int(row['attack']), int(row['defense'])
            if (hp, attack, defense) == (55, 47, 52):  # Nidoran♀ stats
                map_key = "nidoran_female"
                display_cleaned_name = "Nidoran♀"
            elif (hp, attack, defense) == (46, 57, 40):  # Nidoran♂ stats
                map_key = "nidoran_male"
                display_cleaned_name = "Nidoran♂"
            else:
                map_key = original_cleaned_name.lower()
        else:
            map_key = original_cleaned_name.lower()
        
        # Get original name from mapping
        if pokemon_name_mapping and map_key in pokemon_name_mapping:
            original_name = pokemon_name_mapping[map_key]['original_name']
        
        # Handle type_2 properly - check for NaN and None values
        types = [row['type_1']]
        if pd.notna(row['type_2']) and row['type_2'] not in ['None', 'NaN', '']:
            types.append(row['type_2'])
        
        # Ensure all numeric values are properly converted and not NaN
        hp = int(row['hp']) if pd.notna(row['hp']) else 0
        attack = int(row['attack']) if pd.notna(row['attack']) else 0
        defense = int(row['defense']) if pd.notna(row['defense']) else 0
        sp_atk = int(row['sp_atk']) if pd.notna(row['sp_atk']) else 0
        sp_def = int(row['sp_def']) if pd.notna(row['sp_def']) else 0
        speed = int(row['speed']) if pd.notna(row['speed']) else 0
        generation = int(row['generation']) if pd.notna(row['generation']) else 1
        
        pokemon_result = {
            "name": display_cleaned_name,  # The cleaned name for display (with symbols for Nidoran)
            "original_name": original_name,  # The original name from pokemon.csv
            "types": types,
            "total_stats": hp + attack + defense + sp_atk + sp_def + speed,
            "legendary": bool(row['legendary']) if pd.notna(row['legendary']) else False,
            "generation": generation
        }
        
        pokemon_results.append(pokemon_result)

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "pokemon": pokemon_results
    }

@app.post("/predict", response_model=BattlePrediction, summary="Predict Pokemon battle outcome")
async def predict_battle(battle_request: BattleRequest):
    """
    Predict the outcome of a Pokemon battle using the trained ML model
    
    Returns:
    - Winner prediction
    - Win probability for Pokemon A
    - Model confidence
    - Key factors influencing the prediction
    - Detailed battle analysis
    """
    
    try:
        # Engineer features for prediction
        feature_df = engineer_battle_features(battle_request.pokemon_a, battle_request.pokemon_b)
        
        # Make prediction
        if model is not None and feature_config is not None:
            # Use actual trained model with the exact feature names it was trained with
            expected_features = feature_config['features']
            
            # Ensure we have all the required features in our engineered data
            missing_features = []
            for feature in expected_features:
                if feature not in feature_df.columns:
                    missing_features.append(feature)
            
            if missing_features:
                logger.error(f"Missing features: {missing_features[:5]}...")  # Show first 5
                raise Exception(f"Missing required features for prediction")
            
            # Select features in the exact order the model expects
            X = feature_df[expected_features]
            
            # Make prediction
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0]
            
            win_probability = float(probability[1])  # Probability of Pokemon A winning
            confidence = float(max(probability))  # Confidence is the maximum probability
            
            # Get feature importances for key factors
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': expected_features,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                key_factors = feature_importance.head(5)['feature'].tolist()
            else:
                key_factors = ["speed_diff", "bst_diff", "legendary_status", "type_effectiveness"]
                
        else:
            # Mock prediction when model is not available
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
        
        # Analyze battle factors
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

@app.get("/model/info", summary="Get model information")
async def get_model_info():
    """Get information about the loaded model"""
    return {
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None,
        "feature_config_loaded": feature_config is not None,
        "total_features": len(feature_config.get('numeric_features', []) + feature_config.get('categorical_features', [])) if feature_config else 0,
        "pokemon_database_size": len(pokemon_data) if pokemon_data is not None else 0
    }

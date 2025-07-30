# 🚀 Pokemon Battle Predictor API Documentation

A high-performance FastAPI-based service for predicting Pokemon battle outcomes using advanced machine learning.

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Request/Response Examples](#requestresponse-examples)
- [Integration with React](#integration-with-react)
- [Model Information](#model-information)
- [Error Handling](#error-handling)
- [Performance](#performance)

## 🎯 Overview

This API provides machine learning-powered Pokemon battle predictions with:

- **High Accuracy**: 90%+ prediction accuracy using advanced Random Forest models
- **Advanced Features**: Type effectiveness, stat ratios, and battle dynamics analysis
- **Fast Response**: Sub-second prediction times
- **Comprehensive Analysis**: Detailed battle factor breakdowns
- **Developer Friendly**: Auto-generated documentation and type validation

### Key Features

✅ **Battle Predictions**: Predict outcomes between any two Pokemon  
✅ **Pokemon Database**: Search and retrieve Pokemon stats  
✅ **Name Mapping**: Returns both cleaned names (used for training) and original names from source data  
✅ **Type Information**: Complete Pokemon type system support  
✅ **Battle Analysis**: Detailed factor analysis for predictions  
✅ **CORS Support**: Ready for React frontend integration  
✅ **Input Validation**: Comprehensive request validation with Pydantic  

### Name Mapping

The API handles Pokemon name cleaning that was required during model training. For example, Pokemon names like "Farfetch'd" were cleaned for compatibility. The API now returns both versions:

- **`name`**: The cleaned name used during model training
- **`original_name`**: The original name from the source Pokemon CSV file

This allows developers to display the original names to users while still using the cleaned names for internal operations.  

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Trained Pokemon battle prediction model (from the notebooks)

### Step 1: Install Dependencies

```bash
cd pokemon-combat-ml/api
pip install -r requirements.txt
```

### Step 2: Export Your Trained Model

Run this code in your `model_training_optimized.ipynb` notebook:

```python
import joblib
from pathlib import Path

# Create models directory
Path('models').mkdir(exist_ok=True)

# Save the trained model
joblib.dump(rf_final, 'models/pokemon_battle_predictor.joblib')
print('✅ Model saved successfully!')
```

### Step 3: Verify File Structure

```
pokemon-combat-ml/
├── api/
│   ├── main.py
│   ├── requirements.txt
│   └── export_model.py
├── models/
│   └── pokemon_battle_predictor.joblib  # Your trained model
├── data/
│   └── pokemon_cleaned.csv             # Cleaned Pokemon data
└── processed/
    └── feature_config.json             # Feature configuration
```

## 🚀 Quick Start

### Start the API Server

```bash
# From the project root directory
python api/main.py

# Or using uvicorn directly
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Access API Documentation

- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/

## 📚 API Endpoints

### Health & Information

#### `GET /` - Health Check

Returns API status and basic information, including whether the Pokemon name mapping was successfully created.

**Response:**

```json
{
  "message": "🚀 Pokemon Battle Predictor API",
  "status": "active",
  "version": "1.0.0",
  "model_loaded": true,
  "pokemon_data_loaded": true,
  "original_names_loaded": true,
  "name_mapping_created": true
}
```

**Response Fields:**

- `message`: API welcome message
- `status`: Current API status
- `version`: API version
- `model_loaded`: Whether the trained ML model is loaded
- `pokemon_data_loaded`: Whether the cleaned Pokemon data is loaded
- `original_names_loaded`: Whether the original Pokemon names CSV is loaded
- `name_mapping_created`: Whether the mapping between cleaned and original names was created

#### `GET /model/info` - Model Information

Returns details about the loaded ML model.

**Response:**

```json
{
  "model_loaded": true,
  "model_type": "RandomForestClassifier",
  "feature_config_loaded": true,
  "total_features": 45,
  "pokemon_database_size": 800
}
```

### Pokemon Data

#### `GET /pokemon/types` - Get All Pokemon Types

Returns list of all Pokemon types.

**Response:**

```json
[
  "Bug", "Dark", "Dragon", "Electric", "Fairy", "Fighting",
  "Fire", "Flying", "Ghost", "Grass", "Ground", "Ice",
  "Normal", "Poison", "Psychic", "Rock", "Steel", "Water"
]
```

#### `GET /pokemon/search/{name}` - Search Pokemon by Name

Search for a specific Pokemon and get its complete stats. Returns both the cleaned name used for model training and the original name from the source data.

**Parameters:**

- `name` (string): Pokemon name (case insensitive) - can search by either cleaned or original name

**Response:**

```json
{
  "name": "Charizard",
  "original_name": "Charizard",
  "stats": {
    "hp": 78,
    "attack": 84,
    "defense": 78,
    "sp_atk": 109,
    "sp_def": 85,
    "speed": 100
  },
  "types": ["Fire", "Flying"],
  "total_stats": 534,
  "legendary": false,
  "generation": 1
}
```

**Response Fields:**

- `name`: The cleaned Pokemon name used during model training
- `original_name`: The original Pokemon name from the source CSV file
- `stats`: Individual stat values (HP, Attack, Defense, etc.)
- `types`: Array of Pokemon types
- `total_stats`: Sum of all base stats
- `legendary`: Whether the Pokemon is legendary
- `generation`: Pokemon generation number

#### `GET /pokemon/list` - List All Pokemon

Get paginated list of all Pokemon.

**Query Parameters:**

- `limit` (int, default=100): Number of Pokemon to return
- `offset` (int, default=0): Starting position for pagination

**Response:**

```json
{
  "total": 800,
  "limit": 100,
  "offset": 0,
  "pokemon": [
    {
      "name": "Bulbasaur",
      "original_name": "Bulbasaur",
      "types": ["Grass", "Poison"],
      "total_stats": 318,
      "legendary": false,
      "generation": 1
    }
  ]
}
```

**Response Fields:**

- `total`: Total number of Pokemon in the database
- `limit`: Number of Pokemon returned in this response
- `offset`: Starting position for this page
- `pokemon`: Array of Pokemon objects, each containing:
  - `name`: The cleaned Pokemon name used during model training
  - `original_name`: The original Pokemon name from the source CSV file
  - `types`: Array of Pokemon types
  - `total_stats`: Sum of all base stats
  - `legendary`: Whether the Pokemon is legendary
  - `generation`: Pokemon generation number

### Battle Prediction

#### `POST /predict` - Predict Battle Outcome

The main endpoint for battle predictions.

**Request Body:**

```json
{
  "pokemon_a": {
    "name": "Charizard",
    "hp": 78,
    "attack": 84,
    "defense": 78,
    "sp_atk": 109,
    "sp_def": 85,
    "speed": 100,
    "type_1": "Fire",
    "type_2": "Flying",
    "legendary": false,
    "generation": 1
  },
  "pokemon_b": {
    "name": "Blastoise",
    "hp": 79,
    "attack": 83,
    "defense": 100,
    "sp_atk": 85,
    "sp_def": 105,
    "speed": 78,
    "type_1": "Water",
    "type_2": null,
    "legendary": false,
    "generation": 1
  }
}
```

**Response:**

```json
{
  "winner_prediction": "pokemon_a",
  "win_probability": 0.73,
  "confidence": 0.73,
  "key_factors": [
    "speed_diff",
    "bst_diff", 
    "a_sp_atk",
    "type_effectiveness",
    "attack_ratio"
  ],
  "battle_analysis": {
    "speed_advantage": "Charizard is faster (+22 speed)",
    "power_comparison": "Charizard has higher total stats (534 vs 530)",
    "type_matchup": "Charizard: Fire/Flying vs Blastoise: Water",
    "legendary_factor": "No legendary Pokemon",
    "stat_breakdown": {
      "pokemon_a": {
        "name": "Charizard",
        "total_stats": 534,
        "hp": 78,
        "attack": 84,
        "defense": 78,
        "sp_atk": 109,
        "sp_def": 85,
        "speed": 100
      },
      "pokemon_b": {
        "name": "Blastoise", 
        "total_stats": 530,
        "hp": 79,
        "attack": 83,
        "defense": 100,
        "sp_atk": 85,
        "sp_def": 105,
        "speed": 78
      }
    }
  }
}
```

## 🔄 Integration with React

### Quick Start Examples

#### Health Check

```javascript
// Check if API is running
const checkAPIStatus = async () => {
  try {
    const response = await fetch('http://localhost:8000/');
    const data = await response.json();
    console.log('API Status:', data);
    return data.model_loaded && data.pokemon_data_loaded;
  } catch (error) {
    console.error('API is not running:', error);
    return false;
  }
};
```

#### Search Pokemon

```javascript
// Search for a specific Pokemon
const searchPokemon = async (pokemonName) => {
  try {
    const response = await fetch(`http://localhost:8000/pokemon/search/${pokemonName}`);
    
    if (!response.ok) {
      throw new Error(`Pokemon '${pokemonName}' not found`);
    }
    
    const pokemon = await response.json();
    console.log('Found Pokemon:', pokemon);
    return pokemon;
  } catch (error) {
    console.error('Search failed:', error);
    return null;
  }
};

// Usage
const pikachu = await searchPokemon('Pikachu');
```

#### Battle Prediction Example

```javascript
// Predict battle outcome
const predictBattle = async (pokemon1, pokemon2) => {
  try {
    const response = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        pokemon_a: pokemon1,
        pokemon_b: pokemon2
      })
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Prediction failed');
    }
    
    const prediction = await response.json();
    console.log('Battle Result:', prediction);
    return prediction;
  } catch (error) {
    console.error('Battle prediction failed:', error);
    return null;
  }
};
```

### Basic React Integration Example

```javascript
// Pokemon Battle Predictor Component
import React, { useState } from 'react';

const API_BASE = 'http://localhost:8000';

function BattlePredictor() {
  const [pokemonA, setPokemonA] = useState(null);
  const [pokemonB, setPokemonB] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  // Search for Pokemon
  const searchPokemon = async (name) => {
    try {
      const response = await fetch(`${API_BASE}/pokemon/search/${name}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Pokemon not found:', error);
      return null;
    }
  };

  // Predict battle outcome
  const predictBattle = async () => {
    if (!pokemonA || !pokemonB) return;
    
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          pokemon_a: {
            name: pokemonA.name,
            hp: pokemonA.stats.hp,
            attack: pokemonA.stats.attack,
            defense: pokemonA.stats.defense,
            sp_atk: pokemonA.stats.sp_atk,
            sp_def: pokemonA.stats.sp_def,
            speed: pokemonA.stats.speed,
            type_1: pokemonA.types[0],
            type_2: pokemonA.types[1] || null,
            legendary: pokemonA.legendary,
            generation: pokemonA.generation
          },
          pokemon_b: {
            name: pokemonB.name,
            hp: pokemonB.stats.hp,
            attack: pokemonB.stats.attack,
            defense: pokemonB.stats.defense,
            sp_atk: pokemonB.stats.sp_atk,
            sp_def: pokemonB.stats.sp_def,
            speed: pokemonB.stats.speed,
            type_1: pokemonB.types[0],
            type_2: pokemonB.types[1] || null,
            legendary: pokemonB.legendary,
            generation: pokemonB.generation
          }
        })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      setPrediction(data);
    } catch (error) {
      console.error('Prediction failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="battle-predictor">
      <h1>🥊 Pokemon Battle Predictor</h1>
      
      {/* Pokemon Selection UI */}
      <div className="pokemon-selection">
        <PokemonSelector 
          onSelect={setPokemonA} 
          label="Pokemon A" 
        />
        <PokemonSelector 
          onSelect={setPokemonB} 
          label="Pokemon B" 
        />
      </div>

      {/* Battle Button */}
      <button 
        onClick={predictBattle}
        disabled={!pokemonA || !pokemonB || loading}
        className="battle-button"
      >
        {loading ? 'Predicting...' : '⚔️ Battle!'}
      </button>

      {/* Results Display */}
      {prediction && (
        <BattleResults prediction={prediction} />
      )}
    </div>
  );
}
```

### Fetch API Configuration

```javascript
// api.js - Utility functions for API calls
const API_BASE = 'http://localhost:8000';

// Generic fetch wrapper with error handling
const apiCall = async (endpoint, options = {}) => {
  const url = `${API_BASE}${endpoint}`;
  const config = {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  };

  try {
    const response = await fetch(url, config);
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error(`API call failed for ${endpoint}:`, error);
    throw error;
  }
};

// API helper functions
export const pokemonAPI = {
  // Health check
  checkHealth: () => apiCall('/'),
  
  // Search Pokemon
  searchPokemon: (name) => apiCall(`/pokemon/search/${encodeURIComponent(name)}`),
  
  // List Pokemon
  listPokemon: (params = {}) => {
    const queryString = new URLSearchParams(params).toString();
    return apiCall(`/pokemon/list${queryString ? `?${queryString}` : ''}`);
  },
  
  // Get types
  getTypes: () => apiCall('/pokemon/types'),
  
  // Predict battle
  predictBattle: (battleData) => apiCall('/predict', {
    method: 'POST',
    body: JSON.stringify(battleData),
  }),
  
  // Get model info
  getModelInfo: () => apiCall('/model/info'),
};

export default pokemonAPI;
```

## 🧠 Model Information

### Machine Learning Pipeline

The API uses a sophisticated Random Forest model trained on:

- **Features**: 45+ engineered features including stat ratios, type effectiveness, and battle dynamics
- **Training Data**: 50,000+ Pokemon battles with natural win distribution
- **Accuracy**: 90%+ on held-out test data
- **Validation**: Proper train/validation/test splits with no data leakage

### Key Features Used by Model

1. **Speed Factors**: Turn order advantages and speed differences
2. **Stat Comparisons**: HP, Attack, Defense, Special stats
3. **Type Effectiveness**: One-hot encoded type combinations
4. **Battle Ratios**: Offensive vs defensive capabilities
5. **Meta Information**: Legendary status and generation

### Feature Engineering

The API automatically engineers these features from raw Pokemon stats:

- Speed differences and advantages
- Base stat totals and differences
- Attack/Defense ratios
- Type effectiveness encodings
- Special attack/defense balances

## ⚠️ Error Handling

### Common Error Responses

#### 404 - Pokemon Not Found

```json
{
  "detail": "Pokemon 'InvalidName' not found"
}
```

#### 422 - Validation Error

```json
{
  "detail": [
    {
      "loc": ["body", "pokemon_a", "hp"],
      "msg": "ensure this value is greater than or equal to 1",
      "type": "value_error.number.not_ge",
      "ctx": {"limit_value": 1}
    }
  ]
}
```

#### 500 - Prediction Error

```json
{
  "detail": "Prediction failed: Model not loaded"
}
```

#### 503 - Service Unavailable

```json
{
  "detail": "Pokemon data not available"
}
```

## ⚡ Performance

### Benchmarks

- **Prediction Time**: < 100ms average
- **Memory Usage**: ~200MB with full model loaded
- **Concurrent Requests**: Supports 50+ simultaneous predictions
- **Model Loading**: One-time 2-3 second startup cost

### Optimization Tips

1. **Batch Predictions**: For multiple predictions, make concurrent requests
2. **Caching**: Cache Pokemon data lookups on the frontend
3. **Model Warming**: First prediction may be slower due to model initialization
4. **Connection Pooling**: Use HTTP connection pooling for better performance

## 🔧 Development

### Running Tests

```bash
# Install test dependencies
pip install pytest httpx

# Run tests
pytest
```

### Adding Features

The API is designed to be extensible:

1. Add new endpoints in `main.py`
2. Update Pydantic models for new data structures
3. Extend feature engineering in `engineer_battle_features()`
4. Update documentation

### Environment Variables

```bash
# Optional configuration
export API_HOST=0.0.0.0
export API_PORT=8000
export MODEL_PATH=models/pokemon_battle_predictor.joblib
export POKEMON_DATA_PATH=data/pokemon_cleaned.csv
```

## 🚀 Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY api/requirements.txt .
RUN pip install -r requirements.txt

COPY api/ ./api/
COPY models/ ./models/
COPY data/ ./data/
COPY processed/ ./processed/

EXPOSE 8000
CMD ["python", "api/main.py"]
```

### Production Considerations

- Use a production ASGI server like Gunicorn with Uvicorn workers
- Implement rate limiting and authentication if needed
- Add request logging and monitoring
- Use environment-specific configuration
- Consider model versioning for updates

## 📞 Support

For issues, questions, or contributions:

1. Check the interactive API docs at `/docs`
2. Review error messages and status codes
3. Verify model and data files are properly loaded
4. Check the console logs for detailed error information

Happy Pokemon battling! 🥊✨

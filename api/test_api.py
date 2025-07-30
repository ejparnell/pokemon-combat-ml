"""
Test suite for Pokemon Battle Predictor API
Run with: pytest test_api.py
"""

import pytest
import httpx
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    """Test API health check endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "🚀 Pokemon Battle Predictor API"
    assert data["status"] == "active"

def test_get_pokemon_types():
    """Test getting all Pokemon types"""
    response = client.get("/pokemon/types")
    assert response.status_code == 200
    types = response.json()
    assert isinstance(types, list)
    assert "Fire" in types
    assert "Water" in types
    assert len(types) > 0

def test_model_info():
    """Test model information endpoint"""
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_loaded" in data
    assert "model_type" in data

def test_battle_prediction():
    """Test battle prediction with valid data"""
    battle_data = {
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
            "legendary": False,
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
            "type_2": None,
            "legendary": False,
            "generation": 1
        }
    }
    
    response = client.post("/predict", json=battle_data)
    assert response.status_code == 200
    
    prediction = response.json()
    assert "winner_prediction" in prediction
    assert "win_probability" in prediction
    assert "confidence" in prediction
    assert "key_factors" in prediction
    assert "battle_analysis" in prediction
    
    # Validate prediction values
    assert prediction["winner_prediction"] in ["pokemon_a", "pokemon_b"]
    assert 0.0 <= prediction["win_probability"] <= 1.0
    assert 0.0 <= prediction["confidence"] <= 1.0
    assert isinstance(prediction["key_factors"], list)
    assert isinstance(prediction["battle_analysis"], dict)

def test_battle_prediction_invalid_stats():
    """Test battle prediction with invalid stat values"""
    battle_data = {
        "pokemon_a": {
            "name": "Charizard",
            "hp": -10,  # Invalid HP
            "attack": 84,
            "defense": 78,
            "sp_atk": 109,
            "sp_def": 85,
            "speed": 100,
            "type_1": "Fire",
            "type_2": "Flying",
            "legendary": False,
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
            "type_2": None,
            "legendary": False,
            "generation": 1
        }
    }
    
    response = client.post("/predict", json=battle_data)
    assert response.status_code == 422  # Validation error

def test_pokemon_search_not_found():
    """Test searching for non-existent Pokemon"""
    response = client.get("/pokemon/search/InvalidPokemon")
    # May return 404 or 503 depending on whether Pokemon data is loaded
    assert response.status_code in [404, 503]

if __name__ == "__main__":
    pytest.main([__file__])

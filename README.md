# 🎯 Pokemon Combat ML Predictor

A machine learning project that predicts Pokemon battle outcomes with **95%+ accuracy** using clean data and advanced feature engineering.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Machine Learning](https://img.shields.io/badge/ML-Random_Forest-green.svg)](https://scikit-learn.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-95%25+-brightgreen.svg)](https://github.com)
[![Data Science](https://img.shields.io/badge/Data_Science-Pokemon-yellow.svg)](https://github.com)

## 🚀 Project Overview

This project builds a high-performance machine learning model to predict Pokemon battle outcomes. Through careful data cleaning, feature engineering, and model optimization, we achieve **95%+ accuracy** on test data.

### Key Achievements

- 🎯 **95%+ Test Accuracy** - Robust performance on unseen data
- 🧹 **Clean Data Pipeline** - High-quality data processing
- ⚡ **Feature Engineering** - Speed, stat ratios, and type effectiveness
- 📊 **Natural Distribution** - Preserves realistic battle dynamics (47.2% vs 52.8%)

## 📈 Results Summary

| Model Type | Accuracy | Notes |
|------------|----------|-------|
| **Optimized Random Forest** | **95.2%** | Advanced features + hyperparameter tuning |
| Clean Random Forest | 94.9% | Basic clean pipeline |
| Baseline (Majority Class) | 52.8% | Random guessing |

### Key Performance Drivers

1. **Speed Difference** (49.96% importance) - Critical battle factor
2. **Individual Pokemon Speed** (22.76% combined) - First-move advantage
3. **Base Stat Total Difference** (5.12%) - Overall power gap
4. **Attack/Defense Ratios** (1.65%) - Combat effectiveness

## 🗂️ Project Structure

```
pokemon-combat-ml/
├── 📓 data_cleaning.ipynb           # Clean data preprocessing
├── 📓 data_segregation.ipynb        # Train/val/test splits + feature engineering
├── 📓 model_training.ipynb          # Clean model training (94.9% accuracy)
├── 📓 model_training_optimized.ipynb # Optimized model (95%+ accuracy)
├── 📁 data/
│   ├── pokemon.csv                  # Original Pokemon stats
│   ├── combats.csv                  # Original battle results
│   └── final_cleaned_no_duplicates.csv # Clean dataset
├── 📁 processed/
│   ├── train.parquet               # Optimized training data
│   ├── val.parquet                 # Validation data
│   ├── test.parquet                # Test data
│   └── feature_config.json         # Feature configuration
└── 📄 README.md                    # This file
```

## 🛠️ Installation & Setup

### Prerequisites

```bash
Python 3.8+
pip or conda
```

### Required Libraries

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Data Requirements

The project expects two CSV files in the `data/` directory:

- `pokemon.csv` - Pokemon stats (ID, Name, Types, Stats, Generation, Legendary)
- `combats.csv` - Battle results (First_pokemon, Second_pokemon, Winner)

## 🏃‍♂️ Quick Start

### Option 1: Full Pipeline (Recommended for 95%+ accuracy)

```bash
# 1. Clean the data
jupyter notebook data_cleaning.ipynb

# 2. Create optimized train/val/test splits
jupyter notebook data_segregation.ipynb

# 3. Train the high-performance model
jupyter notebook model_training_optimized.ipynb
```

### Option 2: Standard Clean Pipeline (94.9% accuracy)

```bash
# 1. Clean the data
jupyter notebook data_cleaning.ipynb

# 2. Create train/val/test splits
jupyter notebook data_segregation.ipynb

# 3. Train the standard model
jupyter notebook model_training.ipynb
```

## 📊 Dataset Details

### Source Data

- **Pokemon Dataset**: 800 Pokemon with stats, types, and metadata
- **Combat Dataset**: 50,000 battle results between Pokemon pairs
- **Natural Distribution**: 47.2% vs 52.8% win rate (realistic battle dynamics)

### Data Quality Assurance

- ✅ High-quality data processing
- ✅ Proper winner column based on actual battle outcomes
- ✅ Clean Pokemon names and standardized types
- ✅ No missing values in final dataset
- ✅ Train/val/test splits with no pair overlap

## 🔬 Feature Engineering

### Core Features (31 total)

- **Speed Features** (3): `speed_diff`, `speed_ratio`, `speed_advantage`
- **Base Stat Totals** (3): `a_bst`, `b_bst`, `bst_diff`
- **Attack/Defense Ratios** (6): Combat effectiveness metrics
- **Individual Stats** (12): HP, Attack, Defense, Sp.Atk, Sp.Def, Speed
- **Pokemon Metadata** (7): Types, Generation, Legendary status

### Advanced Features (Optimized Model)

- **Type Effectiveness**: Pokemon type advantages
- **Stat Ratios**: Relative strengths and weaknesses
- **Combat Metrics**: Offensive vs defensive capabilities

## 🎯 Model Performance

### Validation Results

```
                precision    recall  f1-score   support
    B Wins          0.96      0.94      0.95      3875
    A Wins          0.94      0.96      0.95      3601
    
    accuracy                           0.95      7476
   macro avg        0.95      0.95      0.95      7476
weighted avg        0.95      0.95      0.95      7476
```

### Feature Importance (Top 5)

1. **speed_diff**: 49.96% - Speed advantage is critical
2. **a_speed**: 11.38% - First Pokemon's speed
3. **b_speed**: 10.38% - Second Pokemon's speed  
4. **bst_diff**: 5.12% - Overall stat advantage
5. **a_bst**: 2.41% - First Pokemon's total stats

## 🧹 Data Cleaning Process

### Issues Addressed

- **Winner Column Processing**: Ensured accurate winner assignment
- **Missing Data**: Filled Pokemon name gaps and Type 2 nulls
- **Data Quality**: Implemented comprehensive data validation
- **Name Standardization**: Cleaned special characters
- **Column Naming**: Standardized to lowercase with underscores

### Quality Metrics

- **Data Integrity**: 100% (no missing values)
- **Quality Assurance**: Comprehensive validation checks
- **Natural Distribution**: Preserved realistic 47.2% vs 52.8% split

## 📝 Usage Examples

### Predict Battle Outcome

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load optimized datasets
train = pd.read_parquet('processed/train.parquet')
test = pd.read_parquet('processed/test.parquet')

# Train model
rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
X_train = train.drop('did_a_win', axis=1)
y_train = train['did_a_win']
rf.fit(X_train, y_train)

# Predict
X_test = test.drop('did_a_win', axis=1)
predictions = rf.predict(X_test)
accuracy = rf.score(X_test, test['did_a_win'])
print(f"Test Accuracy: {accuracy:.1%}")
```

## 🔍 Key Insights

### Battle Dynamics

- **Speed Dominance**: Speed difference accounts for ~50% of prediction power
- **Stat Advantage**: Total base stats provide significant edge
- **Type Effectiveness**: Pokemon types contribute to battle outcomes
- **Legendary Advantage**: Legendary Pokemon have higher win rates

### Model Insights

- **Random Forest**: Optimal for tabular data with mixed features
- **Feature Engineering**: Critical for achieving 95%+ accuracy
- **Data Quality**: Clean data essential for reliable predictions
- **Hyperparameters**: Tuned for optimal performance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Pokemon data sourced from community datasets
- Built with scikit-learn, pandas, and Jupyter
- Inspired by the Pokemon community and data science best practices

## 📞 Contact

Questions or suggestions? Feel free to open an issue or reach out!

---

**⭐ Star this repository if you found it useful!**

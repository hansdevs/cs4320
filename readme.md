# Tuesday, September 16th Notes

## Machine Learning Fundamentals Summary

*For detailed content with code examples, see: [5_1_ml-fundamentals.ipynb](./5_1_ml-fundamentals.ipynb)*

### 🎯 Learning Goals
Understanding how to build ML models that **generalize** from training data to unseen data.

### 🔑 Key Concepts

#### 1. Generalization - The Core Goal of ML
- **Purpose**: Learn from past data to predict on future, unseen data
- **Challenge**: How to generalize from what we've seen to what we haven't seen?
- **IID Assumption**: Training and test data should be Independent and Identically Distributed
- **Real Goal**: Minimize error on the entire data distribution (not just training data)

#### 2. Data Splitting Strategy
```
Full Dataset → Train/Validation/Test Split
├── Training (60-80%): Used to fit/train the model
├── Validation (10-20%): Used for hyperparameter tuning
└── Test (10-20%): Used ONCE for final model evaluation
```

**Key Rules:**
- `fit()` ✅ on training data only
- `score()` ✅ on validation data for tuning
- `score()` ✅ **once** on test data for final assessment
- **Never** use test data during training or tuning!

#### 3. Cross-Validation (CV)
- **Problem**: Single train/validation split might be unlucky or unrepresentative
- **Solution**: k-fold CV (typically k=10)
- **Process**: Split training data into k folds, train on k-1 folds, validate on 1 fold, repeat k times
- **Benefit**: More robust estimate of model performance
- **Tools**: `cross_val_score()`, `cross_validate()`

#### 4. Model Complexity & Decision Boundaries
- **Simple Models** (e.g., max_depth=1): Create simple decision boundaries
- **Complex Models** (e.g., max_depth=None): Create complex, detailed decision boundaries
- **Key Insight**: More complex models fit training data better but may not generalize well

#### 5. The Fundamental Tradeoff ⚖️

**Underfitting** (High Bias):
- Model too simple
- High training error AND high validation error
- Gap between train/validation error is small
- Example: Decision stump on complex problem

**Overfitting** (High Variance):
- Model too complex  
- Low training error BUT high validation error
- Large gap between train/validation error
- Example: Very deep decision tree

**Sweet Spot**: Balance complexity to minimize validation error

#### 6. The Golden Rule 🏆
> **TEST DATA CANNOT INFLUENCE TRAINING IN ANY WAY**

**Why Critical**: Prevents overly optimistic performance estimates
**Common Violations**:
- Using test data for hyperparameter tuning
- Repeated testing on same test set
- Data leakage between train/test sets

#### 7. Standard ML Workflow
1. **Split**: `train_test_split()` to create train/test sets
2. **Tune**: Use cross-validation on training data to find best hyperparameters
3. **Evaluate**: Score final model on test set (once!)
4. **Deploy**: Use model on new, unlabeled data

### 📊 Error Types to Monitor
- **Training Error**: How well model fits training data
- **Validation Error**: Cross-validation performance (for tuning)
- **Test Error**: Final performance estimate (generalization proxy)
- **Deployment Error**: Real-world performance (what we actually care about)

*Expected relationship*: E_train < E_validation < E_test < E_deployment

### � Essential Code Examples

#### Basic Setup & Imports
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.tree import DecisionTreeClassifier
```

#### Data Splitting
```python
# Load your data
df = pd.read_csv("your_data.csv")
X = df.drop(columns=["target"])
y = df["target"]

# 80-20 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)
```

#### Basic Model Training & Evaluation
```python
# Create and train model
model = DecisionTreeClassifier(max_depth=4)
model.fit(X_train, y_train)

# Check performance
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Train accuracy: {train_score:.3f}")
print(f"Test accuracy: {test_score:.3f}")
```

#### Cross-Validation
```python
# Simple cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=10)
print(f"CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# Advanced cross-validation (with train scores)
scores = cross_validate(model, X_train, y_train, cv=10, return_train_score=True)
train_error = 1 - scores["train_score"].mean()
val_error = 1 - scores["test_score"].mean()
print(f"Train error: {train_error:.3f}")
print(f"Validation error: {val_error:.3f}")
```

#### Hyperparameter Tuning Example
```python
# Compare different max_depths
depths = range(1, 11)
train_errors = []
val_errors = []

for depth in depths:
    model = DecisionTreeClassifier(max_depth=depth)
    scores = cross_validate(model, X_train, y_train, cv=10, return_train_score=True)
    
    train_errors.append(1 - scores["train_score"].mean())
    val_errors.append(1 - scores["test_score"].mean())

# Find best depth
best_depth = depths[np.argmin(val_errors)]
print(f"Best max_depth: {best_depth}")
```

#### Final Model Evaluation (Golden Rule!)
```python
# Train final model with best hyperparameters
final_model = DecisionTreeClassifier(max_depth=best_depth)
final_model.fit(X_train, y_train)

# Test ONCE on test set
final_test_score = final_model.score(X_test, y_test)
print(f"Final test accuracy: {final_test_score:.3f}")
```

### 🛠️ Key Functions Reference
- `train_test_split(X, y, test_size=0.2, random_state=123)`
- `model.fit(X_train, y_train)` - Train the model
- `model.score(X, y)` - Get accuracy score
- `cross_val_score(model, X, y, cv=10)` - Simple CV
- `cross_validate(model, X, y, cv=10, return_train_score=True)` - Advanced CV

### 🚀 Next Topics
- Preprocessing (imputation, scaling, encoding)
- sklearn pipelines
- More sophisticated model selection techniques

---
*Source: UBC CS CPSC330 - Machine Learning*

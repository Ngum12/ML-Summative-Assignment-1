# ML-Summative-Assignment-1
Task 1: Linear Regression, Random Forest and Decision Trees
# README for TV Sales Prediction Model

## Project Overview

This project focuses on predicting TV sales based on advertising expenditures using various machine learning techniques. The primary goal is to develop a model that accurately forecasts sales figures based on the amount spent on TV advertising.

## Dataset

The dataset used for this project is `tvmarketing.csv`, which contains the following columns:

- **TV**: The amount spent on TV advertising (in thousands of dollars).
- **Sales**: The resulting sales (in thousands of units).

### Sample Data
```
TV,Sales
230.1,22.1
44.5,10.4
17.2,9.3
151.5,18.5
...
```

## Machine Learning Models

The following machine learning models were implemented in this project:

1. **Linear Regression**: 
   - A linear regression model was created and optimized using gradient descent.
   - The model predicts sales based on the TV advertising budget.

2. **Decision Trees**:
   - A decision tree regressor was trained to capture non-linear relationships in the data.

3. **Random Forests**:
   - A random forest regressor was employed to improve prediction accuracy by aggregating multiple decision trees.

### Model Evaluation

The models were evaluated using the Root Mean Squared Error (RMSE) metric to compare their performance:

- **Linear Regression RMSE**: [Insert RMSE value]
- **Decision Tree RMSE**: [Insert RMSE value]
- **Random Forest RMSE**: [Insert RMSE value]

The models were ranked based on their RMSE values, with lower values indicating better performance.

## Implementation Details

### Libraries Used

- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical computations.
- `scikit-learn`: For implementing machine learning algorithms.
- `joblib` and `pickle`: For saving and loading trained models.

### Code Snippets

#### Loading the Dataset
```python
import pandas as pd

data = pd.read_csv("data/tvmarketing.csv")
```

#### Training the Linear Regression Model
```python
from sklearn.linear_model import LinearRegression

X = data[['TV']]
y = data['Sales']
lr_model = LinearRegression()
lr_model.fit(X, y)
```

#### Saving the Model
```python
import joblib
import pickle as pk

# Save using joblib
joblib.dump(lr_model, 'tv_sales_model.joblib')

# Save using pickle
with open('regressiontv.pkl', 'wb') as file:
    pk.dump(lr_model, file)
```

#### Loading the Model
```python
# Load with pickle
with open('tv_sales_prediction_model.pkl', 'rb') as file:
    loaded_model_pickle = pk.load(file)
```

## Conclusion

This project successfully demonstrates the use of machine learning techniques to predict TV sales based on advertising expenditures. The implemented models provide insights into how advertising budgets can influence sales outcomes, and further improvements can be explored through feature engineering and hyperparameter tuning.



# Fitness & Weight Prediction App

This Flask web application allows users to upload a CSV fitness dataset, select predictors and a target variable, choose an ML algorithm, train a model using a 70/30 train-test split, and view results.

## How to Run

1. Open this folder in VS Code or PyCharm.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python app.py
```

4. Open the browser link:

```text
http://127.0.0.1:5000
```

## Test Dataset

Use `sample_fitness_data.csv`.

Recommended first test:
- Algorithm: Linear Regression
- Predictors: Age, Height_cm, Calories_Per_Day, Steps_Per_Day, Workouts_Per_Week
- Target: Weight_kg

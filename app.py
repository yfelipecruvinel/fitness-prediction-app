from flask import Flask, render_template, request
import pandas as pd
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
PLOT_FOLDER = "static/plots"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)

uploaded_data = None
trained_model = None
trained_predictors = []
trained_target = None
trained_target_encoder = None


def add_bmi_columns(data):
    data = data.copy()
    if "bmi" not in data.columns and "height_cm" in data.columns:
        weight_col = "final_weight_kg" if "final_weight_kg" in data.columns else None
        if weight_col is None and "current_weight_kg" in data.columns:
            weight_col = "current_weight_kg"
        if weight_col:
            data["bmi"] = (data[weight_col] / ((data["height_cm"] / 100) ** 2)).round(1)
    if "weight_category" not in data.columns and "bmi" in data.columns:
        def categorize_bmi(bmi):
            if bmi < 18.5:
                return "Underweight"
            if bmi < 25:
                return "Healthy"
            if bmi < 30:
                return "Overweight"
            return "Obese"
        data["weight_category"] = data["bmi"].apply(categorize_bmi)
    return data


def is_classification_target(series):
    clean_series = series.dropna()
    if clean_series.dtype == "object" or str(clean_series.dtype).startswith("bool"):
        return True
    return clean_series.nunique() <= 10


def prepare_data(data, predictors, target):
    model_data = data[predictors + [target]].copy().dropna()
    encoders = {}
    for col in model_data.columns:
        if model_data[col].dtype == "object" or str(model_data[col].dtype).startswith("bool"):
            encoder = LabelEncoder()
            model_data[col] = encoder.fit_transform(model_data[col].astype(str))
            encoders[col] = encoder
    return model_data, model_data[predictors], model_data[target], encoders


def make_plots(data, target, predictors):
    plot_paths = []
    histogram_path = os.path.join(PLOT_FOLDER, "histogram.png")
    boxplot_path = os.path.join(PLOT_FOLDER, "boxplot.png")
    scatter_path = os.path.join(PLOT_FOLDER, "scatter.png")

    plt.figure(figsize=(7, 4))
    data[target].hist(bins=20)
    plt.title(f"Distribution of {target}")
    plt.xlabel(target)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(histogram_path, bbox_inches="tight")
    plt.close()
    plot_paths.append("static/plots/histogram.png")

    if pd.api.types.is_numeric_dtype(data[target]):
        plt.figure(figsize=(6, 4))
        data.boxplot(column=target)
        plt.title(f"Box Plot of {target}")
        plt.ylabel(target)
        plt.tight_layout()
        plt.savefig(boxplot_path, bbox_inches="tight")
        plt.close()
        plot_paths.append("static/plots/boxplot.png")

        numeric_predictors = [col for col in predictors if pd.api.types.is_numeric_dtype(data[col])]
        if numeric_predictors:
            x_col = numeric_predictors[0]
            plt.figure(figsize=(7, 4))
            plt.scatter(data[x_col], data[target], alpha=0.65)
            plt.title(f"{target} vs. {x_col}")
            plt.xlabel(x_col)
            plt.ylabel(target)
            plt.tight_layout()
            plt.savefig(scatter_path, bbox_inches="tight")
            plt.close()
            plot_paths.append("static/plots/scatter.png")
    return plot_paths


def regression_results(model_name, y_test, predictions, target):
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return {
        "model": model_name,
        "metric_1": f"Mean Squared Error: {mse:.2f}",
        "metric_2": f"R² Score: {r2:.2f}",
        "metric_3": "Regression model used for predicting a number.",
        "prediction_example": f"Example predicted {target}: {predictions[0]:.2f}"
    }


def classification_results(model_name, y_test, predictions, target, target_encoder=None):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
    recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
    example = predictions[0]
    if target_encoder is not None:
        example = target_encoder.inverse_transform([int(example)])[0]
    return {
        "model": model_name,
        "metric_1": f"Accuracy: {accuracy:.2f}",
        "metric_2": f"Precision: {precision:.2f}",
        "metric_3": f"Sensitivity/Recall: {recall:.2f}",
        "prediction_example": f"Example predicted {target}: {example}"
    }


@app.route("/", methods=["GET", "POST"])
def index():
    global uploaded_data, trained_model, trained_predictors, trained_target, trained_target_encoder
    columns = uploaded_data.columns.tolist() if uploaded_data is not None else []
    stats = uploaded_data.describe(include="all").to_html(classes="data-table") if uploaded_data is not None else None
    results = None
    plot_paths = []
    error = None
    input_prediction = None
    selected_predictors = []

    if request.method == "POST":
        action = request.form.get("action")
        file = request.files.get("csv_file")

        if file and file.filename != "":
            try:
                filepath = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(filepath)
                uploaded_data = add_bmi_columns(pd.read_csv(filepath))
                columns = uploaded_data.columns.tolist()
                stats = uploaded_data.describe(include="all").to_html(classes="data-table")
                trained_model = None
                trained_predictors = []
                trained_target = None
                trained_target_encoder = None
            except Exception as e:
                error = f"Error reading CSV file: {e}"

        elif action == "predict" and trained_model is not None:
            try:
                values = [float(request.form.get(f"input_{col}")) for col in trained_predictors]
                input_df = pd.DataFrame([values], columns=trained_predictors)
                prediction = trained_model.predict(input_df)[0]
                if trained_target_encoder is not None:
                    prediction = trained_target_encoder.inverse_transform([int(prediction)])[0]
                    input_prediction = f"Predicted {trained_target}: {prediction}"
                else:
                    input_prediction = f"Predicted {trained_target}: {prediction:.2f}"
                selected_predictors = trained_predictors
            except Exception as e:
                error = f"Error making prediction from your input: {e}"

        elif uploaded_data is not None:
            try:
                target = request.form.get("target")
                predictors = request.form.getlist("predictors")
                algorithm = request.form.get("algorithm")
                selected_predictors = predictors

                if not predictors or not target:
                    error = "Please select at least one predictor variable and one target variable."
                elif target in predictors:
                    error = "The target variable cannot also be selected as a predictor."
                else:
                    target_is_classification = is_classification_target(uploaded_data[target])
                    model_data, X, y, encoders = prepare_data(uploaded_data, predictors, target)
                    if len(model_data) < 10:
                        error = "Not enough complete rows after removing missing values. Try a dataset with more rows."
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                        target_encoder = encoders.get(target)

                        if algorithm == "Linear Regression":
                            if target_is_classification:
                                error = "Linear Regression needs a numeric target, like final_weight_kg or bmi."
                            else:
                                model = LinearRegression()
                                model.fit(X_train, y_train)
                                predictions = model.predict(X_test)
                                results = regression_results("Linear Regression", y_test, predictions, target)

                        elif algorithm == "Logistic Regression":
                            if not target_is_classification:
                                error = "Logistic Regression needs a categorical target, like weight_category."
                            else:
                                model = LogisticRegression(max_iter=3000)
                                model.fit(X_train, y_train)
                                predictions = model.predict(X_test)
                                results = classification_results("Logistic Regression Classification", y_test, predictions, target, target_encoder)

                        elif algorithm == "Decision Tree":
                            if target_is_classification:
                                model = DecisionTreeClassifier(random_state=42, max_depth=5)
                                model.fit(X_train, y_train)
                                predictions = model.predict(X_test)
                                results = classification_results("Decision Tree Classifier", y_test, predictions, target, target_encoder)
                            else:
                                model = DecisionTreeRegressor(random_state=42, max_depth=5)
                                model.fit(X_train, y_train)
                                predictions = model.predict(X_test)
                                results = regression_results("Decision Tree Regressor", y_test, predictions, target)

                        if results is not None:
                            trained_model = model
                            trained_predictors = predictors
                            trained_target = target
                            trained_target_encoder = target_encoder if target_is_classification else None
                            plot_paths = make_plots(model_data, target, predictors)
                            stats = uploaded_data.describe(include="all").to_html(classes="data-table")
            except Exception as e:
                error = f"Error training model: {e}"

    return render_template("index.html", columns=columns, results=results, stats=stats, plot_paths=plot_paths, error=error, input_prediction=input_prediction, trained_predictors=trained_predictors, trained_target=trained_target, selected_predictors=selected_predictors)


if __name__ == "__main__":
    app.run(debug=True)

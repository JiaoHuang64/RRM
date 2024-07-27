import string
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin
import joblib


def normalize_text(s):
    s = ''.join([i for i in s if not i.isdigit()])  # Remove numbers
    s = s.replace('-', ' ')
    s = s.replace('/', ' ')
    s = s.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    s = s.lower()  # Convert to lowercase
    s = s.replace('  ', ' ')  # Remove double spaces
    return s


# Load the data from four different files
file_paths = [
    '/Users/jiaohuangbixia/Downloads/1/MODELTRAIN/traindataA.csv',
    '/Users/jiaohuangbixia/Downloads/1/MODELTRAIN/traindataB.csv',
    '/Users/jiaohuangbixia/Downloads/1/MODELTRAIN/traindataC.csv',
    '/Users/jiaohuangbixia/Downloads/1/MODELTRAIN/traindataD.csv'
]

# Define weights for each dataset
weights = [1, 2, 5, 0.4]  # Adjust weights here, e.g., [1, 2, 0.5, 1.5]

print("Loading and preprocessing data from files...")
# Load and preprocess data from each file
all_data = []
for file_path, weight in zip(file_paths, weights):
    print(f"Processing file: {file_path} with weight: {weight}")
    data = pd.read_csv(file_path)
    data = data[pd.to_numeric(data['relevance'], errors='coerce').notnull()]
    data['relevance'] = data['relevance'].astype(float)
    data['text'] = [normalize_text(str(title) + ' ' + str(summary)) for title, summary in
                    zip(data['title'], data['summary'])]
    data['weight'] = weight  # Add weight to the data
    all_data.append(data)

print("Combining all datasets...")
# Combine all datasets
data = pd.concat(all_data, ignore_index=True)

print("Vectorizing text data...")
# Vectorize text data
vectorizer = HashingVectorizer(ngram_range=(1, 3))
X = vectorizer.fit_transform(data['text'])

# Extract target values and weights
y = data['relevance']
sample_weights = data['weight']

print("Splitting data into training and testing sets...")
# Split into training and testing sets
X_train, X_test, y_train, y_test, sample_weights_train, sample_weights_test = train_test_split(
    X, y, sample_weights, test_size=0.1, random_state=42)  # Reserve 10% as testing set


# Custom Ridge regression model with sample weights
class WeightedRidge(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        print(f"Initializing WeightedRidge with alpha: {alpha}")
        self.model = Ridge(alpha=self.alpha)

    def fit(self, X, y, sample_weight=None):
        print("Fitting the model...")
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        print("Making predictions...")
        return self.model.predict(X)

    def set_params(self, **params):
        if 'alpha' in params:
            self.alpha = params['alpha']
            self.model.set_params(alpha=self.alpha)
        return self

    def get_params(self, deep=True):
        return {'alpha': self.alpha}


# Define parameter grid for hyperparameter tuning
param_grid = {
    'alpha': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 5e-1, 5e-2, 5e-3]  # Expanded search range
}

print("Performing grid search for hyperparameter tuning...")
# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(WeightedRidge(), param_grid, cv=600, scoring='neg_mean_squared_error',
                           n_jobs=-1)  # Increase cross-validation folds and use multi-core processing
grid_search.fit(X_train, y_train, sample_weight=sample_weights_train)

# Output best parameters and score
best_params = grid_search.best_params_
best_score = -grid_search.best_score_
print(f"Best Params: {best_params}")
print(f"Best Score: {best_score}")

print("Using the best model for prediction on the test set...")
# Use the best model for prediction
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

print("Saving the trained model...")
# Save the trained model
model_path = '/Users/jiaohuangbixia/Downloads/1/MODELTRAIN/relevance_model5.pkl'
joblib.dump(best_model, model_path)


# Example code to use the trained model for new data prediction
def predict_new_data(new_titles, new_summaries):
    print("Loading the trained model for new data prediction...")
    model = joblib.load('/Users/jiaohuangbixia/Downloads/1/MODELTRAIN/relevance_model5.pkl')
    new_texts = [normalize_text(title + ' ' + summary) for title, summary in zip(new_titles, new_summaries)]
    new_vectors = vectorizer.transform(new_texts)
    new_predictions = model.predict(new_vectors)
    for title, prediction in zip(new_titles, new_predictions):
        print(f"Title: {title}")
        print(f"Predicted Relevance: {prediction}\n")


# Example usage
new_titles = ["Leveraging Machine Learning for High-Dimensional Option Pricing within the Uncertain Volatility Model"]
new_summaries = [
    "This paper explores the application of Machine Learning techniques for pricing high-dimensional options within the framework of the Uncertain Volatility Model (UVM). The UVM is a robust framework that accounts for the inherent unpredictability of market volatility by setting upper and lower bounds on volatility and the correlation among underlying assets. By leveraging historical data and extreme values of estimated volatilities and correlations, the model establishes a confidence interval for future volatility and correlations, thus providing a more realistic approach to option pricing. By integrating advanced Machine Learning algorithms, we aim to enhance the accuracy and efficiency of option pricing under the UVM, especially when the option price depends on a large number of variables, such as in basket or path-dependent options. Our approach evolves backward in time, dynamically selecting at each time step the most expensive volatility and correlation for each market state. Specifically, it identifies the particular values of volatility and correlation that maximize the expected option value at the next time step. This is achieved through the use of Gaussian Process regression, the computation of expectations via a single step of a multidimensional tree and the Sequential Quadratic Programming optimization algorithm. The numerical results demonstrate that the proposed approach can significantly improve the precision of option pricing and risk management strategies compared with methods already in the literature, particularly in high-dimensional contexts."
]
predict_new_data(new_titles, new_summaries)
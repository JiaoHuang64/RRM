import csv
import requests
import string
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
import joblib
from sklearn.linear_model import Ridge  # 假设 WeightedRidge 继承自 Ridge
from sklearn.base import BaseEstimator, RegressorMixin  # 确保正确导入
import re
from datetime import datetime, timedelta

# Define the custom class (or import it if it's in another module)
class WeightedRidge(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=self.alpha)

    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def set_params(self, **params):
        if 'alpha' in params:
            self.alpha = params['alpha']
            self.model.set_params(alpha=self.alpha)
        return self

    def get_params(self, deep=True):
        return {'alpha': self.alpha}

# Define the function to get the title and abstract from arXiv API
def get_arxiv_info(arxiv_id):
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    response = requests.get(url)
    if response.status_code == 200:
        xml = response.text
        title_start = xml.find('<title>') + len('<title>')
        title_end = xml.find('</title>', title_start)
        title = xml[title_start:title_end].strip()

        abstract_start = xml.find('<summary>') + len('<summary>')
        abstract_end = xml.find('</summary>', abstract_start)
        abstract = xml[abstract_start:abstract_end].strip()

        return title, abstract
    else:
        print(f"Error: Received status code {response.status_code} from arXiv API")
        return None, None

def normalize_text(s):
    s = ''.join([i for i in s if not i.isdigit()])  # Remove numbers
    s = s.replace('-', ' ')
    s = s.replace('/', ' ')
    s = s.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    s = s.lower()  # Convert to lowercase
    s = s.replace('  ', ' ')  # Remove double spaces
    return s

# Load the trained model
try:
    model = joblib.load('/Users/jiaohuangbixia/Downloads/1/MODELTRAIN/relevance_model5.pkl')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Initialize the vectorizer
vectorizer = HashingVectorizer(ngram_range=(1, 3))

def calculate_relevance(title, abstract):
    """Calculate the relevance of an article using a pre-trained model."""
    text = normalize_text(title + ' ' + abstract)
    features = vectorizer.transform([text])
    relevance = model.predict(features)[0]

    # 检查文章标题和摘要中是否包含"dark matter"字符（忽略大小写）
    if re.search(r'\bdark matter\b', text, re.IGNORECASE):
        relevance = 90 + 0.1 * relevance

    # 判断relevance是否大于100或小于0
    if relevance > 100:
        relevance = 100
    elif relevance < 0:
        relevance = 0

    return relevance

# List of arXiv IDs from provided text (from the images)
arxiv_ids = [
    # Group A (relevance: 70 or 90)
    "2407.15943v1", "2407.16373v1", "2407.16563v1", "2407.15419v1", "2407.15926v1",
    "2407.16202v1", "2404.08571v1", "2407.16156v1", "2407.16373v1", "2407.13405v1",

    # Group B (relevance: 0)
    "1604.00369v3", "1605.03133v1", "1705.05943v3", "1709.05272v1", "1711.08245v3",
    "1806.01924v1", "1806.05262v1", "1806.06105v1", "1807.03893v1", "1807.08404v3",

    # Group C (relevance: 20)
    "2403.01727v1", "2403.01556v1", "2403.01313v1", "2403.01239v2", "2403.01143v1",
    "2403.01052v3", "2403.01017v2", "2403.00983v1", "2403.00664v2", "2403.00657v1",

    # Group D (relevance: 100)
    "2407.02557v1", "2407.02573v1", "2407.02574v1", "2407.02593v1", "2407.02729v1",
    "2407.02872v1", "2407.02916v1", "2407.02954v1", "2407.02973v1", "2407.02991v1"
]

# Iterate over each arXiv ID, fetch the information, and predict relevance
for arxiv_id in arxiv_ids:
    title, abstract = get_arxiv_info(arxiv_id)
    if title and abstract:
        try:
            predicted_relevance = calculate_relevance(title, abstract)
            print(f"arXiv ID: {arxiv_id}")
            print(f"Title: {title}")
            print(f"Predicted Relevance: {predicted_relevance}\n")
        except Exception as e:
            print(f"Error predicting relevance for arXiv ID {arxiv_id}: {e}")
    else:
        print(f"Failed to retrieve information for arxiv_id: {arxiv_id}")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv("Life Expectancy Data.csv")

# Separate features and target
X = data.drop(columns=["Year"])
y = data["Year"]

# Define categorical and numerical features
categorical_features = ['Status', 'Country']
numerical_features = [col for col in X.columns if col not in categorical_features]

# Preprocessing pipeline
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# Define classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Bagging Classifier": BaggingClassifier(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
    "XGB Classifier": XGBClassifier()
}

# Define time periods or ranges for binning the years
year_bins = {2000: 0, 2001: 1, 2002: 2, 2003: 3, 2004: 4, 2005: 5,
             2006: 6, 2007: 7, 2008: 8, 2009: 9, 2010: 10, 2011: 11,
             2012: 12, 2013: 13, 2014: 14, 2015: 15}

# Map the 'Year' column to categorical classes
y = y.map(year_bins)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = {}

# Model training and evaluation
for name, classifier in classifiers.items():
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', classifier)])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    results[name] = {"accuracy": accuracy, "report": report, "confusion_matrix": confusion}

# Print results
for name, metrics in results.items():
    print(f"Classifier: {name}")
    print(f"Accuracy: {metrics['accuracy']}")
    # print(f"Classification Report:\n{metrics['report']}")
    # print(f"Confusion Matrix:\n{metrics['confusion_matrix']}\n")

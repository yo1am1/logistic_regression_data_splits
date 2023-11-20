import joblib
import pandas as pd
from icecream import ic
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# work with data
data = pd.read_csv("./data/kc_house_data.csv")

data = data.drop(["date"], axis=1)

data_normalized = preprocessing.normalize(data)
data_normalized = pd.DataFrame(data_normalized, columns=data.columns)

X = data_normalized[
    [
        "sqft_living",
        "sqft_lot",
        "sqft_above",
        "sqft_basement",
        "sqft_living15",
        "sqft_lot15",
        "zipcode",
    ]
]
y = data_normalized["price"]

# binarize y to make it a classification problem
y_binary = (y > y.median()).astype(int)

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.25, random_state=20
)

numeric_features = [
    "sqft_living",
    "sqft_lot",
    "sqft_above",
    "sqft_basement",
    "sqft_living15",
    "sqft_lot15",
]

categorical_features = ["zipcode"]

numeric_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

model = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression())]
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# evaluation
mse = mean_squared_error(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

joblib.dump(model, "./models/logistic_regression_model_scaled.joblib")

ic(accuracy, mse)

import pandas as pd
from icecream import ic
from sklearn import preprocessing
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

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
    ]
]
y = data_normalized["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=25
)

alphas = [0.1, 0.5, 1, 5, 10]

for alpha in alphas:
    # Ridge Regression
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train, y_train)
    ridge_pred = ridge_model.predict(X_test)
    ridge_mse = mean_squared_error(y_test, ridge_pred)
    ic(alpha, ridge_mse)

    # Lasso Regression
    lasso_model = Lasso(alpha=alpha)
    lasso_model.fit(X_train, y_train)
    lasso_pred = lasso_model.predict(X_test)
    lasso_mse = mean_squared_error(y_test, lasso_pred)
    ic(alpha, lasso_mse)

    # Elastic Net Regression
    elastic_net_model = ElasticNet(alpha=alpha, l1_ratio=0.5)
    elastic_net_model.fit(X_train, y_train)
    elastic_net_pred = elastic_net_model.predict(X_test)
    elastic_net_mse = mean_squared_error(y_test, elastic_net_pred)
    ic(alpha, elastic_net_mse)

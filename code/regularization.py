import numpy as np
import pandas as pd
from icecream import ic
from sklearn import preprocessing
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV

data = pd.read_csv("../data/kc_house_data.csv")

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
    X, y, test_size=0.2, random_state=10
)

param_grid = {
    "alpha": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10],
}

ic(param_grid)

# Ridge Regression
ridge_model = Ridge()
ridge_grid = GridSearchCV(ridge_model, param_grid)
ridge_grid.fit(X_train, y_train)
ridge_pred = ridge_grid.predict(X_test)
ridge_mse = mean_squared_error(y_test, ridge_pred)

ic("Ridge Best Parameters:", ridge_grid.best_params_, 'MSE', ridge_mse)

# Lasso Regression
lasso_model = Lasso()
lasso_grid = GridSearchCV(lasso_model, param_grid)
lasso_grid.fit(X_train, y_train)
lasso_pred = lasso_grid.predict(X_test)
lasso_mse = mean_squared_error(y_test, lasso_pred)

ic("Lasso Best Parameters:", lasso_grid.best_params_, "MSE:", lasso_mse)

# Elastic Net Regression
elastic_net_model = ElasticNet()
elastic_net_grid = GridSearchCV(elastic_net_model, param_grid)
elastic_net_grid.fit(X_train, y_train)
elastic_net_pred = elastic_net_grid.predict(X_test)
elastic_net_mse = mean_squared_error(y_test, elastic_net_pred)

ic(
    "Elastic Net Best Parameters:",
    elastic_net_grid.best_params_,
    "MSE:",
    elastic_net_mse,
)

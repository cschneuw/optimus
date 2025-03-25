import xgboost as xgb
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin

import time

# From the demo provided : https://xgboost.readthedocs.io/en/stable/python/examples/multioutput_regression.html#sphx-glr-python-examples-multioutput-regression-py


# As the experimental support status, custom objective doesn't support matrix as
# gradient and hessian, which will be changed in future release.
def gradient(predt, dtrain):
    """Compute the gradient squared error."""
    y = dtrain.get_label().reshape(predt.shape)
    return (predt - y).reshape(y.size)


def hessian(predt, dtrain):
    """Compute the hessian for squared error."""
    return np.ones(predt.shape).reshape(predt.size)


def squared_log(predt, dtrain):
    grad = gradient(predt, dtrain)
    hess = hessian(predt, dtrain)
    return grad, hess


def rmse(predt, dtrain):
    y = dtrain.get_label().reshape(predt.shape)
    v = np.sqrt(np.sum(np.power(y - predt, 2)))
    return "PyRMSE", v

def train_xgb(X_train, X_test, y_train, y_test):
    results = {}

    Xy_train = xgb.DMatrix(X_train, y_train)
    Xy_test = xgb.DMatrix(X_test, y_test)

    start = time.time()
    booster = xgb.train(
    {
        "tree_method": "hist",
        "num_target": y_train.shape[1],
        "max_depth": 3,
        "lambda": 1.5  

    },
    dtrain=Xy_train,
    num_boost_round=200,
    obj=squared_log,
    evals=[(Xy_train, "Train"), (Xy_test, "Test")],
    evals_result=results,
    custom_metric=rmse,
    )
    end = time.time()

    y_pred = booster.predict(xgb.DMatrix(X_test))

    print(f"Training time : {end-start} s.")
    
    return booster, results

class XGBoostRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, **xgb_params):
        """
        Initialize the XGBoostRegressor with XGBoost parameters.
        
        Parameters:
        - **xgb_params: Keyword arguments for XGBoost parameters (e.g., 'objective', 'max_depth', etc.)
        """
        self.xgb_params = xgb_params
        self.model = None

    def fit(self, X, y):
        """
        Fit the XGBoost model on the given training data.
        
        Parameters:
        - X: Training features, expected as a DataFrame or 2D array.
        - y: Target values, expected as a 1D array or Series.
        """
        if isinstance(X, pd.DataFrame):
            X = X.astype('float32')
        if isinstance(y, pd.DataFrame):
            y = y.astype('float32')

        # Convert data to DMatrix format, which is required for XGBoost
        dtrain = xgb.DMatrix(data=X, label=y, enable_categorical=True)
        
        # Train the model
        self.model = xgb.train(params=self.xgb_params, dtrain=dtrain)
        
        return self

    def predict(self, X):
        """
        Predict with the fitted XGBoost model.
        
        Parameters:
        - X: Test features, expected as a DataFrame or 2D array.
        
        Returns:
        - y_pred: Predicted values.
        """
        if isinstance(X, pd.DataFrame):
            X = X.astype('float32')
        # Convert test data to DMatrix format
        dtest = xgb.DMatrix(data=X, enable_categorical=True)
        
        # Predict using the trained model
        y_pred = self.model.predict(dtest)

        return y_pred
        
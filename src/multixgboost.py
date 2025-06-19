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
    def __init__(self,
                 num_boost_round=100,
                 max_depth=3,
                 lambda_=1.0,
                 learning_rate=0.1,
                 subsample=1.0,
                 colsample_bytree=1.0,
                 alpha=0.0,
                 tree_method="hist",
                 min_child_weight=1,
                 custom_obj=True,
                 custom_metric=True):
        
        # Expose all as attributes
        self.num_boost_round = num_boost_round
        self.max_depth = max_depth
        self.lambda_ = lambda_
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.alpha = alpha
        self.tree_method = tree_method
        self.min_child_weight = min_child_weight
        self.custom_obj = custom_obj
        self.custom_metric = custom_metric
        
        self.model = None

    def get_params(self, deep=True):
        # Return all parameters
        return {
            'num_boost_round': self.num_boost_round,
            'max_depth': self.max_depth,
            'lambda_': self.lambda_,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'alpha': self.alpha,
            'tree_method': self.tree_method,
            'custom_obj': self.custom_obj,
            'custom_metric': self.custom_metric
        }

    def set_params(self, **params):
        for key, val in params.items():
            setattr(self, key, val)
        return self

    def fit(self, X, y):
        dtrain = xgb.DMatrix(data=X, label=y, enable_categorical=True)
        params = {
            'max_depth': self.max_depth,
            'lambda': self.lambda_,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'alpha': self.alpha,
            'tree_method': self.tree_method,
            'num_target': y.shape[1]
        }
        obj_fn = squared_log if self.custom_obj else 'reg:squarederror'
        metric_fn = rmse if self.custom_metric else None

        self.model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=self.num_boost_round,
            obj=obj_fn,
            custom_metric=metric_fn
        )
        return self

    def predict(self, X):
        dtest = xgb.DMatrix(data=X, enable_categorical=True)
        return self.model.predict(dtest)

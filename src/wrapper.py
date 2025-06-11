import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from missforest import MissForest

from pytorch_tabular import TabularModel

from pytorch_tabnet.tab_model import TabNetRegressor

class MissForestWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        self.imputer = MissForest(**kwargs)

    def fit(self, X, y=None):
        return self.imputer.fit(X)

    def transform(self, X):
        return self.imputer.transform(X)

    def fit_transform(self, X, y=None):
        return self.imputer.fit_transform(X)

class TabNetModelWrapper:
    def __init__(self, n_d=8, n_a=8, max_epochs=250, patience=25):
        """
        Wrapper class for TabNetRegressor.

        Args:
            n_d (int): Dimension of the decision prediction layer.
            n_a (int): Dimension of the attention layer.
            max_epochs (int): Maximum number of training epochs.
            patience (int): Number of epochs to wait for improvement before stopping.
        """
        self.model = TabNetRegressor(n_d=n_d, n_a=n_a)
        self.max_epochs = max_epochs
        self.patience = patience

    def fit(self, X_train, y_train):
        """
        Fit the TabNetRegressor model.

        Args:
            X_train (np.array or pd.DataFrame): Training data features.
            y_train (np.array or pd.Series): Training data labels.
        """
        # Convert to NumPy arrays if the input is a DataFrame or Series
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values.copy().astype(np.float32)
        
        if isinstance(y_train, pd.Series) or isinstance(y_train, pd.DataFrame):
            y_train = y_train.values.copy().astype(np.float32)

        # Fit the model
        self.model.fit(
            X_train=X_train, y_train=y_train,
            max_epochs=self.max_epochs,
            patience=self.patience,
            num_workers=0,
            drop_last=False
        )

    def predict(self, X_test):
        """
        Make predictions with the trained model.

        Args:
            X_test (np.array or pd.DataFrame): Test data features.

        Returns:
            np.array: Predicted values.
        """
        # Convert to NumPy arrays if the input is a DataFrame
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values.copy().astype(np.float32)

        return self.model.predict(X_test)
    

# Utility to wrap TabularModel into sklearn-like fit/predict
class TabularModelWrapper:
    def __init__(self, model_config, data_config, trainer_config, optimizer_config):
        self.model_config = model_config
        self.data_config = data_config
        self.trainer_config = trainer_config
        self.optimizer_config = optimizer_config
        self.model = None

    def fit(self, X, y):
        df = X.copy().reset_index(drop=True)
        for c in y.columns:
            df[c] = y[c].reset_index(drop=True)

        self.model = TabularModel(
            data_config=self.data_config,
            model_config=self.model_config,
            optimizer_config=self.optimizer_config,
            trainer_config=self.trainer_config
        )
        self.model.fit(train=df, validation=df)  # ideally separate validation
        return self

    def predict(self, X):
        assert self.model is not None, "You must call .fit(...) before .predict(...)"
        preds = self.model.predict(X.reset_index(drop=True))
        return preds

import pandas as pd
import numpy as np

class StandardScaler:
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.var_ = np.var(X, axis=0)

    def transform(self, X):
        X_std = (X - self.mean_) / np.sqrt(self.var_)
        return X_std

    def fit_transform(self, X):
        self.fit(X)

        return self.transform(X)

    def inverse_transform(self, X_std):
        X = X_std * np.sqrt(self.var_) + self.mean_
        return X 
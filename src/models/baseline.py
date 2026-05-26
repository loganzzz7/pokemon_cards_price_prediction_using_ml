import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# used as moduel in train.py
class LinearRegressionModel:
    """Linear regression baseline model"""
    
    def __init__(self):
        self.model = LinearRegression()
    
    def fit(self, X_train, y_train):
        """Train model w split data"""
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X_test):
        """Run predict"""
        return self.model.predict(X_test)


class MovingAverageModel:
    """Moving average baseline model"""
    
    def __init__(self):
        self.mean_price = None
    
    def fit(self, X_train, y_train):
        """Calc mean price from training"""
        self.mean_price = np.mean(y_train)
        return self
    
    def predict(self, X_test):
        """Ret mean price for all"""
        return np.full(len(X_test), self.mean_price)

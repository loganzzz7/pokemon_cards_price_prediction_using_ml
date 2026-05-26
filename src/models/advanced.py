from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


class RandomForestModel:
    """
    Random forest advanced model
    
    100 trees w/ parallel processing n_jobs
    """
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, 
                                        random_state=random_state,
                                        n_jobs=-1)
    
    def fit(self, X_train, y_train):
        """Train model"""
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X_test):
        """Predict prices"""
        return self.model.predict(X_test)


class XGBoostModel:
    """
    XGBoost gradient boosting advanced model
    
    100 rounds w/ learning rate of 0.1
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, random_state=42):
        self.model = xgb.XGBRegressor(n_estimators=n_estimators,
                                    learning_rate=learning_rate,
                                    random_state=random_state,
                                    verbosity=0)
    
    def fit(self, X_train, y_train):
        """Train model"""
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, X_test):
        """Predict prices"""
        return self.model.predict(X_test)

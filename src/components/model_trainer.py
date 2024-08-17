import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor # type: ignore
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor # type: ignore

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("splitting train and test dataset")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'Linear Regression' : LinearRegression(),
                'K-Neighbours Regressor' : KNeighborsRegressor(),
                'Decision Tree' : DecisionTreeRegressor(),
                'Random Forest Regressor' : RandomForestRegressor(),
                'XGB Regressor' : XGBRegressor(),
                'Cat Boost Regressor' : CatBoostRegressor(verbose=False),
                'Ada Boost' : AdaBoostRegressor()
            }

            params = {
                "Decision Tree" : {
                    'criterion' : ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                "Random Forest Regressor" : {
                    'n_estimators' : [8, 16, 32, 64, 128, 256]
                },
                "XGB Regressor" : {
                    'learning_rate' : [0.1, 0.01, 0.05, 0.001],
                    'n_estimators' : [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression" : {},
                "K-Neighbours Regressor" : {
                    "n_neighbors" : [1, 3, 5, 7, 9, 11]
                },
                "Cat Boost Regressor" : {
                    'depth' : [6, 8, 10],
                    'learning_rate' : [0.01, 0.05, 0.1],
                    'iterations' : [30, 50, 100]
                },
                "Ada Boost" : {
                    'learning_rate' : [0.01, 0.05, 0.1],
                    'n_estimators' : [8, 16, 32, 64, 128, 256]
                }
            }

            model_report : dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info("Best model found on train and test")

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r_square = r2_score(y_test, predicted)

            return r_square

        except Exception as e:
            raise CustomException(e, sys)


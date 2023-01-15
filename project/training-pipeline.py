import hopsworks
import os
import joblib
import xgboost as xgb
import optuna
import pickle
from optuna.integration import XGBoostPruningCallback
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from hsml.schema import Schema
from hsml.model_schema import ModelSchema
from functions import data_encoder

project = hopsworks.login()

fs = project.get_feature_store() 
air_quality_fg = fs.get_or_create_feature_group(
    name = 'air_quality_fg',
    version = 3
)
weather_fg = fs.get_or_create_feature_group(
    name = 'weather_fg',
    version = 5
)


try:
    feature_view = fs.get_feature_view(name="air_data", version=4)
except:
    query = air_quality_fg.select(["date", "aqi"]).join(weather_fg.select_all(), on=["date"])
    feature_view = fs.create_feature_view(name="air_data",
        version=4,
        description="Joint feature view on air quality and weather data",
        query = query
    )

try:
    train_data = feature_view.get_training_data(1)[0]
except:
    version, job = feature_view.create_training_data(
    description='Training data for aqi model',
    data_format='csv',
    write_options={"wait_for_job": True}
    )
    train_data = feature_view.get_training_data(version)[0]

train_data = train_data.sort_values(by=["date"]).reset_index(drop=True)
train_data["aqi_next_day"] = train_data['aqi'].shift(1)
X = train_data.drop(columns=["aqi", "date"]).fillna(0)
y = X.pop("aqi_next_day")

X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size = 0.8, shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5)
    
X_train, X_val, X_test = data_encoder(X_train), data_encoder(X_val), data_encoder(X_test)

# optuna objective for hyperparameter-tuning
class Objective:
    def __init__(self, x_train, x_val, y_train, y_val):
        self.x_train = x_train
        self.x_val = x_val
        self.y_train = y_train
        self.y_val = y_val
    def __call__(self, trial):
        params = {
        "objective": 'reg:squarederror',
        "n_estimators": 1000,
        "eval_metric": "rmse",
        "early_stopping_rounds": 50,
        "eta": 0.05,
        "callbacks": [XGBoostPruningCallback(trial, "validation_0-rmse")],
        "booster": "gbtree",
        "lambda": trial.suggest_float("lambda", 1e-8, 10, log=True),
        "alpha": trial.suggest_float("alpha", 0.01, 10, log=True),
        "subsample": trial.suggest_float("subsample", 0.4, 0.8, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "gamma": trial.suggest_float("gamma", 1e-8, 10, log=True),
        }
        model = xgb.XGBRegressor(**params)
        model.fit(self.x_train, self.y_train,
            eval_set=[(self.x_val, self.y_val)]
        )
        preds = model.predict(self.x_val)
        rmse =  mean_squared_error(y_val, [int(elem) for elem in preds], squared=False)

        # Save a trained model to a file.
        with open(f"{trial.number}.pickle", "wb") as fout:
            pickle.dump(model, fout)
        
        return rmse

study = optuna.create_study(direction='minimize')
study.optimize(Objective(X_train, X_val, y_train, y_val), n_trials=100)

# Load the best model.
with open(f"{study.best_trial.number}.pickle", "rb") as fin:
    best_xgbr = pickle.load(fin)

final_mse = mean_squared_error(y_test, [int(elem) for elem in best_xgbr.predict(X_test)], squared=False)
print(f"The mean squared error on the test set is {final_mse}.")

mr = project.get_model_registry()
    
model_dir="aqi_model"
if os.path.isdir(model_dir) == False:
    os.mkdir(model_dir)

joblib.dump(best_xgbr, model_dir + "/aqi_model.pkl")

input_schema = Schema(X_train)
output_schema = Schema(y_train)
model_schema = ModelSchema(input_schema, output_schema)

aqi_model = mr.python.create_model(
    name="aqi_model", 
    metrics={"rmse" : final_mse},
    model_schema=model_schema,
    description="Air Quality Index Predictor"
)
aqi_model.save(model_dir)
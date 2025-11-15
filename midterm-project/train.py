import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load and prepare data
df = pd.read_csv("defects_data.csv")

df["defect_id"] = df["defect_id"].astype(str)
df["product_id"] = df["product_id"].astype(str)
df["date_converted"] = pd.to_datetime(df["defect_date"])
df["month"] = df["date_converted"].dt.month
df["day_of_week"] = df["date_converted"].dt.dayofweek

df.drop(["defect_id", "defect_date", "date_converted"], axis=1, inplace=True)


# Train, test, split
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

y_full_train = df_full_train.pop("repair_cost").values
y_test = df_test.pop("repair_cost").values


# Feature columns
categorical = ['product_id', 'defect_type', 'defect_location', 'severity', 'inspection_method']
numerical = ['month', 'day_of_week']


# Dict vectorizer
dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(df_full_train[categorical + numerical].to_dict(orient="records"))
X_test = dv.transform(df_test[categorical + numerical].to_dict(orient="records"))


# Regression error metrics
def rmse(y_pred, y_true):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))

def mae(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true))


# Train final model
final_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=1,
    random_state=1,
    n_jobs=-1
)

final_model.fit(X_full_train, y_full_train)


# Evaluate on test set
y_pred = final_model.predict(X_test)
print(f"Test MAE : {mae(y_pred, y_test):.4f}")
print(f"Test RMSE: {rmse(y_pred, y_test):.4f}")


# Save model
with open("model.bin", "wb") as f_out:
    pickle.dump((dv, final_model), f_out)
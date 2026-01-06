from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASES = ['A', 'C', 'G', 'T']

def print_validation(val_y, pred_y):
    print("MAE: ", mean_absolute_error(val_y, pred_y))
    print("MSE: ", mean_squared_error(val_y, pred_y))
    print("R2:", r2_score(val_y, pred_y))


def export_as_csv_and_pkl(df: pd.DataFrame, file_name: str, path: Path = Path.cwd()): 
    file_path = path / file_name
    df.to_csv(f"{file_path}.csv", index=False)
    df.to_pickle(f"{file_path}.pkl")
    print(f"Data saved to {file_path}.csv and {file_path}.pkl")

def export_model(model, file_name: str, path: Path = Path.cwd()):
    import joblib
    file_path = path / file_name
    joblib.dump(model, f"{file_path}.joblib")
    print(f"Model saved to {file_path}.joblib")

def load_model(file_name: str, path: Path = Path.cwd()):
    import joblib
    file_path = path / file_name
    model = joblib.load(f"{file_path}.joblib")
    print(f"Model loaded from {file_path}.joblib")
    return model

def drop_seq(df: pd.DataFrame) -> None:
    df.drop(columns=["23-nt_sequence"], inplace=True)


def load_training(df: pd.DataFrame, target: str = "normalized_efficacy"):
    X = df.drop(columns=[target])
    y = df[target]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0)
    return X, X_train, y, y_train

def load_testing(df: pd.DataFrame, target: str = "normalized_efficacy"):
    X = df.drop(columns=[target])
    y = df[target]
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
    return X, X_val, y, y_val
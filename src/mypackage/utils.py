from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# DATA RELEVANT

BASES = ['A', 'C', 'G', 'T']

def drop_seq(df: pd.DataFrame) -> None:
    df.drop(columns=["23-nt_sequence"], inplace=True)

def export_as_csv_and_pkl(df: pd.DataFrame, file_name: str, path: Path = Path.cwd()): 
    file_path = path / file_name
    df.to_csv(f"{file_path}.csv", index=False)
    df.to_pickle(f"{file_path}.pkl")
    print(f"Data saved to {file_path}.csv and {file_path}.pkl")

def onehot_encode_sequences(df: pd.DataFrame, seq_col="23-nt_sequence", ending: str = ""):
    sequences = df[seq_col].values
    n_sequences = len(sequences)
    seq_length = len(sequences[0])
    bases = ["A", "C", "G", "T"]

    # Create an empty array: shape (num_sequences, seq_length, 4)
    onehot = np.zeros((n_sequences, seq_length, 4), dtype=int)

    # Fill in the one-hot array
    for i, base in enumerate(bases):
        onehot[:, :, i] = (np.array([list(seq) for seq in sequences]) == base)

    # Flatten to columns
    col_names = [f"pos_{pos}_{base}{ending}" for pos in range(seq_length) for base in bases]
    onehot_flat = onehot.reshape(n_sequences, seq_length * 4)

    # Return a new DataFrame with one-hot columns
    #return pd.DataFrame(onehot_flat, columns=col_names, index=df.index).merge(df.drop(columns=[seq_col]), left_index=True, right_index=True)
    return df.drop(columns=[seq_col]).merge(pd.DataFrame(onehot_flat, columns=col_names, index=df.index), left_index=True, right_index=True)



## MODEL RELEVANT

def load_model(file_name: str, path: Path = Path.cwd()):
    import joblib
    file_path = path / file_name
    model = joblib.load(f"{file_path}.joblib")
    print(f"Model loaded from {file_path}.joblib")
    return model

def export_model(model, file_name: str, path: Path = Path.cwd()):
    import joblib
    file_path = path / file_name
    joblib.dump(model, f"{file_path}.joblib")
    print(f"Model saved to {file_path}.joblib")


def load_data(df: pd.DataFrame, target: str = "normalized_efficacy"):
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_val, y_train, y_val

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


def evaluate_model(val_y, pred_y):
    evaluation = {
        "R2": r2_score(val_y, pred_y),
        "MAE": mean_absolute_error(val_y, pred_y),
        "MSE": mean_squared_error(val_y, pred_y),
        "RMSE": np.sqrt(mean_squared_error(val_y, pred_y))

    }
    return evaluation

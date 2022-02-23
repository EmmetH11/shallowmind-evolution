import numpy as np
import pandas as pd
import utils

def load_data():
    # Read dataset
    filename = "heart.csv"
    return pd.read_csv(filename)

def prepare_data(df):
    dff = df.copy(True)
    dff["age"] = utils.normalize(df["age"])
    dff["cp"] = utils.normalize(df["cp"])
    dff["trestbps"] = utils.standardize_normal(df["trestbps"])
    dff["chol"] = utils.standardize_normal(df["chol"])
    dff["thalach"] = utils.standardize_normal(df["thalach"])
    dff["oldpeak"] = utils.normalize(df["oldpeak"])
    dff["slope"] = utils.normalize(df["slope"])
    dff["thal"] = utils.normalize(df["thal"])
    dff["ca"] = utils.normalize(df["ca"])
    split = dff.sample(frac=1)
    df_split = np.array_split(split, 2)

    training = df_split[0]
    training_labels = dff.iloc[:, -1].values
    training_inputs = dff.iloc[:, :-1].values
    
    val = df_split[1]
    val_labels = dff.iloc[:, -1].values
    val_inputs = dff.iloc[:, :-1].values

    return training_inputs, training_labels, val_inputs, val_labels
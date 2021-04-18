from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
import pandas as pd


def read_data(filepath):
    data_df = pd.read_csv(filepath).dropna()
    data_df['type'] = data_df['type'].replace({'white': 0, 'red': 1})
    return data_df


def scale(data, scaling_method='MinMax'):
    if scaling_method == 'MinMax':
        scaler = MinMaxScaler(feature_range=(0, 1))
    elif scaling_method == 'Standard':
        scaler = StandardScaler()
    elif scaling_method == 'Normal':
        scaler = Normalizer()
    else:
        scaler = None

    data = scaler.fit_transform(data)
    return data, scaler


def generate_dataset(data_df):
    inputs = data_df.loc[:, data_df.columns != 'quality']
    targets = data_df['quality']
    return inputs, targets


def generate_train_test_dataset(inputs, targets):
    return train_test_split(inputs, targets, test_size=0.1, random_state=42)


def inverse_scaling(data, scaler):
    return scaler.inverse_transform(data)

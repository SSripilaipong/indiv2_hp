from os import path
import pandas as pd

def load_data(
        csv_name: str,
        data_path: str = 'res/',
):
    df = pd.read_csv(path.join(data_path, csv_name), low_memory=False)

    return df

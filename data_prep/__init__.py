from typing import Union, List

import numpy as np
import pandas as pd

from sklearn.cluster import FeatureAgglomeration

from load_data import load_data


is_number = np.vectorize(lambda dtype: np.issubdtype(dtype, np.number))


def get_clusters(X: pd.DataFrame, n_clusters: int):
    clt = FeatureAgglomeration(n_clusters=n_clusters)
    clt.fit(X)

    clusters = []

    for i in range(n_clusters):
        clusters.append(X.columns[clt.labels_ == i].tolist())

    return clusters # type: list[str]


def drop_outliers(df: pd.DataFrame, std_range: float = 3.0, inplace: bool = False):
    mean = mean_maintain_cols(df)
    std = std_maintain_cols(df)
    numeric = to_numeric(df)

    index_list = numeric[
        ((numeric >= mean+std_range*std) | (numeric <= mean-std_range*std)).any(axis=1)
    ].index.tolist()

    return df.drop(index_list, inplace=inplace)


def get_cat_cols(df: pd.DataFrame, unrange: List[int] = [2, 30], addition: List[str] = []):
    usizes = uniques_count(df)
    columns = df.columns[((unrange[0] <= usizes) & (usizes < unrange[1]))].tolist() + addition

    others = list(set(df.columns)-set(columns))

    return columns, others


def drop_ids(df: pd.DataFrame, inplace: bool = False):
    column_list = [
        'customer_name',
        'Mobile1',
        'RWD_CARD_NO',
    ]

    return df.drop(column_list, axis=1, inplace=inplace)


def drop_dates(df: pd.DataFrame, excepts: set = None, inplace: bool = False):
    column_list = df.columns[df.apply(
        lambda rows: rows.str.contains('20[0-1][0-9]-[0-1][1-9]-[0-3][0-9]').any()
    )].tolist()

    if excepts:
        column_list = set(column_list) - excepts

    return df.drop(column_list, axis=1, inplace=inplace)


def drop_nan_cols(df: pd.DataFrame, thresh: int = 0.20, inplace: bool = False):
    return df.dropna(thresh=df.shape[0]*(1-thresh), axis=1, inplace=inplace)


def drop_flat(df: pd.DataFrame, thresh: int = 0.70, inplace: bool = False):
    t = mode_pc(df)
    return df.drop(t[t >= thresh].index.tolist(), axis=1, inplace=inplace)


def drop_auto(df: pd.DataFrame, drop_cols: bool = True, drop_idxs: bool = True, inplace: bool = False):
    drops = []

    if drop_cols:
        drops += [
            drop_ids,
            drop_dates,
            drop_nan_cols,
            drop_flat,
        ]

    if drop_idxs:
        drops += [
            drop_outliers
        ]

    for drop in drops:
        df = drop(df, inplace=inplace)
    print(df.shape)

    return df


def to_numeric(X: Union[pd.Series, pd.DataFrame]):
    if type(X) is pd.Series:
        return pd.to_numeric(X, errors='coerece')

    return X.apply(lambda rows: pd.to_numeric(rows, errors='coerece'))


def mean_maintain_cols(df: pd.DataFrame):
    if not is_number(df.dtypes).all():
        return df.apply(lambda rows: pd.to_numeric(rows, errors='coerece').mean())

    return df.apply(lambda rows: rows.mean())


def median_maintain_cols(df: pd.DataFrame):
    if not is_number(df.dtypes).all():
        return df.apply(lambda rows: pd.to_numeric(rows, errors='coerece').median())

    return df.apply(lambda rows: rows.median())

def mode_maintain_cols(df: pd.DataFrame):
    if not is_number(df.dtypes).all():
        return df.apply(lambda rows: pd.to_numeric(rows, errors='coerece').mode())

    return df.apply(lambda rows: rows.mode())


def std_maintain_cols(df: pd.DataFrame):
    if not is_number(df.dtypes).all():
        return df.apply(lambda rows: pd.to_numeric(rows, errors='coerece').std())

    return df.apply(lambda rows: rows.std())


def uniques_count(df: pd.DataFrame):
    return df.apply(lambda rows: rows.unique().shape[0])


def mode_pc(df: pd.DataFrame):
    return df.apply(lambda rows: max(rows.value_counts() / rows.count()))


def nan_pc(df: pd.DataFrame):
    return df.isnull().sum(axis=0) / df.shape[0]


def outlier_pc(df: pd.DataFrame, std_range: float = 3.0, ):
    mean = mean_maintain_cols(df)
    std = std_maintain_cols(df)
    numeric = to_numeric(df)

    return 1 - (~((numeric >= mean + std_range * std) | (numeric <= mean - std_range * std))).mean()


def test():
    df = load_data('expo_valid_data.csv', data_path='../res/').iloc[:10, :5]
    drop_ids(df, inplace=True)
    # drop_outliers(df)
    print(df > df.mean())


if __name__ == '__main__':
    test()

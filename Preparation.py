from typing import List, Union
from itertools import compress

import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.base import TransformerMixin, BaseEstimator

from load_data import load_data
import data_prep as dprep


class Preparation(TransformerMixin):
    def __init__(self, selected=None, *args):
        super(Preparation, self).__init__(*args)

        if selected:
            self.selected = selected

    def select_from_clusters(
            self,
            df: pd.DataFrame,
            y: Union[pd.DataFrame, pd.Series] = None,
            clusters: List[str] = None,
            fitting=False
    ):
        if fitting:
            self.selected = []

            for cluster in clusters:
                assert len(cluster) > 0

                if len(cluster) == 1:
                    self.selected.append(cluster[0])
                else:
                    selecter = SelectKBest(k=1)
                    selecter.fit(df.loc[:, cluster], y)

                    self.selected += list(compress(cluster, selecter.get_support()))

        return df.loc[:, self.selected]

    def fill_nan(self, df: pd.DataFrame, fitting=False):
        if fitting:
            self.cat_cols, self.oth_cols = dprep.get_cat_cols(df, [2, 30], 'sor_s sor_v'.split())
            for k in 'nearby_store_s_sor_fl sor_s sor_v'.split():
                self.cat_cols.remove(k)

        cats = df.loc[:, self.cat_cols]  # type: pd.DataFrame
        if fitting:
            self.cat_fill = dprep.mode_maintain_cols(cats).iloc[0]

        cats.fillna(self.cat_fill, inplace=True)

        cats = pd.get_dummies(cats, drop_first=True)  # type: pd.DataFrame
        if fitting:
            self.cat_dmm_cols = cats.columns  # type: List[str]

        oths = df.loc[:, self.oth_cols]  # type: pd.DataFrame
        if fitting:
            self.oth_fill = dprep.mean_maintain_cols(oths)

        oths.fillna(self.oth_fill, inplace=True)

        df_filled = pd.concat([oths, cats], axis=1)

        return df_filled # type: pd.DataFrame

    def fit_transform(self, df: pd.DataFrame, y: Union[pd.DataFrame, pd.Series], **fit_params):
        print('fit_transform')

        df = dprep.drop_auto(df, drop_cols=True, drop_idxs=False)

        df_filled = self.fill_nan(df, fitting=True)
        print('filled')
        print('isnull', (dprep.nan_pc(df_filled) == 1).any())
        clusters = dprep.get_clusters(df_filled, n_clusters=30)

        filtered = self.select_from_clusters(df_filled, y, clusters, fitting=True)

        return filtered.values

    def transform(self, df: pd.DataFrame, *args):
        print('transform')

        df_filled = self.fill_nan(df)

        filtered = self.select_from_clusters(df_filled)

        return filtered.values


if __name__ == '__main__':
    df = load_data('expo_train_data.csv')

    prep = Preparation()
    print(prep.fit_transform(df.drop('response_flag', axis=1), df.loc[:, ['response_flag']]))
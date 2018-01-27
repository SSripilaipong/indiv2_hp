import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from load_data import load_data
from Preparation import Preparation


def main():
    df = load_data('expo_train_data.csv')

    prep = Preparation()
    clf = RandomForestClassifier()

    pipeline = Pipeline(steps=[
        ('prep', prep),
        ('clf', clf),
    ])
    X = df.drop('response_flag', axis=1)
    y = df[['response_flag']]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30
    )

    pipeline.fit_transform(X_train, y_train)


if __name__ == '__main__':
    main()

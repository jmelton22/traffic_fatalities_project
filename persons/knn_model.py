#!/usr/bin/env python3

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import utils


def main():
    path = '../data/persons'
    data = pd.read_csv(f'{path}/person_data_clean.csv', header=0)

    cat_cols = ['person_type', 'trafficway_type', 'manner_of_collision', 'body_type', 'seating_position',
                'ejection', 'safety_equipment_use']
    binary_cols = ['sex', 'land_use_urban', 'rollover', 'air_bag_deployed']
    numeric_cols = ['age']

    data[cat_cols] = data[cat_cols].apply(lambda x: x.astype('category'))

    labels = data['fatality']
    features = data[cat_cols + binary_cols + numeric_cols]
    feature_names = features.columns

    # oe = OrdinalEncoder()
    # features = oe.fit_transform(features)

    features = pd.get_dummies(features, columns=cat_cols)

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                        test_size=0.2, random_state=2020)
    print('Class Balance')
    print(y_test.value_counts())
    print()

    model = GridSearchCV(estimator=KNeighborsClassifier(),
                         param_grid={'n_neighbors': range(1, 20, 2)},
                         cv=5, scoring='f1')
    model.fit(X_train, y_train)
    print(model.best_params_)
    print()

    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    utils.print_metrics(y_test, y_pred)
    utils.roc_curve(y_test, y_probs, 'KNN', 'knn')


if __name__ == '__main__':
    main()

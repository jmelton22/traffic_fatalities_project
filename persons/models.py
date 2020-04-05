#!/usr/bin/env python3

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
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

    # features = pd.get_dummies(features, columns=cat_cols)
    # features.rename(columns={'manner_of_collision_Not Collision with Motor Vehicle in Transport (Not Necessarily in Transport for\n2005-2009)': 'manner_of_collision_Not Collision with Motor Vehicle in Transport'},
    #                 inplace=True)
    feature_names = features.columns

    oe = OrdinalEncoder()
    features = oe.fit_transform(features)

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                        test_size=0.2, random_state=2020)
    print('Class Balance')
    print(y_test.value_counts())
    print()

    models = {
        'Random Forest': (RandomForestClassifier(n_estimators=100,
                                                 min_samples_leaf=5,
                                                 class_weight='balanced',
                                                 random_state=2020),
                          'rf'),
        'Logistic Regression': (LogisticRegressionCV(cv=5, scoring='f1',
                                                     class_weight='balanced',
                                                     max_iter=500,
                                                     random_state=2020),
                                'lr')
    }

    for name, (model, suffix) in models.items():
        print(name)
        print('-' * 20)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1]

        utils.print_metrics(y_test, y_pred)
        utils.roc_curve(y_test, y_probs, name, suffix)
        utils.feature_importance(model, feature_names, name, suffix)
        utils.permutation_importances(model, X_test, y_test, feature_names, name, suffix)
        # utils.permutation_importances(model, X_train, y_train, feature_names, name, suffix + '_ohe', dataset='train')
        print('#' * 50)


if __name__ == '__main__':
    main()

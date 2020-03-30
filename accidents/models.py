#!/usr/bin/env python3

import utils
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV


def main():
    path = '../data/accidents'
    data = pd.read_csv(f'{path}/accident_data_clean_balanced.csv', header=0)

    # Feature columns
    cat_cols = ['month', 'roadway_type', 'intersection', 'light_condition', 'atmospheric_conditions',
                'manner_of_collision', 'body_type', 'vehicle_conditions', 'part_of_day']
    binary_cols = ['land_use_urban', 'national_highway_system', 'previous_dwi_convictions',
                   'previous_speeding_convictions', 'speeding_related', 'driver_vision_obscured', 'is_weekend',
                   'multiple_vehicles', 'nonmotorist_involved', 'multiple_motorists', 'drunk_driver_involved']
    numeric_cols = ['vehicle_year', 'speed_limit']

    data[cat_cols] = data[cat_cols].apply(lambda x: x.astype('category'))

    labels = data['multiple_fatalities']
    features = data[cat_cols + binary_cols + numeric_cols]
    feature_names = features.columns

    # features = pd.get_dummies(features, columns=cat_cols, drop_first=True)

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
                                                 random_state=2020),
                          'rf'),
        'Logistic Regression': (LogisticRegressionCV(cv=5, scoring='f1',
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
        print('#' * 50)


if __name__ == '__main__':
    main()

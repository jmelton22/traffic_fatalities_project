#!/usr/bin/env python3

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.feature_selection as fs
from sklearn.linear_model import LinearRegression, RidgeCV, ElasticNetCV
from sklearn.svm import LinearSVR
import utils

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def main():
    path = '../data/states'
    data = pd.read_csv(f'{path}/state_mean_accident_data.csv', header=0, index_col='state_name')

    non_feature_cols = ['state_number', 'state_code', 'accidents', 'fatalities', 'fatalities_per_accident', 'accidents_per_100k', 'num_vehicles',
                        'hour_of_day', 'num_fatalities', 'num_drunk_drivers']

    labels = data['accidents_per_100k']
    features = data.drop(non_feature_cols, axis=1)
    feature_names = features.columns

    scores_df = features.corrwith(labels, axis=0,
                                  method='pearson').to_frame('r_coef')
    scores_df['f_score'], scores_df['p_value'] = fs.f_regression(features, labels)

    # Sort by p value
    scores_df.sort_values('p_value', inplace=True)
    print(scores_df)

    # feature_subset_cols = scores_df[scores_df['p_value'] < 0.05].index.tolist()
    # features = features[feature_subset_cols]

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # X_train, X_test, y_train, y_test = train_test_split(features, labels,
    #                                                     test_size=0.2, random_state=2020)

    models = {
        'Linear Regression': (LinearRegression(),
                              'linreg'),
        'Ridge': (RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0],
                          cv=5, scoring='neg_mean_squared_error'),
                  'ridge'),
        'Elastic Net': (ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
                                     alphas=[0.01, 0.1, 1.0, 10.0],
                                     max_iter=3000,
                                     cv=5),
                        'elastic_net'),
        'Linear SVR': (LinearSVR(),
                       'svr')
    }

    for name, (model, suffix) in models.items():
        print(name)
        print('-' * 20)
        model.fit(features, labels)

        y_pred = model.predict(features)

        utils.print_regression_metrics(labels, y_pred)

        utils.hist_resids(labels, y_pred, name, suffix)
        utils.resid_qq(labels, y_pred, name, suffix)
        utils.resid_plot(labels, y_pred, name, suffix)

        utils.feature_importance_regression(model, feature_names, name, suffix)
        utils.permutation_importances(model, features, labels, feature_names, name, suffix)
        print('#' * 50)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import sys
import math
import numpy as np
import scipy.stats as ss
from sklearn import metrics
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def print_metrics(labels, preds):
    """
        Prints confusion matrix and metrics scores for a binary classification
    """
    scores = metrics.precision_recall_fscore_support(labels, preds)
    conf = metrics.confusion_matrix(labels, preds)
    print(' ' * 4 + 'Confusion Matrix')
    print(' ' * 17 + 'Predict Positive    Predict Negative')
    print('Actual Positive         {}                 {}'.format(conf[1, 1], conf[1, 0]))
    print('Actual Negative         {}                 {}'.format(conf[0, 1], conf[0, 0]))
    print()
    print('Accuracy: {:.3f}'.format(metrics.accuracy_score(labels, preds)))
    print()
    print(' ' * 4 + 'Classification Report')
    print(' ' * 11 + 'Positive    Negative')
    print('Num cases    {}      {}'.format(scores[3][1], scores[3][0]))
    print('Precision    {:.2f}       {:.2f}'.format(scores[0][1], scores[0][0]))
    print('Recall       {:.2f}       {:.2f}'.format(scores[1][1], scores[1][0]))
    print('F1 Score     {:.2f}       {:.2f}'.format(scores[2][1], scores[2][0]))


def print_regression_metrics(y_true, y_preds):
    print('Mean Square Error      = {:.3f}'.format(metrics.mean_squared_error(y_true, y_preds)))
    print('Root Mean Square Error = {:.3f}'.format(math.sqrt(metrics.mean_squared_error(y_true, y_preds))))
    print('Mean Absolute Error    = {:.3f}'.format(metrics.mean_absolute_error(y_true, y_preds)))
    print('Median Absolute Error  = {:.3f}'.format(metrics.median_absolute_error(y_true, y_preds)))
    print('R^2                    = {:.3f}'.format(metrics.r2_score(y_true, y_preds)))


def roc_curve(labels, probs, name, suffix):
    fpr, tpr, thresholds = metrics.roc_curve(labels, probs)
    roc_auc = metrics.auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.rcParams['font.size'] = 12
    plt.title(f'ROC Curve for {name} Classifier')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.legend(loc='lower right')
    plt.show()
    fig.savefig(f'visualizations/roc_curve_{suffix}.png')


def feature_importance(model, feature_names, name, suffix, n_features=10):
    try:
        feature_importances = model.feature_importances_
        sorted_idx = feature_importances.argsort()[-n_features:]
    except AttributeError:
        try:
            feature_importances = model.coef_[0]
            sorted_idx = abs(feature_importances).argsort()[:n_features]
        except AttributeError as ae:
            print(ae)
            sys.exit(1)

    y_ticks = np.arange(0, n_features)
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.barh(y_ticks, feature_importances[sorted_idx])
    ax.set_yticklabels(feature_names[sorted_idx])
    ax.set_yticks(y_ticks)
    ax.set_title(f'{name} Feature Importances')
    plt.show()
    fig.savefig(f'visualizations/feature_importances_{suffix}.png')


def feature_importance_regression(model, feature_names, name, suffix, n_features=10):
    feature_importances = model.coef_
    sorted_idx = abs(feature_importances).argsort()[-n_features:]

    y_ticks = np.arange(0, n_features)
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.barh(y_ticks, feature_importances[sorted_idx])
    ax.set_yticklabels(feature_names[sorted_idx])
    ax.set_yticks(y_ticks)
    ax.set_title(f'{name} Feature Importances')
    plt.show()
    fig.savefig(f'visualizations/feature_importances_{suffix}.png')


def permutation_importances(model, X, y, feature_names, name, suffix, dataset='test', n_features=10):
    result = permutation_importance(model, X, y, n_repeats=10, random_state=2020)
    sorted_idx = result.importances_mean.argsort()[-n_features:]

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.boxplot(result.importances[sorted_idx].T,
               vert=False, labels=feature_names[sorted_idx])
    ax.set_title(f'{name} Permutation Importances ({dataset} set)')
    plt.show()
    fig.savefig(f'visualizations/permutation_importances_{suffix}_{dataset}.png')


def hist_resids(y_true, y_preds, name, suffix):
    # Compute vector of residuals
    resids = np.subtract(y_true, y_preds)
    # Make residuals histogram
    sns.distplot(resids)
    plt.title(f'{name}: Histogram of residuals')
    plt.xlabel('Residual value')
    plt.ylabel('Count')
    plt.show()
    plt.savefig(f'visualizations/residuals_hist_{suffix}.png')


def resid_qq(y_true, y_preds, name, suffix):
    # Compute vector of residuals
    resids = np.subtract(y_true, y_preds)
    # Make residuals quantile-quantile plot
    ss.probplot(resids, plot=plt)
    plt.title(f'{name}: Residuals vs. Predicted values')
    plt.xlabel('Predicted value')
    plt.ylabel('Residual')
    plt.show()
    plt.savefig(f'visualizations/residuals_qq_{suffix}.png')


def resid_plot(y_true, y_preds, name, suffix):
    # Compute vector of residuals
    resids = np.subtract(y_true, y_preds)
    # Make residuals scatter plot
    sns.regplot(y_preds, resids, fit_reg=False)
    plt.title(f'{name}: Residuals vs. Predicted values')
    plt.xlabel('Predicted value')
    plt.ylabel('Residual')
    plt.show()
    plt.savefig(f'visualizations/residuals_scatter_{suffix}.png')

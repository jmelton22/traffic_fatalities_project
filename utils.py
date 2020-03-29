#!/usr/bin/env python3

from sklearn import metrics
import matplotlib.pyplot as plt


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


def roc_curve(labels, probs, name, suffix):
    fpr, tpr, thresholds = metrics.roc_curve(labels, probs)
    roc_auc = metrics.auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.rcParams['font.size'] = 12
    plt.title(f'ROC Curve for Multiple Fatality {name} Classifier')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.legend(loc='lower right')
    plt.show()
    fig.savefig(f'visualizations/roc_curve_{suffix}.png')


def feature_importance():
    pass


def permutation_importance():
    pass

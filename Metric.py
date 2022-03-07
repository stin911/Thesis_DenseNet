import matplotlib.pyplot as plt


def plot_roc_curve(fpr, tpr):
    """
    :param fpr: False positive rate
    :param tpr: True positive rate
    :return:
    """
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


# calculate true/negative occurrences of each classes
def calculate_rate(pred, label):
    """

    :param pred: list of predicticted values
    :param label: list of target values
    :return:True_pos, False_pos, True_neg, False_neg
    """
    True_pos = 0
    False_pos = 0
    True_neg = 0
    False_neg = 0
    for i in range(len(label)):
        if label[i] == 1:
            if pred[i] == 1:
                True_pos += 1
            elif pred[i] == 0:
                False_neg += 1
        elif label[i] == 0:
            if pred[i] == 1:
                False_pos += 1
            elif pred[i] == 0:
                True_neg += 1
    print(True_pos, False_pos, True_neg, False_neg)
    return True_pos, False_pos, True_neg, False_neg


def PPV_NPV(pred, label):
    True_pos, False_pos, True_neg, False_neg = calculate_rate(pred, label)
    PPV = True_pos / (True_pos + False_pos)
    NPV = True_neg / (True_neg + False_neg)
    print('ppv',PPV,'npv', NPV)


# calculate rate
def roc_auc(pred, label):
    """

    :param pred: list of predicticted values
    :param label: list of target values
    :return:
    """
    True_pos, False_pos, True_neg, False_neg = calculate_rate(pred, label)
    if True_pos == 0 and False_neg == 0:
        TPR = 0
    else:
        TPR = True_pos / (True_pos + False_neg)
    if True_neg == 0 and False_pos == 0:
        specificity = 0
    else:
        specificity = True_neg / (True_neg + False_pos)
    FPR = 1 - specificity
    print("TPR", TPR, "FPR", FPR, "specificity ", specificity)
    return TPR, FPR, specificity

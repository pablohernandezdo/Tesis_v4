import os
import numpy as np
import pandas as pd


def get_results(outputs_csv, dset, beta, n_thresholds=500):

    os.makedirs(f'Results/{dset}', exist_ok=True)

    df = pd.read_csv(outputs_csv)
    thresholds = np.linspace(0, 1, n_thresholds + 1)[:-1]
    model_name = outputs_csv.split('/')[-1].split('.')[0]

    # Preallocate variables
    acc = np.zeros(len(thresholds))
    prec = np.zeros(len(thresholds))
    rec = np.zeros(len(thresholds))
    fpr = np.zeros(len(thresholds))
    fscore = np.zeros(len(thresholds))

    tps =  np.zeros(len(thresholds))
    fps =  np.zeros(len(thresholds))
    fns =  np.zeros(len(thresholds))
    tns =  np.zeros(len(thresholds))

    for i, thr in enumerate(thresholds):
        predicted = (df['out'] > thr)
        tp = sum(predicted & df['label'])
        fp = sum(predicted & ~df['label'])
        fn = sum(~predicted & df['label'])
        tn = sum(~predicted & ~df['label'])

        tps[i], fps[i], fns[i], tns[i] = tp, fp, fn, tn

        # Evaluation metrics
        acc[i], prec[i], rec[i], fpr[i], fscore[i] = get_metrics(tp,
                                                                 fp,
                                                                 tn,
                                                                 fn,
                                                                 beta)
    res_dframe = pd.DataFrame({'thresholds': thresholds,
                               'tps': tps,
                               'fps': fps,
                               'fns': fns,
                               'tns': tns,
                               'acc': acc,
                               'prec': prec,
                               'rec': rec,
                               'fpr': fpr,
                               'fscore': fscore})

    res_dframe.to_csv(f'Results/{dset}/{model_name}.csv')


def get_metrics(tp, fp, tn, fn, beta):
    acc = (tp + tn) / (tp + fp + tn + fn)

    # Evaluation metrics
    if (not tp) and (not fp):
        precision = 1
    else:
        precision = tp / (tp + fp)

    if (not tp) and (not fn):
        recall = 0
    else:
        recall = tp / (tp + fn)

    if (not fp) and (not tn):
        fpr = 0
    else:
        fpr = fp / (fp + tn)

    if (not precision) and (not recall):
        fscore = 0
    else:
        fscore = (1 + beta ** 2) * (precision * recall) / \
                 ((beta ** 2) * precision + recall)

    return acc, precision, recall, fpr, fscore

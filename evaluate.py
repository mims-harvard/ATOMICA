#!/usr/bin/python
# -*- coding:utf-8 -*-
import json
import argparse
from functools import partial

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc
from sklearn.linear_model import LinearRegression


def rmse(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(((y_pred - y_true) ** 2).mean())


def mae(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.abs(y_true - y_pred).mean()


def minimized_rmse(y_true, y_pred):
    # from https://github.com/luost26/RDE-PPI/blob/main/rde/utils/skempi.py#L135
    y_true, y_pred = np.array(y_true), np.array(y_pred)[:, None]
    reg = LinearRegression().fit(y_pred, y_true)
    pred_corrected = reg.predict(y_pred)
    return rmse(y_true, pred_corrected)


def minimized_mae(y_true, y_pred):
    # from https://github.com/luost26/RDE-PPI/blob/main/rde/utils/skempi.py#L135
    y_true, y_pred = np.array(y_true), np.array(y_pred)[:, None]
    reg = LinearRegression().fit(y_pred, y_true)
    pred_corrected = reg.predict(y_pred)
    return mae(y_true, pred_corrected)


def continuous_auroc(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true = y_true > 0
    return roc_auc_score(y_true, y_pred)

def auprc(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)

def delta_auprc(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    auprc = auc(recall, precision)
    return auprc - np.mean(y_true)

def perstruct_corr(y_true, y_pred, struct_ids, _type='pearson'):
    assert len(y_pred) == len(struct_ids)
    results = {}
    for i, _id in enumerate(struct_ids):
        if _id not in results:
            results[_id] = [[], []]
        results[_id][0].append(y_true[i])
        results[_id][1].append(y_pred[i])

    corr = pearsonr if _type == 'pearson' else spearmanr
    valid_ids = []
    for _id in results:
        if len(results[_id][0]) < 10:  # from https://github.com/luost26/RDE-PPI/blob/main/rde/utils/skempi.py
            continue
        results[_id] = corr(results[_id][0], results[_id][1]).statistic
        valid_ids.append(_id)

    return np.mean([results[_id] for _id in valid_ids])


def parse():
    parser = argparse.ArgumentParser(description='Calculate evaluation metrics')
    parser.add_argument('--task')
    parser.add_argument('--predictions', type=str, required=True, help='Path to the predicted results')
    parser.add_argument('--reference', type=str, default=None, help='Path to the reference dataset')
    return parser.parse_args()


def main(args):
    with open(args.predictions, 'r') as fin:
        preds = [json.loads(s) for s in fin.readlines()]
        task = preds[0]['task']
    if args.reference is not None:
        with open(args.reference, 'r') as fin:
            test_set = [json.loads(s) for s in fin.readlines()]
        test_set = { item['id']: item['affinity']['neglog_aff'] for item in test_set }
        cover_ids = [ item['id'] for item in preds if item['id'] in test_set ]
    else:
        test_set = { item['id']: item['gt'] for item in preds }
        cover_ids = [_id for _id in test_set]
    
    preds = { item['id']: item['label'] for item in preds }
    print(f'prediction: {len(preds)}, test set: {len(test_set)}')
    print(f'Number of entries in both sets: {len(cover_ids)}')

    if task == 'PPA' or task == 'PLA' or task == 'PDBBind' or task == 'PLA_frag' or task == 'PLA_noisy_nodes':
        metrics = {
            'Pearson': pearsonr,
            'Spearman': spearmanr,
            'RMSE': rmse,
            'MAE': mae
        }
        y_pred = [ preds[_id] for _id in cover_ids ]
    elif task == 'NL' or task == 'PN' or task == 'DDG':
        metrics = {
            'Pearson': pearsonr,
            'Spearman': spearmanr,
            'RMSE': rmse,
            'MAE': mae,
            'min_RMSE': minimized_rmse
        }
        y_pred = [ preds[_id] for _id in cover_ids ]
    elif task == 'LEP'or task == 'MSP':
        metrics = {
            'AUROC': roc_auc_score,
            'Ave. Precision Recall': average_precision_score,
            'AUPRC': auprc,
            'Delta AUPRC': delta_auprc,
        }
        y_pred = [ preds[_id] for _id in cover_ids ]  # probability of label == 1
    else:
        raise NotImplementedError(f'Evaluation for task {task} not implemented')

    y_true = [ test_set[_id] for _id in cover_ids ]

    results = {}
    for name in metrics:
        func = metrics[name]
        results[name] = func(y_true, y_pred)
        print(f'{name}: {results[name]}')

    return results




if __name__ == '__main__':
    main(parse())

    # # test
    # y_pred = [-5, -10, -4, -8, -20]
    # y_label = [-4.5, -8.3, -3.2, -5, -14]
    # corr, p = pearsonr(y_label, y_pred)
    # print(f'pearson: {corr}, {p}')
    # corr, p = spearmanr(y_label, y_pred)
    # print(f'spearman: {corr}, {p}')
    # print(f'rmse: {rmse(y_label, y_pred)}')
    # print(f'mae: {mae(y_label, y_pred)}')
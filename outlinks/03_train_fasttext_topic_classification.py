# Train fastText model
import argparse
import csv
import multiprocessing
import os
import sys
import time

csv.field_size_limit(sys.maxsize)

import fasttext
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, average_precision_score

VERBOSITY = 2
WORDNGRAMS = 1
MINN = 0
MAXN = 0
MAXTHREADS = min(16, multiprocessing.cpu_count() - 1)
LOSS = 'ova'

def ft_to_toplevel(fasttext_lbl):
    return fasttext_lbl.replace('__label__','').split('.')[0]

def grid_search(train_fn, val_fn, learning_rates, minCounts, epochs, ws, wvs, ndims):
    best_lr = None
    best_ndim = None
    best_minCount = None
    best_epochs = None
    best_ws = None
    best_wv = None
    highest_f1 = float("-inf")
    label_counts_val = {}
    with open(val_fn) as fin:
        for line in fin:
            lbls = [l for l in line.strip().split(" ") if l.startswith('__label__')]
            for lbl in lbls:
                label_counts_val[lbl] = label_counts_val.get(lbl, 0) + 1
    label_counts_train = {}
    with open(train_fn) as fin:
        for line in fin:
            lbls = [l for l in line.strip().split(" ") if l.startswith('__label__')]
            for lbl in lbls:
                label_counts_train[lbl] = label_counts_train.get(lbl, 0) + 1
    grid_search_results = []
    wv = wvs
    for lr in learning_rates:
            for minCount in minCounts:
                for epoch in epochs:
                    for w in ws:
                        for i in range(0, len(ndims)):
                            ndim = ndims[i]
                            print("Building fasttext model: {0} lr; {1} dim; {2} min count; {3} epochs. {4} ws. wv: {5}.".format(
                                lr, ndim, minCount, epoch, w, wv))
                            # train model
                            model = fasttext.train_supervised(input=train_fn,
                                                              minCount=minCount,
                                                              wordNgrams=WORDNGRAMS,
                                                              pretrainedVectors=wv,
                                                              lr=lr,
                                                              epoch=epoch,
                                                              dim=ndim,
                                                              ws=w,
                                                              minn=MINN,
                                                              maxn=MAXN,
                                                              thread=MAXTHREADS,
                                                              loss=LOSS,
                                                              verbose=VERBOSITY)
                            # val
                            results_by_lbl = model.test_label(val_fn, threshold=0.5, k=-1)
                            f1_scores, support = zip(*[(res['f1score'], label_counts_val[lbl]) for lbl, res in results_by_lbl.items() if lbl in label_counts_val])
                            macro_f1 = np.average(f1_scores)
                            micro_f1 = np.average(f1_scores, weights=support)
                            f1_avg = np.average([micro_f1, macro_f1])
                            if f1_avg > highest_f1:
                                best_lr = lr
                                best_ndim = ndim
                                best_minCount = minCount
                                best_epochs = epoch
                                best_ws = w
                                best_wv = wv
                                highest_f1 = f1_avg

                            # train (check overfitting)
                            results_by_lbl = model.test_label(train_fn, threshold=0.5, k=-1)
                            f1_scores, support = zip(
                                *[(res['f1score'], label_counts_train[lbl]) for lbl, res in results_by_lbl.items() if
                                  lbl in label_counts_train])
                            tr_macro_f1 = np.average(f1_scores)
                            tr_micro_f1 = np.average(f1_scores, weights=support)

                            print("{0:.3f} micro f1. {1:.3f} macro f1. {2:.3f} train micro f1. {3:.3f} train macro f1".format(
                                micro_f1, macro_f1, tr_micro_f1, tr_macro_f1))
                            grid_search_results.append({'lr':lr, 'ndim':ndim, 'minCount':minCount, 'epoch':epoch, 'ws':w,
                                                        'val_micro_f1':micro_f1, 'val_macro_f1':macro_f1,
                                                        'tra_micro_f1':tr_micro_f1, 'tra_macro_f1':tr_macro_f1, 'wv':wv})

    print("\n==== Grid Search Results====\n")
    print(pd.DataFrame(grid_search_results)[
              ['lr','ndim','minCount','epoch','ws','val_micro_f1','tra_micro_f1', 'val_macro_f1', 'tra_macro_f1', 'wv']])
    print("\nBest: {0} lr; {1} dim; {2} min count; {3} epochs; {4} ws; {5} wv\n".format(best_lr, best_ndim, best_minCount, best_epochs, best_ws, best_wv))
    return best_lr, best_ndim, best_minCount, best_epochs, best_ws, best_wv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data",
                        default="/home/isaacj/fastText/drafttopic/wikitext/enwiki.balanced_article_sample.w_article_text_50413_train_data.txt")
    parser.add_argument("--val_data",
                        default="/home/isaacj/fastText/drafttopic/wikitext/enwiki.balanced_article_sample.w_article_text_6301_val_data.txt")
    parser.add_argument("--test_data",
                        default="/home/isaacj/fastText/drafttopic/wikitext/enwiki.balanced_article_sample.w_article_text_6303_test_data.txt")
    parser.add_argument("--false_negatives_fn")
    parser.add_argument("--output_model")
    parser.add_argument("--word_vectors", default='')
    parser.add_argument("--learning_rates", nargs="+", default=[0.1], type=float)
    parser.add_argument("--minCounts", nargs="+", default=[3], type=int)
    parser.add_argument("--epochs", nargs="+", default=[25], type=int)
    parser.add_argument("--ws", nargs="+", default=[20], type=int)
    parser.add_argument("--ndims", nargs="+", default=[50], type=int)
    parser.add_argument("--mlc_res_tsv")
    args = parser.parse_args()

    if os.path.exists(args.output_model):
        print("Loading model:", args.output_model)
        model = fasttext.load_model(args.output_model)
    else:
        if args.val_data and len(args.learning_rates + args.minCounts + args.epochs + args.ws + args.ndims) > 5:
            lr, ndim, minCount, epochs, ws, wv = grid_search(
                args.training_data, args.val_data, args.learning_rates, args.minCounts, args.epochs, args.ws, args.word_vectors, args.ndims)
        else:
            lr = args.learning_rates[0]
            minCount = args.minCounts[0]
            epochs = args.epochs[0]
            ws = args.ws[0]
            wv = args.word_vectors
            ndim = args.ndims[0]
    
        print("Building fasttext model: {0} lr; {1} min count; {2} epochs; {3} ws; wv: {4}".format(
            lr, minCount, epochs, ws, wv))
        start = time.time()
        model = fasttext.train_supervised(input=args.training_data,
                                          minCount=minCount,
                                          wordNgrams=WORDNGRAMS,
                                          lr=lr,
                                          epoch=epochs,
                                          pretrainedVectors=wv,
                                          ws=ws,
                                          dim=ndim,
                                          minn=MINN,
                                          maxn=MAXN,
                                          thread=MAXTHREADS,
                                          loss=LOSS,
                                          verbose=VERBOSITY)
        print("{0} seconds elapsed in training.".format(time.time() - start))

        if args.output_model:
            print("Dumping fasttext model to {0}".format(args.output_model))
            model.save_model(args.output_model)

    if args.test_data:
        # build statistics dataframe for printing
        print("==== test statistics ====")
        lbl_statistics = {}
        toplevel_statistics = {}
        threshold = 0.5
        all_lbls = model.get_labels()
        for lbl in all_lbls:
            lbl_statistics[lbl] = {'n': 0, 'FP': 0, 'TP': 0, 'FN': 0, 'TN': 0, 'true':[], 'pred':[]}
            toplevel_statistics[ft_to_toplevel(lbl)] = {'n': 0, 'FP': 0, 'TP': 0, 'FN': 0, 'TN': 0}
        with open(args.test_data, 'r') as fin:
            for line_no, datapoint in enumerate(fin):
                _, topics = model.get_line(datapoint.strip())
                prediction = model.predict(datapoint.strip(), k=-1)
                predicted_labels = []
                for idx in range(len(prediction[0])):
                    prob = prediction[1][idx]
                    lbl = prediction[0][idx]
                    lbl_statistics[lbl]['true'].append(int(lbl in topics))
                    lbl_statistics[lbl]['pred'].append(prob)
                    if prob > threshold:
                        predicted_labels.append(lbl)
                for lbl in all_lbls:
                    if lbl in topics and lbl in predicted_labels:
                        lbl_statistics[lbl]['n'] += 1
                        lbl_statistics[lbl]['TP'] += 1
                    elif lbl in topics:
                        lbl_statistics[lbl]['n'] += 1
                        lbl_statistics[lbl]['FN'] += 1
                    elif lbl in predicted_labels:
                        lbl_statistics[lbl]['FP'] += 1
                    else:
                        lbl_statistics[lbl]['TN'] += 1
                toplevel_topics = [ft_to_toplevel(l) for l in topics]
                toplevel_predictions = [ft_to_toplevel(l) for l in predicted_labels]
                for lbl in toplevel_statistics:
                    if lbl in toplevel_topics and lbl in toplevel_predictions:
                        toplevel_statistics[lbl]['n'] += 1
                        toplevel_statistics[lbl]['TP'] += 1
                    elif lbl in toplevel_topics:
                        toplevel_statistics[lbl]['n'] += 1
                        toplevel_statistics[lbl]['FN'] += 1
                    elif lbl in toplevel_predictions:
                        toplevel_statistics[lbl]['FP'] += 1
                    else:
                        toplevel_statistics[lbl]['TN'] += 1

        for lbl in all_lbls:
            s = lbl_statistics[lbl]
            fpr, tpr, _ = roc_curve(s['true'], s['pred'])
            s['pr-auc'] = auc(fpr, tpr)
            s['avg_pre'] = average_precision_score(s['true'], s['pred'])
            try:
                s['precision'] = s['TP'] / (s['TP'] + s['FP'])
            except ZeroDivisionError:
                s['precision'] = 0
            try:
                s['recall'] = s['TP'] / (s['TP'] + s['FN'])
            except ZeroDivisionError:
                s['recall'] = 0
            try:
                s['f1'] = 2 * (s['precision'] * s['recall']) / (s['precision'] + s['recall'])
            except ZeroDivisionError:
                s['f1'] = 0

        for lbl in toplevel_statistics:
            s = toplevel_statistics[lbl]
            try:
                s['precision'] = s['TP'] / (s['TP'] + s['FP'])
            except ZeroDivisionError:
                s['precision'] = 0
            try:
                s['recall'] = s['TP'] / (s['TP'] + s['FN'])
            except ZeroDivisionError:
                s['recall'] = 0
            try:
                s['f1'] = 2 * (s['precision'] * s['recall']) / (s['precision'] + s['recall'])
            except ZeroDivisionError:
                s['f1'] = 0

        print("\n=== Mid Level Categories ===")
        mlc_statistics = pd.DataFrame(lbl_statistics).T
        mlc_statistics['mid-level-category'] = [s.replace('__label__', '').replace('_', ' ') for s in mlc_statistics.index]
        mlc_statistics.set_index('mid-level-category', inplace=True)
        mlc_statistics[''] = '-->'
        mlc_statistics = mlc_statistics[['n', '', 'TP', 'FP', 'TN', 'FN', 'precision', 'recall', 'f1', 'pr-auc', 'avg_pre']]
        with pd.option_context('display.max_rows', None):
            print(mlc_statistics)
        if args.mlc_res_tsv:
            mlc_statistics.to_csv(args.mlc_res_tsv, sep='|')

        print("Dropping {0} because of Null values.".format(mlc_statistics[mlc_statistics.isnull().T.any()].index))
        mlc_statistics = mlc_statistics.dropna()
        print("\nPrecision: {0:.3f} micro; {1:.3f} macro".format(np.average(mlc_statistics['precision'], weights=mlc_statistics['n']), np.mean(mlc_statistics['precision'])))
        print("Recall: {0:.3f} micro; {1:.3f} macro".format(np.average(mlc_statistics['recall'], weights=mlc_statistics['n']), np.mean(mlc_statistics['recall'])))
        print("F1: {0:.3f} micro; {1:.3f} macro".format(np.average(mlc_statistics['f1'], weights=mlc_statistics['n']), np.mean(mlc_statistics['f1'])))
        print("PR-AUC: {0:.3f} micro; {1:.3f} macro".format(np.average(mlc_statistics['pr-auc'], weights=mlc_statistics['n']), np.mean(mlc_statistics['pr-auc'])))
        print("Avg pre.: {0:.3f} micro; {1:.3f} macro".format(np.average(mlc_statistics['avg_pre'], weights=mlc_statistics['n']), np.mean(mlc_statistics['avg_pre'])))

        print("\n=== Top Level Categories ===")
        tlc_statistics = pd.DataFrame(toplevel_statistics).T
        tlc_statistics.index.name = 'top-level-category'
        tlc_statistics[''] = '-->'
        tlc_statistics = tlc_statistics[['n', '', 'TP', 'FP', 'TN', 'FN', 'precision', 'recall', 'f1']]
        print(tlc_statistics)

        print("\nPrecision: {0:.3f} micro; {1:.3f} macro".format(np.average(tlc_statistics['precision'], weights=tlc_statistics['n']), np.mean(tlc_statistics['precision'])))
        print("Recall: {0:.3f} micro; {1:.3f} macro".format(np.average(tlc_statistics['recall'], weights=tlc_statistics['n']), np.mean(tlc_statistics['recall'])))
        print("F1: {0:.3f} micro; {1:.3f} macro".format(np.average(tlc_statistics['f1'], weights=tlc_statistics['n']), np.mean(tlc_statistics['f1'])))

        if args.false_negatives_fn:
            num_examples_per_label = 10
            false_negatives = {}
            for lbl in all_lbls:
                false_negatives[lbl] = []
            with open(args.test_data, 'r') as fin_data:
                with open(args.test_data.replace('data.txt', 'meta.txt'), 'r') as fin_metadata:
                    for line_no, datapoint in enumerate(fin_data):
                        claims, topics = model.get_line(datapoint.strip())
                        metadata = next(fin_metadata)
                        prediction = model.predict(datapoint.strip(), k=-1)
                        predicted_labels = [l for idx, l in enumerate(prediction[0]) if prediction[1][idx] > threshold]
                        for lbl in topics:
                            if lbl not in predicted_labels:
                                false_negatives[lbl].append('{0}\t{1}'.format(lbl, metadata))
                    with open(args.false_negatives_fn, 'w') as fout:
                        for lbl in false_negatives:
                            num_examples = min(len(false_negatives[lbl]), num_examples_per_label)
                            random_examples = np.random.choice(false_negatives[lbl],
                                                               num_examples,
                                                               replace=False)
                            for ex in random_examples:
                                fout.write(ex)


if __name__ == "__main__":
    main()

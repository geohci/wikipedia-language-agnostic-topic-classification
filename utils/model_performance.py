import argparse
import csv
import multiprocessing
import os
import sys

csv.field_size_limit(sys.maxsize)

import fasttext
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, average_precision_score  # auc

VERBOSITY = 2
WORDNGRAMS = 1
MINN = 0
MAXN = 0
MAXTHREADS = min(16, multiprocessing.cpu_count() - 1)
LOSS = 'ova'

def ft_to_toplevel(fasttext_lbl):
    return fasttext_lbl.replace('__label__','').split('.')[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_metadata", help="Metadata file with article QIDs and languages.")
    parser.add_argument("--test_data",
                        nargs="+", type=str,
                        default=[], help="List of test data to evaluate model.")
    parser.add_argument("--test_lbls", nargs="+", type=str, default=[], help="Labels for each test_data file.")
    parser.add_argument("--model_bin")
    parser.add_argument("--output_wikitext")
    args = parser.parse_args()

    test_lbls = args.test_lbls
    if len(test_lbls) and len(test_lbls) != len(args.test_data):
        raise IndexError("Mismatch in length of test files and labels")
    elif not len(test_lbls):
        test_lbls = [fn for fn in args.test_data]

    for fn in args.test_data:
        if not os.path.exists(fn):
            raise IOError("{0} does not exist.".format(fn))

    if args.training_metadata:
        langs = {}
        with open(args.training_metadata, 'r') as fin:
            csvreader = csv.reader(fin, delimiter='\t')
            for line in csvreader:
                try:
                    lang = line[1]
                except IndexError:
                    lang = "N/A"
                langs[lang] = langs.get(lang, 0) + 1
        print("Training data language breakdown:")
        sorted_langs = [(k, langs[k], langs[k] / sum(langs.values())) for k in sorted(langs, key=langs.get, reverse=True)]
        for l, num, prop in sorted_langs:
            print("{0}: {1} ({2:.3f})".format(l, num, prop))

    model = fasttext.load_model(args.model_bin)
    print("\nModel architecture:")
    print("Embedding size: {0}".format(model.get_dimension()))
    print("Total # params: {0}".format(model.get_output_matrix().size))
    print("Vocab size: {0}".format(len(model.get_words())))
    print("File size: {0:.3f} MB".format(os.path.getsize(args.model_bin) / 1048576))

    print("\nTest statistics:")
    model_performance = pd.DataFrame()
    model_summary = []
    all_lbls = model.get_labels()
    threshold = 0.5
    for i, fn in enumerate(args.test_data):
        # build statistics dataframe for printing
        lbl_statistics = {}
        toplevel_statistics = {}
        for lbl in all_lbls:
            lbl_statistics[lbl] = {'n': 0, 'FP': 0, 'TP': 0, 'FN': 0, 'TN': 0, 'true':[], 'pred':[], 'lang':test_lbls[i]}
            toplevel_statistics[ft_to_toplevel(lbl)] = {'n': 0, 'FP': 0, 'TP': 0, 'FN': 0, 'TN': 0, 'lang':test_lbls[i]}
        with open(fn, 'r') as fin:
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
            #s['pr-auc'] = auc(fpr, tpr)
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
        mlc_statistics = mlc_statistics[['lang', 'n', 'TP', 'FP', 'TN', 'FN', 'precision', 'recall', 'f1', 'avg_pre']]  # 'pr-auc'
        with pd.option_context('display.max_rows', None):
            print(mlc_statistics)


        model_performance = model_performance.append(mlc_statistics)
        print("Dropping {0} because of Null values.".format(mlc_statistics[mlc_statistics.isnull().T.any()].index))
        mlc_statistics = mlc_statistics.dropna()
        mi_pre = np.average(mlc_statistics['precision'], weights=mlc_statistics['n'])
        ma_pre = np.mean(mlc_statistics['precision'])
        mi_rec = np.average(mlc_statistics['recall'], weights=mlc_statistics['n'])
        ma_rec = np.mean(mlc_statistics['recall'])
        mi_f1 = np.average(mlc_statistics['f1'], weights=mlc_statistics['n'])
        ma_f1 = np.mean(mlc_statistics['f1'])
        mi_avgpre = np.average(mlc_statistics['avg_pre'], weights=mlc_statistics['n'])
        ma_avgpre = np.mean(mlc_statistics['avg_pre'])
        model_summary.append([test_lbls[i], mi_pre, ma_pre, mi_rec, ma_rec, mi_f1, ma_f1, mi_avgpre, ma_avgpre])
        print("\nPrecision: {0:.3f} micro; {1:.3f} macro".format(mi_pre, ma_pre))
        print("Recall: {0:.3f} micro; {1:.3f} macro".format(mi_rec, ma_rec))
        print("F1: {0:.3f} micro; {1:.3f} macro".format(mi_f1, ma_f1))
        print("Avg pre.: {0:.3f} micro; {1:.3f} macro".format(mi_avgpre, ma_avgpre))

        print("\n=== Top Level Categories ===")
        tlc_statistics = pd.DataFrame(toplevel_statistics).T
        tlc_statistics.index.name = 'top-level-category'
        tlc_statistics[''] = '-->'
        tlc_statistics = tlc_statistics[['lang', 'n', 'TP', 'FP', 'TN', 'FN', 'precision', 'recall', 'f1']]
        print(tlc_statistics)

        print("\nPrecision: {0:.3f} micro; {1:.3f} macro".format(np.average(tlc_statistics['precision'], weights=tlc_statistics['n']), np.mean(tlc_statistics['precision'])))
        print("Recall: {0:.3f} micro; {1:.3f} macro".format(np.average(tlc_statistics['recall'], weights=tlc_statistics['n']), np.mean(tlc_statistics['recall'])))
        print("F1: {0:.3f} micro; {1:.3f} macro".format(np.average(tlc_statistics['f1'], weights=tlc_statistics['n']), np.mean(tlc_statistics['f1'])))

    table = '{{| class="wikitable sortable mw-collapsible"\n|-\n!topic || {0}\n'.format(' || '.join([c for c in model_performance.columns]))
    for row in model_performance.itertuples():
        table = table + "|-\n"
        table = table + '|{0}\n'.format(' || '.join([format_output(v) for v in row]))
    table = table + ("|}")

    if args.output_wikitext:
        with open(args.output_wikitext, 'w') as fout:
            fout.write(table)
    else:
        print("\nWikitext format:\n")
        print(table)

    header = ['language', 'micro precision', 'macro precision', 'micro recall', 'macro recall', 'micro f1', 'macro f1', 'micro avg. pre.', 'macro avg. pre.']
    table = '{{| class="wikitable sortable mw-collapsible"\n|-\n!{0}\n'.format(
        ' || '.join([c for c in header]))
    for row in model_summary:
        table = table + "|-\n"
        table = table + '|{0}\n'.format(' || '.join([format_output(v) for v in row]))
    table = table + ("|}")
    print(table)

def format_output(v):
    if type(v) == float or type(v) == np.float64:
        return '{0:.3f}'.format(v)
    else:
        return str(v)

if __name__ == "__main__":
    main()
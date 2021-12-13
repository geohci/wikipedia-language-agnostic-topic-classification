# Make predictions for articles with trained model
# Expects folder with TSVs with every single article and outlinks
# Outputs stats to wikitext
import argparse
import bz2
import csv
import json
import os
import sys

csv.field_size_limit(sys.maxsize)

import fasttext
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

def ft_to_toplevel(fasttext_lbl):
    return fasttext_lbl.replace('__label__','').split('.')[0]

def bucket_outlinks(outlink_count):
    if outlink_count < 10:
        return str(outlink_count)
    elif outlink_count < 20:
        return '10-19'
    elif outlink_count < 30:
        return '20-29'
    elif outlink_count < 40:
        return '30-39'
    elif outlink_count < 50:
        return '40-49'
    else:
        return '>=50'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collected_outlinks_tsvs", help="Folder with TSVs of format wiki_db, article QID, article PID, outlinks.")
    parser.add_argument("--output_predictions_tsv", help="Output TSV with predictions")
    parser.add_argument("--output_lang_data", help="TSV with language-label counts")
    parser.add_argument("--model_bin", help="Path to trained fastText model binary.")
    parser.add_argument("--labeled_qids", help="Path to Bzipped JSON with QIDs + labels.")
    parser.add_argument("--wikidb_wikitext", help="Path to wiki_db wikitext performance table.")
    parser.add_argument("--outlink_wikitext", help="Path to outlink wikitext performance table.")
    args = parser.parse_args()

    if not os.path.exists(args.output_predictions_tsv):
        model = fasttext.load_model(args.model_bin)
        print("\nModel architecture:")
        print("Embedding size: {0}".format(model.get_dimension()))
        print("Total # params: {0}".format(model.get_output_matrix().size))
        print("Vocab size: {0}".format(len(model.get_words())))
        print("File size: {0:.3f} MB".format(os.path.getsize(args.model_bin) / 1048576))

        all_lbls = sorted(model.get_labels())  # alphabetical order
        all_lbls_header = [l.replace("__label__", "") for l in all_lbls]
        all_lbls_idx = {l:i for i,l in enumerate(all_lbls)}
        num_lbls = len(all_lbls)

        test_tsvs = [fn for fn in os.listdir(args.collected_outlinks_tsvs) if fn.endswith('.csv.bz2')]
        processed = {}
        topics_by_lang = {}
        threshold = 0.5
        with bz2.open(args.output_predictions_tsv, 'wt') as fout:
            tsvwriter = csv.writer(fout, delimiter='\t')
            tsvwriter.writerow(['wiki_db', 'qid', 'pid', 'num_outlinks'] + all_lbls_header)
            for i, test_tsv in enumerate(test_tsvs, start=1):
                with bz2.open(os.path.join(args.collected_outlinks_tsvs, test_tsv), 'rt') as fin:
                    tsvreader = csv.reader(fin, delimiter='\t')
                    header = next(tsvreader)
                    assert header == ['wiki_db', 'pid_from', 'qid_from', 'outlinks']
                    for line in tsvreader:
                        wiki_db = line[0]
                        if wiki_db not in topics_by_lang:
                            topics_by_lang[wiki_db] = {'n':0}
                            for l in all_lbls_header:
                                topics_by_lang[wiki_db][l] = 0
                        pid_from = line[1]
                        qid_from = line[2]
                        outlinks = line[3]
                        processed[wiki_db] = processed.get(wiki_db, 0) + 1
                        topics_by_lang[wiki_db]['n'] += 1
                        prediction = model.predict(outlinks, k=-1)
                        scores_headerorder = [None] * num_lbls
                        for label, score in zip(*prediction):
                            scores_headerorder[all_lbls_idx[label]] = '{0:.3f}'.format(score)
                            if score > threshold:
                                topics_by_lang[wiki_db][label.replace("__label__", "")] += 1
                        tsvwriter.writerow([wiki_db, qid_from, pid_from, len(outlinks.split())] + scores_headerorder)

                top_five = [(w,processed[w]) for w in sorted(processed, key=processed.get, reverse=True)][:5]
                print("File {0}/{1}. {2} total lines processed. Top 5 wiki_dbs: {3}".format(i, len(test_tsvs), sum(processed.values()), top_five))

        print("Completed!")
        wikidb_counts = [(w,processed[w]) for w in sorted(processed, key=processed.get, reverse=True)]
        for w,c in wikidb_counts:
            print("{0}: {1} articles.".format(w, c))

        topics_by_lang = pd.DataFrame(topics_by_lang).T
        topics_by_lang.sort_values('n', inplace=True)
        topics_by_lang = topics_by_lang[['n'] + all_lbls_header]
        if args.output_lang_data:
            topics_by_lang.to_csv(args.output_lang_data, sep='\t', header=True, index=True)
    else:
        model_performance(args.output_predictions_tsv, args.labeled_qids, args.wikidb_wikitext, args.outlink_wikitext)

def model_performance_avgpre(predictions_tsv, labeled_qids, output_wikidb_wikitext_fn, output_outlink_wikitext_fn):
    qid_to_topic = {}
    all_lbls = set()
    qid_missing = 0
    with bz2.open(labeled_qids, 'rt') as fin:
        for line in fin:
            line = json.loads(line)
            try:
                qid = line['qid']
                topics = line['topics']
                qid_to_topic[qid] = topics
                for t in topics:
                    all_lbls.add(t)
            except KeyError:
                qid_missing += 1
    print("{0} QIDs with topics. {1} skipped because no QIDs.".format(len(qid_to_topic), qid_missing))

    threshold = 0.5
    lbl_lang_stats = {}
    lbl_outlink_stats = {}
    wiki_db_counts = {}
    outlink_counts = {}
    with bz2.open(predictions_tsv, 'rt') as fin:
        tsvreader = csv.reader(fin, delimiter='\t')
        header = next(tsvreader)
        metadata = ['wiki_db', 'qid', 'pid', 'num_outlinks']
        wdb_idx = metadata.index('wiki_db')
        qid_idx = metadata.index('qid')
        out_idx = metadata.index('num_outlinks')
        topic_start_idx = len(metadata)
        assert header[:topic_start_idx] == metadata
        topic_lbls = header[topic_start_idx:]
        topic_to_idx = {t.replace("_", " "):i for i,t in enumerate(topic_lbls, start=topic_start_idx)}
        for idx, line in enumerate(tsvreader, start=1):
            qid = line[qid_idx]
            wiki_db = line[wdb_idx]
            outlink_bucket = bucket_outlinks(int(line[out_idx]))
            if wiki_db not in lbl_lang_stats:
                lbl_lang_stats[wiki_db] = {}
                wiki_db_counts[wiki_db] = {'n':0, '-n':0}
                for lbl in all_lbls:
                    lbl_lang_stats[wiki_db][lbl] = {'n': 0, 'FP': 0, 'TP': 0, 'FN': 0, 'TN': 0, 'true': [], 'pred': [], 'wiki_db':wiki_db}
            if outlink_bucket not in lbl_outlink_stats:
                lbl_outlink_stats[outlink_bucket] = {}
                outlink_counts[outlink_bucket] = {'n':0, '-n':0}
                for lbl in all_lbls:
                    lbl_outlink_stats[outlink_bucket][lbl] = {'n': 0, 'FP': 0, 'TP': 0, 'FN': 0, 'TN': 0, 'true': [],
                                                              'pred': [], '# outlinks': outlink_bucket}
            if qid in qid_to_topic:
                wiki_db_counts[wiki_db]['n'] += 1
                outlink_counts[outlink_bucket]['n'] += 1
                predicted_labels = []
                true_labels = qid_to_topic[qid]
                for lbl in topic_to_idx:
                    prob = float(line[topic_to_idx[lbl]])
                    lbl_lang_stats[wiki_db][lbl]['true'].append(int(lbl in true_labels))
                    lbl_lang_stats[wiki_db][lbl]['pred'].append(prob)
                    if prob >= threshold:
                        predicted_labels.append(lbl)
                for lbl in all_lbls:
                    if lbl in true_labels and lbl in predicted_labels:
                        lbl_lang_stats[wiki_db][lbl]['n'] += 1
                        lbl_lang_stats[wiki_db][lbl]['TP'] += 1
                        lbl_outlink_stats[outlink_bucket][lbl]['n'] += 1
                        lbl_outlink_stats[outlink_bucket][lbl]['TP'] += 1
                    elif lbl in true_labels:
                        lbl_lang_stats[wiki_db][lbl]['n'] += 1
                        lbl_lang_stats[wiki_db][lbl]['FN'] += 1
                        lbl_outlink_stats[outlink_bucket][lbl]['n'] += 1
                        lbl_outlink_stats[outlink_bucket][lbl]['FN'] += 1
                    elif lbl in predicted_labels:
                        lbl_lang_stats[wiki_db][lbl]['FP'] += 1
                        lbl_outlink_stats[outlink_bucket][lbl]['FP'] += 1
                    else:
                        lbl_lang_stats[wiki_db][lbl]['TN'] += 1
                        lbl_outlink_stats[outlink_bucket][lbl]['TN'] += 1
            else:
                wiki_db_counts[wiki_db]['-n'] += 1
                outlink_counts[outlink_bucket]['-n'] += 1

            if idx % 100000 == 0:
                print("{0} lines processed.".format(idx))

        print("Fin! {0} lines processed.".format(idx))
        for wiki_db in lbl_lang_stats:
            for lbl in lbl_lang_stats[wiki_db]:
                s = lbl_lang_stats[wiki_db][lbl]
                try:
                    s['avg_pre'] = average_precision_score(s['true'], s['pred'])
                except IndexError:
                    s['avg_pre'] = 0
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
        for outlink_bucket in lbl_outlink_stats:
            for lbl in lbl_outlink_stats[outlink_bucket]:
                s = lbl_outlink_stats[outlink_bucket][lbl]
                try:
                    s['avg_pre'] = average_precision_score(s['true'], s['pred'])
                except IndexError:
                    s['avg_pre'] = 0
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

        model_lbl_stats = pd.DataFrame()
        model_summary = []
        for wiki_db in lbl_lang_stats:
            mlc_statistics = pd.DataFrame(lbl_lang_stats[wiki_db]).T
            mlc_statistics['mid-level-category'] = [s for s in mlc_statistics.index]
            mlc_statistics.set_index('mid-level-category', inplace=True)
            mlc_statistics = mlc_statistics[['wiki_db', 'n', 'TP', 'FP', 'TN', 'FN', 'precision', 'recall', 'f1', 'avg_pre']]
            to_drop = list(mlc_statistics[mlc_statistics.isnull().T.any()].index)
            if len(to_drop):
                #print("{0}: dropping {1} because of Null values.".format(wiki_db, to_drop))
                pass
            mlc_statistics = mlc_statistics.dropna()
            to_drop = list(mlc_statistics[mlc_statistics['n'] == 0].index)
            if len(to_drop):
                #print("{0}: dropping {1} because of no instances.".format(wiki_db, to_drop))
                pass
            mlc_statistics = mlc_statistics[mlc_statistics['n'] > 0]
            model_lbl_stats = model_lbl_stats.append(mlc_statistics)
            try:
                mi_pre = np.average(mlc_statistics['precision'], weights=mlc_statistics['n'])
                ma_pre = np.mean(mlc_statistics['precision'])
                mi_rec = np.average(mlc_statistics['recall'], weights=mlc_statistics['n'])
                ma_rec = np.mean(mlc_statistics['recall'])
                mi_f1 = np.average(mlc_statistics['f1'], weights=mlc_statistics['n'])
                ma_f1 = np.mean(mlc_statistics['f1'])
                mi_avgpre = np.average(mlc_statistics['avg_pre'], weights=mlc_statistics['n'])
                ma_avgpre = np.mean(mlc_statistics['avg_pre'])
                model_summary.append([wiki_db, wiki_db_counts[wiki_db]['n'], wiki_db_counts[wiki_db]['-n'],
                                      mi_pre, ma_pre, mi_rec, ma_rec, mi_f1, ma_f1, mi_avgpre, ma_avgpre])
            except ZeroDivisionError:
                model_summary.append([wiki_db, wiki_db_counts[wiki_db]['n'], wiki_db_counts[wiki_db]['-n'],
                                      0, 0, 0, 0, 0, 0, 0, 0])

        outlink_stats = pd.DataFrame()
        outlink_summary = []
        for outlink_bucket in lbl_outlink_stats:
            outlink_statistics = pd.DataFrame(lbl_outlink_stats[outlink_bucket]).T
            outlink_statistics['mid-level-category'] = [s.replace('_', ' ') for s in outlink_statistics.index]
            outlink_statistics.set_index('mid-level-category', inplace=True)
            outlink_statistics = outlink_statistics[['# outlinks', 'n', 'TP', 'FP', 'TN', 'FN', 'precision', 'recall', 'f1', 'avg_pre']]
            to_drop = list(outlink_statistics[outlink_statistics.isnull().T.any()].index)
            if len(to_drop):
                #print("{0}: dropping {1} because of Null values.".format(wiki_db, to_drop))
                pass
            outlink_statistics = outlink_statistics.dropna()
            to_drop = list(outlink_statistics[outlink_statistics['n'] == 0].index)
            if len(to_drop):
                #print("{0}: dropping {1} because of no instances.".format(wiki_db, to_drop))
                pass
            outlink_statistics = outlink_statistics[outlink_statistics['n'] > 0]
            outlink_stats = outlink_stats.append(outlink_statistics)
            try:
                mi_pre = np.average(outlink_statistics['precision'], weights=outlink_statistics['n'])
                ma_pre = np.mean(outlink_statistics['precision'])
                mi_rec = np.average(outlink_statistics['recall'], weights=outlink_statistics['n'])
                ma_rec = np.mean(outlink_statistics['recall'])
                mi_f1 = np.average(outlink_statistics['f1'], weights=outlink_statistics['n'])
                ma_f1 = np.mean(outlink_statistics['f1'])
                mi_avgpre = np.average(outlink_statistics['avg_pre'], weights=outlink_statistics['n'])
                ma_avgpre = np.mean(outlink_statistics['avg_pre'])
                outlink_summary.append([outlink_bucket, outlink_counts[outlink_bucket]['n'], outlink_counts[outlink_bucket]['-n'],
                                        mi_pre, ma_pre, mi_rec, ma_rec, mi_f1, ma_f1, mi_avgpre, ma_avgpre])
            except ZeroDivisionError:
                outlink_summary.append([outlink_bucket, outlink_counts[outlink_bucket]['n'], outlink_counts[outlink_bucket]['-n'],
                                      0, 0, 0, 0, 0, 0, 0, 0])

        with open(output_wikidb_wikitext_fn, 'w') as fout:
            header = ['wiki_db', 'evaluated', 'skipped',
                      'micro precision', 'macro precision', 'micro recall', 'macro recall',
                      'micro f1', 'macro f1', 'micro avg. pre.', 'macro avg. pre.']
            table = '{{| class="wikitable sortable mw-collapsible"\n|-\n!{0}\n'.format(
                ' || '.join([c for c in header]))
            for row in model_summary:
                table = table + "|-\n"
                table = table + '|{0}\n'.format(' || '.join([format_output(v) for v in row]))
            table = table + ("|}")
            fout.write(table)
            fout.write("\n\n\n")
            table = '{{| class="wikitable sortable mw-collapsible"\n|-\n!topic || {0}\n'.format(
                ' || '.join([c for c in model_lbl_stats.columns]))
            for row in model_lbl_stats.itertuples():
                table = table + "|-\n"
                table = table + '|{0}\n'.format(' || '.join([format_output(v) for v in row]))
            table = table + ("|}")
            fout.write(table)

        with open(output_outlink_wikitext_fn, 'w') as fout:
            header = ['# outlinks', 'evaluated', 'skipped',
                      'micro precision', 'macro precision', 'micro recall', 'macro recall',
                      'micro f1', 'macro f1', 'micro avg. pre.', 'macro avg. pre.']
            table = '{{| class="wikitable sortable mw-collapsible"\n|-\n!{0}\n'.format(
                ' || '.join([c for c in header]))
            for row in outlink_summary:
                table = table + "|-\n"
                table = table + '|{0}\n'.format(' || '.join([format_output(v) for v in row]))
            table = table + ("|}")
            fout.write(table)
            fout.write("\n\n\n")
            table = '{{| class="wikitable sortable mw-collapsible"\n|-\n!topic || {0}\n'.format(
                ' || '.join([c for c in outlink_stats.columns]))
            for row in outlink_stats.itertuples():
                table = table + "|-\n"
                table = table + '|{0}\n'.format(' || '.join([format_output(v) for v in row]))
            table = table + ("|}")
            fout.write(table)

def model_performance(predictions_tsv, labeled_qids, output_wikidb_wikitext_fn, output_outlink_wikitext_fn):
    qid_to_topic = {}
    all_lbls = set()
    qid_missing = 0
    with bz2.open(labeled_qids, 'rt') as fin:
        for line in fin:
            line = json.loads(line)
            try:
                qid = line['qid']
                topics = line['topics']
                qid_to_topic[qid] = topics
                for t in topics:
                    all_lbls.add(t)
            except KeyError:
                qid_missing += 1
    print("{0} QIDs with topics. {1} skipped because no QIDs.".format(len(qid_to_topic), qid_missing))

    threshold = 0.5
    lbl_lang_stats = {}
    lbl_outlink_stats = {}
    wiki_db_counts = {}
    outlink_counts = {}
    with bz2.open(predictions_tsv, 'rt') as fin:
        tsvreader = csv.reader(fin, delimiter='\t')
        header = next(tsvreader)
        metadata = ['wiki_db', 'qid', 'pid', 'num_outlinks']
        wdb_idx = metadata.index('wiki_db')
        qid_idx = metadata.index('qid')
        out_idx = metadata.index('num_outlinks')
        topic_start_idx = len(metadata)
        assert header[:topic_start_idx] == metadata
        topic_lbls = header[topic_start_idx:]
        topic_to_idx = {t.replace("_", " "):i for i,t in enumerate(topic_lbls, start=topic_start_idx)}
        for idx, line in enumerate(tsvreader, start=1):
            qid = line[qid_idx]
            wiki_db = line[wdb_idx]
            outlink_bucket = bucket_outlinks(int(line[out_idx]))
            if wiki_db not in lbl_lang_stats:
                lbl_lang_stats[wiki_db] = {}
                wiki_db_counts[wiki_db] = {'n':0, '-n':0}
                for lbl in all_lbls:
                    lbl_lang_stats[wiki_db][lbl] = {'n': 0, 'FP': 0, 'TP': 0, 'FN': 0, 'TN': 0, 'wiki_db':wiki_db}
            if outlink_bucket not in lbl_outlink_stats:
                lbl_outlink_stats[outlink_bucket] = {}
                outlink_counts[outlink_bucket] = {'n':0, '-n':0}
                for lbl in all_lbls:
                    lbl_outlink_stats[outlink_bucket][lbl] = {'n': 0, 'FP': 0, 'TP': 0, 'FN': 0, 'TN': 0, '# outlinks': outlink_bucket}
            if qid in qid_to_topic:
                wiki_db_counts[wiki_db]['n'] += 1
                outlink_counts[outlink_bucket]['n'] += 1
                predicted_labels = []
                true_labels = qid_to_topic[qid]
                for lbl in topic_to_idx:
                    prob = float(line[topic_to_idx[lbl]])
                    if prob >= threshold:
                        predicted_labels.append(lbl)
                for lbl in all_lbls:
                    if lbl in true_labels and lbl in predicted_labels:
                        lbl_lang_stats[wiki_db][lbl]['n'] += 1
                        lbl_lang_stats[wiki_db][lbl]['TP'] += 1
                        lbl_outlink_stats[outlink_bucket][lbl]['n'] += 1
                        lbl_outlink_stats[outlink_bucket][lbl]['TP'] += 1
                    elif lbl in true_labels:
                        lbl_lang_stats[wiki_db][lbl]['n'] += 1
                        lbl_lang_stats[wiki_db][lbl]['FN'] += 1
                        lbl_outlink_stats[outlink_bucket][lbl]['n'] += 1
                        lbl_outlink_stats[outlink_bucket][lbl]['FN'] += 1
                    elif lbl in predicted_labels:
                        lbl_lang_stats[wiki_db][lbl]['FP'] += 1
                        lbl_outlink_stats[outlink_bucket][lbl]['FP'] += 1
                    else:
                        lbl_lang_stats[wiki_db][lbl]['TN'] += 1
                        lbl_outlink_stats[outlink_bucket][lbl]['TN'] += 1
            else:
                wiki_db_counts[wiki_db]['-n'] += 1
                outlink_counts[outlink_bucket]['-n'] += 1

            if idx % 100000 == 0:
                print("{0} lines processed.".format(idx))

        print("Fin! {0} lines processed.".format(idx))
        for wiki_db in lbl_lang_stats:
            for lbl in lbl_lang_stats[wiki_db]:
                s = lbl_lang_stats[wiki_db][lbl]
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
        for outlink_bucket in lbl_outlink_stats:
            for lbl in lbl_outlink_stats[outlink_bucket]:
                s = lbl_outlink_stats[outlink_bucket][lbl]
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

        model_lbl_stats = pd.DataFrame()
        model_summary = []
        for wiki_db in lbl_lang_stats:
            mlc_statistics = pd.DataFrame(lbl_lang_stats[wiki_db]).T
            mlc_statistics['mid-level-category'] = [s for s in mlc_statistics.index]
            mlc_statistics.set_index('mid-level-category', inplace=True)
            mlc_statistics = mlc_statistics[['wiki_db', 'n', 'TP', 'FP', 'TN', 'FN', 'precision', 'recall', 'f1']]
            to_drop = list(mlc_statistics[mlc_statistics.isnull().T.any()].index)
            if len(to_drop):
                #print("{0}: dropping {1} because of Null values.".format(wiki_db, to_drop))
                pass
            mlc_statistics = mlc_statistics.dropna()
            to_drop = list(mlc_statistics[mlc_statistics['n'] == 0].index)
            if len(to_drop):
                #print("{0}: dropping {1} because of no instances.".format(wiki_db, to_drop))
                pass
            mlc_statistics = mlc_statistics[mlc_statistics['n'] > 0]
            model_lbl_stats = model_lbl_stats.append(mlc_statistics)
            try:
                mi_pre = np.average(mlc_statistics['precision'], weights=mlc_statistics['n'])
                ma_pre = np.mean(mlc_statistics['precision'])
                mi_rec = np.average(mlc_statistics['recall'], weights=mlc_statistics['n'])
                ma_rec = np.mean(mlc_statistics['recall'])
                mi_f1 = np.average(mlc_statistics['f1'], weights=mlc_statistics['n'])
                ma_f1 = np.mean(mlc_statistics['f1'])
                model_summary.append([wiki_db, wiki_db_counts[wiki_db]['n'], wiki_db_counts[wiki_db]['-n'],
                                      mi_pre, ma_pre, mi_rec, ma_rec, mi_f1, ma_f1])
            except ZeroDivisionError:
                model_summary.append([wiki_db, wiki_db_counts[wiki_db]['n'], wiki_db_counts[wiki_db]['-n'],
                                      0, 0, 0, 0, 0, 0])

        outlink_stats = pd.DataFrame()
        outlink_summary = []
        for outlink_bucket in lbl_outlink_stats:
            outlink_statistics = pd.DataFrame(lbl_outlink_stats[outlink_bucket]).T
            outlink_statistics['mid-level-category'] = [s.replace('_', ' ') for s in outlink_statistics.index]
            outlink_statistics.set_index('mid-level-category', inplace=True)
            outlink_statistics = outlink_statistics[['# outlinks', 'n', 'TP', 'FP', 'TN', 'FN', 'precision', 'recall', 'f1']]
            to_drop = list(outlink_statistics[outlink_statistics.isnull().T.any()].index)
            if len(to_drop):
                #print("{0}: dropping {1} because of Null values.".format(wiki_db, to_drop))
                pass
            outlink_statistics = outlink_statistics.dropna()
            to_drop = list(outlink_statistics[outlink_statistics['n'] == 0].index)
            if len(to_drop):
                #print("{0}: dropping {1} because of no instances.".format(wiki_db, to_drop))
                pass
            outlink_statistics = outlink_statistics[outlink_statistics['n'] > 0]
            outlink_stats = outlink_stats.append(outlink_statistics)
            try:
                mi_pre = np.average(outlink_statistics['precision'], weights=outlink_statistics['n'])
                ma_pre = np.mean(outlink_statistics['precision'])
                mi_rec = np.average(outlink_statistics['recall'], weights=outlink_statistics['n'])
                ma_rec = np.mean(outlink_statistics['recall'])
                mi_f1 = np.average(outlink_statistics['f1'], weights=outlink_statistics['n'])
                ma_f1 = np.mean(outlink_statistics['f1'])
                outlink_summary.append([outlink_bucket, outlink_counts[outlink_bucket]['n'], outlink_counts[outlink_bucket]['-n'],
                                        mi_pre, ma_pre, mi_rec, ma_rec, mi_f1, ma_f1])
            except ZeroDivisionError:
                outlink_summary.append([outlink_bucket, outlink_counts[outlink_bucket]['n'], outlink_counts[outlink_bucket]['-n'],
                                        0, 0, 0, 0, 0, 0])


        with open(output_wikidb_wikitext_fn, 'w') as fout:
            header = ['wiki_db', 'evaluated', 'skipped',
                      'micro precision', 'macro precision', 'micro recall', 'macro recall',
                      'micro f1', 'macro f1']
            table = '{{| class="wikitable sortable mw-collapsible"\n|-\n!{0}\n'.format(
                ' || '.join([c for c in header]))
            for row in model_summary:
                table = table + "|-\n"
                table = table + '|{0}\n'.format(' || '.join([format_output(v) for v in row]))
            table = table + ("|}")
            fout.write(table)
            fout.write("\n\n\n")
            table = '{{| class="wikitable sortable mw-collapsible"\n|-\n!topic || {0}\n'.format(
                ' || '.join([c for c in model_lbl_stats.columns]))
            for row in model_lbl_stats.itertuples():
                table = table + "|-\n"
                table = table + '|{0}\n'.format(' || '.join([format_output(v) for v in row]))
            table = table + ("|}")
            fout.write(table)

        with open(output_outlink_wikitext_fn, 'w') as fout:
            header = ['# outlinks', 'evaluated', 'skipped',
                      'micro precision', 'macro precision', 'micro recall', 'macro recall',
                      'micro f1', 'macro f1']
            table = '{{| class="wikitable sortable mw-collapsible"\n|-\n!{0}\n'.format(
                ' || '.join([c for c in header]))
            for row in outlink_summary:
                table = table + "|-\n"
                table = table + '|{0}\n'.format(' || '.join([format_output(v) for v in row]))
            table = table + ("|}")
            fout.write(table)
            fout.write("\n\n\n")
            table = '{{| class="wikitable sortable mw-collapsible"\n|-\n!topic || {0}\n'.format(
                ' || '.join([c for c in outlink_stats.columns]))
            for row in outlink_stats.itertuples():
                table = table + "|-\n"
                table = table + '|{0}\n'.format(' || '.join([format_output(v) for v in row]))
            table = table + ("|}")
            fout.write(table)


def format_output(v):
    if type(v) == float or type(v) == np.float64:
        return '{0:.3f}'.format(v)
    else:
        return str(v)

if __name__ == "__main__":
    main()

import argparse
import csv
import bz2
import gzip
import os
import random

# pagelinks table is of the format pageID -> page title
# there are a two main challenges with this format:
# * we want pageID -> all pageIDs of outlinks (and eventually QID -> all QIDs of outlinks)
# * redirects need to be resolved
#
# the redirects table is also in the format pageID -> page title
# but needs to be page title -> pageID
#
# we also need a general page title -> pageID mapping for the whole wiki
#
# putting all these pieces together, we can then get our desired pageID -> pageID mapping
# and convert it to QIDs
# or flip it for inlinks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default='en', help="Wikipedia language to parse")
    parser.add_argument("--sqldate", default='latest', help='Dump file date to parse.')
    parser.add_argument("--output_outlinks_tsv", help='Bzipped base TSV with article and list of outlinks')
    parser.add_argument("--pid_to_qid_fn", help="Bzipped TSV file with mapping of pageIDs -> QIDs for lang")
    parser.add_argument("--qid_to_topics_fn", help="Bzipped TSV file with the topics associated with each QID.")
    parser.add_argument("--fasttext_data_dir", help="Folder where fastText data can go.")
    args = parser.parse_args()

    # get all redirects of form redirect pageID -> canonical page title
    # this actually needs to be flipped: redirect page title -> canonical pageID
    redirects_pid_title = get_redirects(args.lang, args.sqldate, ns=0)

    # gather mapping of all pageID -> page title
    # this is used to flip the redirects from pageID -> pageTitle to pageTitle -> pageID
    # and eventually also map outlinks in pageTitle format to pageID
    pid_to_title = get_pid_to_ptitle(args.lang, args.sqldate, ns=0)
    # flip for title -> pageID in order to update redirects
    title_to_pid = {t:p for p,t in pid_to_title.items()}
    redirects_title_pid = {}
    src_missing = set()
    tar_missing = set()
    for src_pid, red_title in redirects_pid_title.items():
        try:
            src_title = pid_to_title[src_pid]
        except KeyError:
            #print('https://{0}.wikipedia.org/wiki/?curid={1} missing from page table.'.format(args.lang, src_pid))
            src_missing.add(src_pid)
            continue
        try:
            red_pid = title_to_pid[red_title]
        except KeyError:
            #print('https://{0}.wikipedia.org/wiki/{1} missing from page table.'.format(args.lang, red_title))
            tar_missing.add(red_title)
            continue
        redirects_title_pid[src_title] = red_pid
    print("{0} redirects. {1} source page IDs not found. {2} target page titles not found.".format(
        len(redirects_title_pid), len(src_missing), len(tar_missing)))

    # build an outlinks dataset of format: source_pageID target_pageID
    # each source_pageID might have many lines and associated target_pageIDs
    # duplicate target_pageIDs for a page can happen when a link is included twice in an article with different titles
    # otherwise, the link only is included the first time it appears on the page
    # this is because of Mediawiki not this code: https://www.mediawiki.org/wiki/Manual:Pagelinks_table
    redirects_captured = 0
    redirects_skipped = 0
    direct_find = 0
    outlink_missing = 0
    with bz2.open(args.output_outlinks_tsv, 'wt') as fout:
        print("Writing to {0}".format(args.output_outlinks_tsv))
        i = 0
        for source_pid, outlink_ptitle in get_links_all(args.lang, args.sqldate, ns=0):
            i += 1
            # source page is a redirect -- skip
            if source_pid in redirects_pid_title:
                redirects_skipped += 1
                continue
            # target page is a redirect -- resolve redirect and include
            elif outlink_ptitle in redirects_title_pid:
                outlink_pid = redirects_title_pid[outlink_ptitle]
                redirects_captured += 1
            # standard wiki-link (both pages are articles) -- keep as is
            elif outlink_ptitle in title_to_pid:
                outlink_pid = title_to_pid[outlink_ptitle]
                direct_find += 1
            # link goes to an article that doesn't exist (redlink) or somehow is missing from page/redirect dataset
            else:
                #print("Skipping https://{0}.wikipedia.org/wiki/?curid={1} -> https://{0}.wikipedia.org/wiki/{2}".format(args.lang, source_pid, outlink_ptitle))
                outlink_missing += 1
                continue
            fout.write("{0}\t{1}\n".format(source_pid, outlink_pid))
            if i % 500000 == 0:
                print("{0} links processed. {1} redirects followed. {2} direct hits. {3} missing (red links?). {4} redirects skipped.".format(
                    i, redirects_captured, direct_find, outlink_missing, redirects_skipped))
    print("Complete: {0} links processed. {1} redirects followed. {2} direct hits. {3} missing (red links?). {4} redirects skipped.".format(
        i, redirects_captured, direct_find, outlink_missing, redirects_skipped))

    # convert from each line being a source_pageID and target_pageID
    # to aggregating and deduplicating all target_pageIDs for a given source_pageID
    # resulting format: source_pageID [target_pageID_1, target_pageID_2, ..., target_pageID_n]
    # this overwrites the input file
    print("Aggregating by source PageID.")
    aggregate_outlinks(args.output_outlinks_tsv)

    if args.pid_to_qid_fn:
        # convert pageIDs to QIDs (for language-agnostic topic modeling)
        print("Converting from pageIDs -> QIDs")
        output_qid_links_fn = args.output_outlinks_tsv.replace(".tsv.bz2", "_qids.tsv.bz2")
        pid_to_qid(args.output_outlinks_tsv, args.pid_to_qid_fn, output_qid_links_fn, '{0}wiki'.format(args.lang))

        if args.qid_to_topics_fn and args.fasttext_data_dir:
            # convert into fastText format + train/val/test splits for modeling
            print("Splitting/reformatting for fastText.")
            qid_to_fasttext(output_qid_links_fn, args.qid_to_topics_fn,
                            os.path.join(args.fasttext_data_dir, '{0}.txt'.format(args.lang)))

def aggregate_outlinks(outlinks_fn):
    """Aggregate one outlink per line to one source page per line."""
    outlinks = {}
    with bz2.open(outlinks_fn, 'rt') as fin:
        for line in fin:
            src_pid, target_pid = line.strip().split('\t')
            src_pid = int(src_pid)
            target_pid = int(target_pid)
            if src_pid not in outlinks:
                outlinks[src_pid] = list()
            outlinks[src_pid].append(target_pid)
    with bz2.open(outlinks_fn, 'wt') as fout:
        for src_pid in outlinks:
            fout.write('{0}\t{1}\n'.format(src_pid, sorted(set(outlinks[src_pid]))))

def aggregate_outlinks_en(outlinks_fn):
    """If aggregation is too memory-intensive, this is slower but only handles 10% of the data at a time."""
    for i in range(0, 10):
        print("Processing outlinks that end with {0}".format(i))
        outlinks = {}
        with bz2.open(outlinks_fn, 'rt') as fin:
            for line in fin:
                src_pid, target_pid = line.strip().split('\t')
                if src_pid.endswith(str(i)):
                    src_pid = int(src_pid)
                    target_pid = int(target_pid)
                    if src_pid not in outlinks:
                        outlinks[src_pid] = set()
                    outlinks[src_pid].add(target_pid)
        print("{0} pages processed.".format(len(outlinks)))
        with bz2.open(outlinks_fn.replace('.tsv.bz2', '_{0}.tsv.bz2'.format(i)), 'wt') as fout:
            for src_pid in outlinks:
                fout.write('{0}\t{1}\n'.format(src_pid, list(outlinks[src_pid])))

    with bz2.open(outlinks_fn, 'wt') as fout:
        print("Combining now into single file again.")
        for i in range(0, 10):
            print("Processing outlinks that end with {0}".format(i))
            with bz2.open(outlinks_fn.replace('.tsv.bz2', '_{0}.tsv.bz2'.format(i)), 'rt') as fin:
                for line in fin:
                    fout.write(line)

def pid_to_qid(links_fn, pid_to_qid_fn, output_qid_links_fn, wiki_db):
    """Convert file with outlinks in pageID format to QID format."""
    qids = {}
    with bz2.open(pid_to_qid_fn, 'rt') as fin:
        tsvreader = csv.reader(fin, delimiter='\t')
        assert next(tsvreader) == ['item_id', 'page_id', 'wiki_db']  # item_id first but malformed bz2 file...
        for i, line in enumerate(tsvreader, start=1):
            try:
                qid = line[0]
                pid = int(line[1])
                wdb = line[2]
            except Exception:
                print("Error (line no. {0}: {1}".format(i, line))
                continue
            if not qid.startswith('Q'):
                print("Malformed line: {0}".format(line))
            if wdb == wiki_db:
                qids[pid] = qid
    print("{0} QIDs for {1}".format(len(qids), wiki_db))
    src_missing = 0
    tar_missing = 0
    with bz2.open(links_fn, 'rt') as fin:
        tsvreader = csv.reader(fin, delimiter='\t')
        with bz2.open(output_qid_links_fn, 'wt') as fout:
            tsvwriter = csv.writer(fout, delimiter='\t')
            for line in tsvreader:
                src_pid = int(line[0])
                target_pids = line[1]
                if src_pid in qids:
                    src_qid = qids[src_pid]
                    target_pids = eval(target_pids)
                    target_qids = []
                    for p in target_pids:
                        p = int(p)
                        if p in qids:
                            target_qids.append(qids[p])
                        else:
                            tar_missing += 1
                    if target_qids:
                        tsvwriter.writerow([src_qid, target_qids])
                else:
                    src_missing += 1

def get_links_all(lang, sqldate, ns=0):
    """For a given wiki, generate all links of form source_pageID -> target_pageTitle where both source and target are in namespace `ns`"""
    prefix = 'INSERT INTO `pagelinks` VALUES '
    postfix = ';\n'
    dump_fn = '/mnt/data/xmldatadumps/public/{0}wiki/{1}/{0}wiki-{1}-pagelinks.sql.gz'.format(lang, sqldate)
    checked = 0
    yielded = 0
    blocks = 0
    with gzip.open(dump_fn, 'rt', errors='replace') as fin:
        for line in fin:
            if line.startswith('INSERT INTO'):
                blocks += 1
                if not line.startswith(prefix):
                    print("Invalid line (#{0}): {1}...".format(blocks, line[:200]))
                line = line[len(prefix):-len(postfix)].replace('NULL','None')
                pages = eval(line)
                checked += len(pages)
                for p_from_id, p_to_ns, p_to_title, p_from_ns in pages:
                    if p_to_ns == ns and p_from_ns == ns:
                        yielded += 1
                        yield (p_from_id, p_to_title)

                #if blocks % 10000 == 0:
                #    print("{0} blocks. {1} link pairs. {2} originated from ns {3}.".format(blocks, checked, yielded, ns))
    print("Complete with pagelinks dump: {0} blocks. {1} link pairs. {2} originated from ns {3}.".format(blocks, checked, yielded, ns))

def get_redirects(lang, sqldate, ns=0):
    """For a given language, get all redirects in namespace `ns`"""
    prefix = 'INSERT INTO `redirect` VALUES '
    postfix = ';\n'
    redirects = {}
    dump_fn =  '/mnt/data/xmldatadumps/public/{0}wiki/{1}/{0}wiki-{1}-redirect.sql.gz'.format(lang, sqldate)
    title_matches = 0
    skipped_out_of_ns = 0
    with gzip.open(dump_fn, 'rt', errors='replace') as fin:
        for i, line in enumerate(fin, start=1):
            if line.startswith('INSERT INTO'):
                if not line.startswith(prefix):
                    print("Invalid line (#{0}): {1}...".format(i, line[:200]))
                line = line[len(prefix):-len(postfix)].replace('NULL','None')
                pages = eval(line)
                for rd_from_pid, rd_from_ns, rd_title, rd_interwiki, rd_fragment in pages:
                    if rd_from_ns != ns:
                        skipped_out_of_ns += 1
                    else:
                        title_matches += 1
                        redirects[rd_from_pid] = rd_title
    print("{0} redirects and {1} skipped for being out of ns {2}".format(title_matches, skipped_out_of_ns, ns))
    return redirects

def get_pid_to_ptitle(lang, sqldate, ns=0):
    """For a given language, get all pageID -> pageTitle pairs in namespace `ns`"""
    prefix = 'INSERT INTO `page` VALUES '
    postfix = ';\n'
    pid_to_title = {}
    dump_fn =  '/mnt/data/xmldatadumps/public/{0}wiki/{1}/{0}wiki-{1}-page.sql.gz'.format(lang, sqldate)
    with gzip.open(dump_fn, 'rt', errors='replace') as fin:
        for i, line in enumerate(fin, start=1):
            if line.startswith('INSERT INTO'):
                if not line.startswith(prefix):
                    print("Invalid line (#{0}): {1}...".format(i, line[:200]))
                line = line[len(prefix):-len(postfix)].replace("NULL", "None")
                pages = eval(line)
                for pid, pns, title, rstrctns, rdrct, new, rnd, tched, lnks_upd, ltst, plngth, p_cont_mdl, p_lng in pages:
                    if pns == ns:
                        pid_to_title[pid] = title
    print("{0} page IDs found.".format(len(pid_to_title)))
    return pid_to_title

def fasttextify(topic):
    """Translate articletopic labels into fastText format (prefixed with __label__ and no spaces)."""
    return '__label__{0}'.format(topic.replace(' ', '_'))

def qid_to_fasttext(qid_links_fn, qid_topics_fn, base_fasttext_fn):
    """Reformat for fastText and split into train/val/test."""
    train_prop = 0.9
    val_prop = 0.02
    test_prop = 0.08
    assert train_prop + val_prop + test_prop == 1
    train_fn = base_fasttext_fn.replace('.txt', '_train.txt')
    train_metadata_fn = base_fasttext_fn.replace('.txt', '_train_metadata.txt')
    val_fn = base_fasttext_fn.replace('.txt', '_val.txt')
    val_metadata_fn = base_fasttext_fn.replace('.txt', '_val_metadata.txt')
    test_fn = base_fasttext_fn.replace('.txt', '_test.txt')
    test_metadata_fn = base_fasttext_fn.replace('.txt', '_test_metadata.txt')
    qid_topics = {}
    with bz2.open(qid_topics_fn, 'rt') as fin:
        csvreader = csv.reader(fin)
        assert next(csvreader) == ['qid', 'topics']
        for line in csvreader:
            qid = line[0]
            topics = eval(line[1])
            qid_topics[qid] = topics
    print("{0} QIDs with topics.".format(len(qid_topics)))
    train_written = 0
    val_written = 0
    test_written = 0
    i = 0
    with bz2.open(qid_links_fn, 'rt') as fin:
        tsvreader = csv.reader(fin, delimiter='\t')
        with open(train_fn, 'w') as train_fout:
            with open(train_metadata_fn, 'w') as train_metadata_fout:
                with open(val_fn, 'w') as val_fout:
                    with open(val_metadata_fn, 'w') as val_metadata_fout:
                        with open(test_fn, 'w') as test_fout:
                            with open(test_metadata_fn, 'w') as test_metadata_fout:
                                for i, line in enumerate(tsvreader, start=1):
                                    src_qid = line[0]
                                    if src_qid in qid_topics:
                                        topics = qid_topics[src_qid]
                                        outlinks = eval(line[1])
                                        r = random.random()
                                        if r <= train_prop:
                                            data_fout = train_fout
                                            metadata_fout = train_metadata_fout
                                            train_written += 1
                                        elif r <= train_prop + val_prop:
                                            data_fout = val_fout
                                            metadata_fout = val_metadata_fout
                                            val_written += 1
                                        else:
                                            data_fout = test_fout
                                            metadata_fout = test_metadata_fout
                                            test_written += 1
                                        data_fout.write('{0} {1}\n'.format(' '.join([fasttextify(t) for t in topics]), ' '.join(outlinks)))
                                        metadata_fout.write('{0}\n'.format(src_qid))
    print("{0} total: {1} train. {2} val. {3} test.".format(i, train_written, val_written, test_written))

def mixed_training(lang_props):
    """Utility to make mixed fastText training files for training mixed language models.
    lang_props of format {'en':0.5, 'es':0.5}, which will include 1/2 the English data and 1/2 the Spanish data
    and not make a file that is 50-50 english-spanish (that would probably require something like {'en':0.05, 'es':0.5}
    """
    langs_fn = '_'.join([l for l in lang_props])
    with open('{0}_train.txt'.format(langs_fn), 'w') as fout_data:
        with open('{0}_train_metadata'.format(langs_fn), 'w') as fout_metadata:
            fins = {}
            for l in lang_props:
                fins['{0}_data'.format(l)] = open('{0}_train.txt'.format(l), 'r')
                fins['{0}_metadata'.format(l)] = open('{0}_train_metadata.txt'.format(l), 'r')
                while True:
                    failed = 0
                    for l in lang_props:
                        try:
                            data = next(fins['{0}_data'.format(l)])
                            metadata = next(fins['{0}_metadata'.format(l)])
                            if random.random() < lang_props[l]:
                                fout_data.write(data)
                                fout_metadata.write(metadata.replace('\n', '\t{0}\n'.format(l)))
                        except:
                            failed += 1
                        if failed == len(lang_props):
                            break

if __name__ == "__main__":
    main()
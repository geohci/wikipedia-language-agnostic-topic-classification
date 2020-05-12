import argparse
import bz2
import csv
from collections import defaultdict
import json
import yaml

# NOTE:
# the taxonomy YAML can be downloaded at: https://ndownloader.figshare.com/files/20205654
# the enwiki_wp_templates JSON can be downloaded at: https://ndownloader.figshare.com/files/20183063
# Full figshare item: https://figshare.com/articles/Wikipedia_Articles_and_Associated_WikiProject_Templates/10248344

def generate_wp_to_labels(wp_taxonomy):
    wp_to_labels = defaultdict(set)
    for wikiproject_name, label in _invert_wp_taxonomy(wp_taxonomy):
        wp_to_labels[wikiproject_name].add(label)
    return wp_to_labels


def _invert_wp_taxonomy(wp_taxonomy, path=None):
    catch_all = None
    catch_all_wikiprojects = []
    for key, value in wp_taxonomy.items():
        path_keys = (path or []) + [key]
        if key[-1] == "*":
            # this is a catch-all
            catch_all = path_keys
            catch_all_wikiprojects.extend(value)
            continue
        elif isinstance(value, list):
            catch_all_wikiprojects.extend(value)
            for wikiproject_name in value:
                yield wikiproject_name, ".".join(path_keys)
        else:
            yield from _invert_wp_taxonomy(value, path=path_keys)
    if catch_all is not None:
        for wikiproject_name in catch_all_wikiprojects:
            yield wikiproject_name, ".".join(catch_all)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxonomy_yaml",
                        help="Mapping of WikiProject templates -> topics.")
    parser.add_argument("--enwiki_wp_templates",
                        help="Dataset with QIDs and associated WikiProject templates from English Wikipedia.")
    parser.add_argument("--output_tsv", help="Output Bzipped TSV file with each QID and associated topics.")
    args = parser.parse_args()

    with open(args.taxonomy_yaml, 'r') as fin:
        taxonomy = yaml.safe_load(fin)

    wikiproject_to_topic = generate_wp_to_labels(taxonomy)
    topics = set()
    for wp in wikiproject_to_topic:
        for topic in wikiproject_to_topic[wp]:
            topics.add(topic)
    print("{0} WikiProjects and {1} topics".format(len(wikiproject_to_topic), len(topics)))

    qids_with_topics = 0
    with bz2.open(args.enwiki_wp_templates, 'rt') as fin:
        with bz2.open(args.output_tsv, 'wt') as fout:
            tsvwriter = csv.writer(fout, delimiter='\t')
            tsvwriter.writerow(['qid', 'topics'])
            for line in fin:
                l = json.loads(line)
                qid = l['qid']
                topics = set()
                for wpt in l['wp_templates']:
                    if wpt in wikiproject_to_topic:
                        for t in wikiproject_to_topic[wpt]:
                            topics.add(t)
                if topics:
                    tsvwriter.writerow([qid, topics])
                    qids_with_topics += 1
    print("Wrote {0} QIDs with topics to {1}.".format(qids_with_topics, args.output_tsv))

if __name__ == "__main__":
    main()
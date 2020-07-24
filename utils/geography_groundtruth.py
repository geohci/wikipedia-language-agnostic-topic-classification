import argparse
import bz2
import csv
import json

import mwapi
from mwapi.errors import APIError
from shapely.geometry import shape, Point

"""
# SPARQL query for list of present-day countries used to get list of countries
SELECT DISTINCT ?country ?countryLabel
WHERE
{
  ?country wdt:P31 wd:Q3624078 .
  #not a former country
  FILTER NOT EXISTS {?country wdt:P31 wd:Q3024240}
  #and no an ancient civilisation (needed to exclude ancient Egypt)
  FILTER NOT EXISTS {?country wdt:P31 wd:Q28171280}

  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
}
ORDER BY ?countryLabel
"""


COUNTRY_PROPERTIES = [('P19','place of birth'),
                      ('P17','country'),
                      ('P27','country of citizenship')]
CONTINENT_PROPERTIES = [('P30','continent')]

def get_label(qid, session=None):
    if session is None:
        session = mwapi.Session('https://wikidata.org', user_agent='geography script -- isaac@wikimedia.org')
    try:
        result = session.get(action='wbgetentities', ids=qid, props='labels', languages='en')
        lbl = result['entities'][qid]['labels']['en']['value']
    except (KeyError, APIError):
        lbl = qid
    return lbl

def get_continent_data(continent_geoms):
    qid_to_continent = {'Q15':'Africa',
                        'Q51':'Antarctica',
                        'Q48':'Asia',
                        'Q3960':'Oceania',  # Australia
                        'Q538':'Oceania',  # Oceania
                        'Q46':'Europe',
                        'Q49':'North America',
                        'Q18':'South America'}
    print("Loaded {0} QID-continent pairs -- e.g., Q48 is {1}".format(len(qid_to_continent), qid_to_continent['Q48']))
    # https://gist.githubusercontent.com/hrbrmstr/91ea5cc9474286c72838/raw/59421ff9b268ff0929b051ddafafbeb94a4c1910/continents.json
    with open(continent_geoms, 'r') as fin:
        continents = json.load(fin)['features']
    continent_shapes = {}
    for c in continents:
        continent_name = c['properties']['CONTINENT']
        if continent_name in qid_to_continent.values() or continent_name == 'Australia':
            continent_shapes[continent_name] = shape(c['geometry'])
    print("Loaded {0} continent geometries".format(len(continent_shapes)))
    return continent_shapes, qid_to_continent

def get_country_data(sparql_tsv, country_geoms):
    qid_to_country = {}
    countries_to_skip = ['Sixth Republic of South Korea', 'Tuvalu']
    country_replacements = {'French Fifth Republic': 'France',
                            'Kingdom of Denmark': 'Denmark',
                            'Kingdom of the Netherlands': 'Netherlands'}
    with open(sparql_tsv, 'r') as fin:
        tsvreader = csv.reader(fin, delimiter='\t')
        assert next(tsvreader) == ['country', 'countryLabel']
        prefix = 'http://www.wikidata.org/entity/'
        start_idx = len(prefix)
        for line in tsvreader:
            qid = line[0][start_idx:]
            country = line[1]
            if country in country_replacements:
                country = country_replacements[country]
            if country not in countries_to_skip:
                qid_to_country[qid] = country
    print("Loaded {0} QID-country pairs -- e.g., Q31 is {1}".format(len(qid_to_country), qid_to_country['Q31']))
    all_countries = {c for c in qid_to_country.values()}
    # load in geometries for the countries identified via Wikidata
    country_replacements = {'Somaliland': 'Somalia',
                            'eSwatini': 'Eswatini',
                            'Turkish Republic of Northern Cyprus': 'Cyprus',
                            'Kosovo': 'Serbia',
                            'Republic of Macedonia': 'North Macedonia'}
    with open(country_geoms, 'r') as fin:
        countries = json.load(fin)['features']
    country_shapes = {}
    for c in countries:
        country_name = c['properties']['NAME_EN']
        if country_name in country_replacements:
            country_name = country_replacements[country_name]
        if country_name in all_countries:
            country_shapes[country_name] = shape(c['geometry'])
    print("Loaded {0} country geometries".format(len(country_shapes)))
    return country_shapes, qid_to_country

def main():
    """Get Wikidata country properties for all Wikipedia articles.
    Track which property they came from to compare later.
    Save the resulting dictionary to a TSV file. This process takes ~24 hours.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_tsv', help="Output TSV with data.")
    parser.add_argument('--sparql_tsv', help="TSV with countries and corresponding Wikidata IDs")
    parser.add_argument('--country_geoms', help='GeoJSON with country shapes for locating coordinates.')
    parser.add_argument('--continent_geoms', help='GeoJSON with continent shapes for locating coordinates.')
    args = parser.parse_args()
    # load in list of continents per Wikidata
    continent_shapes, qid_to_continent = get_continent_data(args.continent_geoms)
    # load in list of countries per Wikidata
    country_shapes, qid_to_country = get_country_data(args.sparql_tsv, args.country_geoms)
    # process Wikidata dump
    dump_fn = '/mnt/data/xmldatadumps/public/wikidatawiki/entities/latest/wikidata-latest-all.json.bz2'
    print("Building QID->properties map from {0}".format(dump_fn))
    items_written = 0
    num_has_coords = 0
    coords_mapped_to_country = 0
    coords_mapped_to_continent = 0
    poorly_formatted = 0
    qids_skipped_over = {}
    with open(args.output_tsv, 'w') as fout:
        print("Writing QID->Props to {0}".format(args.output_tsv))
        header = ['item', 'en_title', 'lat', 'lon', 'P625_country', 'P625_continent']
        for _, lbl in COUNTRY_PROPERTIES:
            header.append(lbl)
        for _, lbl in CONTINENT_PROPERTIES:
            header.append(lbl)
        tsvwriter = csv.DictWriter(fout, delimiter='\t', fieldnames=header)
        tsvwriter.writeheader()
        with bz2.open(dump_fn, 'rt') as fin:
            next(fin)
            for idx, line in enumerate(fin, start=1):
                try:
                    item_json = json.loads(line[:-2])
                except Exception:
                    try:
                        item_json = json.loads(line)
                    except Exception:
                        print("Error:", idx, line)
                        continue
                if idx % 100000 == 0:
                    print("{0} lines processed. {1} ({2:.3f}) items kept, {3} ({4:.3f}) w/ coordinates.".format(
                        idx, items_written, items_written / idx, num_has_coords, num_has_coords / items_written))
                qid = item_json.get('id', None)
                if not qid:
                    continue
                sitelinks = [l[:-4] for l in item_json.get('sitelinks', []) if l.endswith('wiki') and l != 'commonswiki' and l != 'specieswiki']
                # check that at least one wikipedia in list
                if not sitelinks:
                    continue
                en_title = item_json.get('sitelinks', {}).get('enwiki', {}).get('title', None)
                claims = item_json.get('claims', {})
                output_row = {'item':qid, 'en_title':en_title}
                try:
                    lat = claims['P625'][0]['mainsnak']['datavalue']['value']['latitude']
                    lon = claims['P625'][0]['mainsnak']['datavalue']['value']['longitude']
                    output_row['lat'] = lat
                    output_row['lon'] = lon
                    num_has_coords += 1
                    pt = (lon, lat)
                    for c in country_shapes:
                        if country_shapes[c].contains(pt):
                            output_row['P625_country'] = c
                            coords_mapped_to_country += 1
                            break
                    for c in continent_shapes:
                        if continent_shapes[c].contains(pt):
                            if c == 'Australia':
                                output_row['P625_continent'] = 'Oceania'
                            else:
                                output_row['P625_continent'] = c
                            coords_mapped_to_continent += 1
                            break
                except KeyError:
                    pass
                for prop, lbl in COUNTRY_PROPERTIES:
                    if prop in claims:
                        for statement in claims[prop]:
                            try:
                                value_qid = statement['mainsnak']['datavalue']['value']['id']
                            except KeyError:
                                poorly_formatted += 1
                                continue
                            if value_qid in qid_to_country:
                                output_row[lbl] = output_row.get(lbl, []) + [qid_to_country[value_qid]]
                            else:
                                qids_skipped_over[value_qid] = qids_skipped_over.get(value_qid, 0) + 1
                for prop, lbl in CONTINENT_PROPERTIES:
                    if prop in claims:
                        for statement in claims[prop]:
                            try:
                                value_qid = statement['mainsnak']['datavalue']['value']['id']
                            except KeyError:
                                poorly_formatted += 1
                                continue
                            if value_qid in qid_to_continent:
                                output_row[lbl] = output_row.get(lbl, []) + [qid_to_continent[value_qid]]
                            else:
                                qids_skipped_over[value_qid] = qids_skipped_over.get(value_qid, 0) + 1
                tsvwriter.writerow(output_row)
                items_written += 1


if __name__ == "__main__":
    main()
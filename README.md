# Language-agnostic Topic Modeling for Wikipedia
This is a prototype for building language-agnostic topic models for Wikipedia.
The main model is built via notebooks in the `outlinks` directory.
There are also two comparison models (`wikidata` and `wikitext`).
Additionally, the `labels` folder has a script for building a mapping of
articles to topics via WikiProject tags and a mapping of WikiProject -> topics.

## Models
* `outlinks`: language-agnostic and article-specific. An article is represented by its outlinks, which are represented by their associated Wikidata items.
    * For example, [en:City Bureau](https://en.wikipedia.org/wiki/City_Bureau) -> `(en:non-profit organization, en:South Side, ..., )` -> `(Q163740, Q3492277, ...)`
* `wikitext`: language-dependent and article-specific. An article is represented by the words in it. Similar to [ORES model](https://github.com/wikimedia/drafttopic).
    * For example, [en:City Bureau](https://en.wikipedia.org/wiki/City_Bureau) -> `(city bureau is an american non-profit organization based in south side of chicago...)`
* `wikidata`: language-agnostic but predictions made at level of Wikidata item, not Wikipedia article. An article is represented by the properties and values in its associated Wikidata item.
    * For example, [en:City Bureau](https://en.wikipedia.org/wiki/City_Bureau) -> [wd:Q72043226](https://www.wikidata.org/wiki/Q72043226) -> `(P31, Q18325436, P571, P17, ..., P2002)`
    
## Groundtruth
The groundtruth labels for these models all come from WikiProject tagging of English Wikipedia articles and [a mapping of those WikiProjects to the the topic taxonomy](https://github.com/wikimedia/wikitax/tree/master/taxonomies/wikiproject/halfak_20191202).
These assessments are most easily accessed via the [PageAssessments extension](https://www.mediawiki.org/wiki/Extension:PageAssessments), which generates database tables with the tags.
These tables are not included in any public dumps but can be accessed in a variety of ways via [replicas](https://wikitech.wikimedia.org/wiki/Wiki_Replicas):
* [Quarry](https://meta.wikimedia.org/wiki/Research:Quarry): e.g., [extract just the articles in the Climate Change WikiProject](https://quarry.wmflabs.org/query/52210).
* [PAWS](https://wikitech.wikimedia.org/wiki/PAWS): given the size of the table, the query must be broken up but XikunHuang [developed a notebook that can extract this data](https://public.paws.wmcloud.org/User:Huangxk/export_tables.ipynb).

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
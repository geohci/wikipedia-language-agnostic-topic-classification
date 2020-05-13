# Language-agnostic Topic Modeling for Wikipedia
This is a prototype for building language-agnostic topic models for Wikipedia,
where an article is represented by its outlinks and the vocabulary is in Wikidata IDs,
so a model trained on any language version of Wikipedia can be used to predict topic labels for any other
language version of Wikipedia without re-training (though in practice, that would be advisable).

## Scripts
* `build_outlinks_representation.py`: for each article in a given Wikipedia language edition, gather all of its outlinks
to articles in namespace 0 (i.e. no templates/categories/etc.). Resolve any redirects and map the outlinks to 
their corresponding Wikidata IDs. Add in topic labels and convert this feature representation to fastText format.
* `drafttopic_article_fasttext_model.py`: learn a simple fastText supervised multilabel model for predicting an article's 
topics based on the bag-of-Wikidata-outlinks representation.
* `utils/build_qid_topic_dump.py`: build the QID -> topic mapping needed for building labeled fastText data.

## Resources
* [topic labels (figshare)](https://figshare.com/articles/Wikipedia_Articles_and_Associated_WikiProject_Templates/10248344)
* [link data (dumps)](https://www.mediawiki.org/wiki/Manual:Pagelinks_table)
* [redirect data (dumps)](https://www.mediawiki.org/wiki/Manual:Redirect_table)
* [page data (dumps)](https://www.mediawiki.org/wiki/Manual:Page_table)
* [wikidata ID mapping (hive)](https://wikitech.wikimedia.org/wiki/Analytics/Data_Lake/Edits/Wikidata_item_page_link)
* See [spark code](https://github.com/wikimedia/research-tutorials/blob/master/access-data_link-graph-via-spark.ipynb) for producing these link graphs. As of May 2020, code does not resolve redirects though.
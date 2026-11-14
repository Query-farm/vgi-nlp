"""Composed example queries: several of the worker's objects in one realistic query.

Each example below is defined exactly once and referenced from both the schema-level
``vgi.example_queries`` list (``vgi_nlp/worker.py``) and the example list of the
object that drives the query (so agents browsing that function see it among its
first five examples). Every query is self-contained -- it builds its input relation
inline with ``VALUES`` -- and is offline and deterministic (pinned ``lang``, fixed
fastText / spaCy / VADER models); ``test/sql/composed.test`` asserts their rows.
"""

from __future__ import annotations

from vgi.metadata import FunctionExample

#: sentences + sentiment: the most negative sentence of each review.
WORST_SENTENCE_PER_REVIEW = FunctionExample(
    description=(
        "Split each review into sentences and score every sentence's sentiment to find the single most "
        "negative sentence per review -- pinpointing the complaint inside otherwise mixed feedback."
    ),
    sql=(
        "WITH reviews(id, body) AS (VALUES "
        "(1, 'The hotel location was perfect. The room was filthy and the staff were rude. Breakfast was fine.'), "
        "(2, 'Checkout was a frustrating mess. Otherwise it was a wonderful stay with a lovely view!')) "
        "SELECT id, arg_min(sentence, nlp.main.sentiment(sentence)) AS worst_sentence, "
        "round(min(nlp.main.sentiment(sentence)), 3) AS worst_score "
        "FROM nlp.main.sentences((SELECT id, body FROM reviews), id := 'id', lang := 'en') "
        "GROUP BY id ORDER BY id"
    ),
)

#: entities + sentiment: average review sentiment per organization mentioned.
SENTIMENT_PER_ORGANIZATION = FunctionExample(
    description=(
        "Extract the organizations (ORG entities) mentioned in each review, join them back to their "
        "review by id, and average the review sentiment per organization -- brand perception from free text."
    ),
    sql=(
        "WITH reviews(id, body) AS (VALUES "
        "(1, 'I love my new Apple laptop, the battery life is amazing.'), "
        "(2, 'Microsoft support was awful and rude.'), "
        "(3, 'The Apple store staff were friendly and quick to help.'), "
        "(4, 'Microsoft keeps crashing my laptop, which is incredibly frustrating.')) "
        "SELECT e.ent_text AS organization, count(*) AS mentions, "
        "round(avg(nlp.main.sentiment(r.body)), 3) AS avg_sentiment "
        "FROM nlp.main.entities((SELECT id, body FROM reviews), id := 'id', lang := 'en') AS e "
        "JOIN reviews AS r USING (id) "
        "WHERE e.label = 'ORG' "
        "GROUP BY e.ent_text ORDER BY avg_sentiment DESC"
    ),
)

#: detect_lang + supported_languages: route a multilingual corpus to the spaCy pipelines.
LANGUAGE_ROUTING = FunctionExample(
    description=(
        "Detect each document's language and look it up in the supported_languages table to see which "
        "default spaCy pipeline would process it -- documents in a language without one (here Swahili) "
        "get NULL and cannot be tokenized or run through NER without naming a model."
    ),
    sql=(
        "WITH docs(id, body) AS (VALUES "
        "(1, 'The quarterly report shows strong growth in every region.'), "
        "(2, 'Le rapport trimestriel montre une forte croissance dans toutes les régions.'), "
        "(3, 'Ripoti ya robo mwaka inaonyesha ukuaji mkubwa katika kila mkoa.')), "
        "detected AS (SELECT id, nlp.main.detect_lang(body) AS lang FROM docs) "
        "SELECT d.id, d.lang, s.spacy_model, s.spacy_model IS NOT NULL AS has_default_pipeline "
        "FROM detected AS d LEFT JOIN nlp.main.supported_languages AS s ON s.lang_code = d.lang "
        "ORDER BY d.id"
    ),
)

#: noun_chunks + sentiment_label + lemmatize: what negative reviews complain about.
NEGATIVE_REVIEW_ASPECTS = FunctionExample(
    description=(
        "Keep only reviews labelled negative, extract their noun chunks, and lemmatize each chunk's head "
        "noun so 'rooms' and 'room' count together -- ranking the aspects that negative reviews mention."
    ),
    sql=(
        "WITH reviews(id, body) AS (VALUES "
        "(1, 'The rooms were dirty and the beds were terrible.'), "
        "(2, 'Awful service, and the room smelled bad.'), "
        "(3, 'The beds were comfortable and the view was great.'), "
        "(4, 'The breakfast was cold and the coffee was disgusting.')) "
        "SELECT nlp.main.lemmatize(lower(root), 'en') AS aspect, count(DISTINCT id) AS negative_reviews "
        "FROM nlp.main.noun_chunks("
        "(SELECT id, body FROM reviews WHERE nlp.main.sentiment_label(body) = 'neg'), id := 'id', lang := 'en') "
        "GROUP BY aspect ORDER BY negative_reviews DESC, aspect"
    ),
)

#: Every composed example, in schema-level display order.
COMPOSED_EXAMPLES: list[FunctionExample] = [
    WORST_SENTENCE_PER_REVIEW,
    SENTIMENT_PER_ORGANIZATION,
    LANGUAGE_ROUTING,
    NEGATIVE_REVIEW_ASPECTS,
]

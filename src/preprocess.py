"""Text preprocessing: tokenization, normalization, stopwords, stemming.

Every piece of text in the system -- document titles, abstracts, and the
user's queries -- passes through this one pipeline. That single fact is
what makes retrieval work at all: a query can only match a document if
both were reduced to the same vocabulary. Preprocessing queries
differently from documents is the classic way search engines silently
break, so this module is the only place in the codebase where text is
turned into terms.

The pipeline has three stages:

1. Tokenization + case folding. Lowercase the text, then extract maximal
   runs of letters/digits. Punctuation, LaTeX markup and symbols act as
   separators, so "State-of-the-Art" yields four tokens and "BM25" stays
   one token.
2. Stopword removal. Function words like "the", "of", "and" occur in
   nearly every document, so they carry no discriminating power for
   retrieval -- they only inflate the index. We drop them using the
   classic English stopword list.
3. Stemming. Porter's rule-based suffix-stripping algorithm collapses
   morphological variants ("network", "networks", "networking") into one
   index term. This trades a sliver of precision for a solid recall gain,
   a good trade on abstracts where authors phrase the same concept many
   different ways.
"""

import re

from nltk.stem import PorterStemmer

# Maximal runs of ASCII letters/digits on lowercased text.
TOKEN_RE = re.compile(r"[a-z0-9]+")

# Classic English stopword list (function words only -- no domain terms).
STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be
    because been before being below between both but by can cannot could did
    do does doing down during each few for from further had has have
    having he her here hers herself him himself his how i if in into is
    it its itself just me more most my myself no nor not now of off on
    once only or other our ours ourselves out over own same she should
    so some such than that the their theirs them themselves then there
    these they this those through to too under until up very was we were
    what when where which while who whom why will with would you your
    yours yourself yourselves
    """.split()
)

_stemmer = PorterStemmer()

# Stemming is by far the slowest stage, but a corpus has few *unique*
# tokens relative to total tokens, so caching stems by token makes
# indexing ~10x faster at negligible memory cost.
_stem_cache: dict[str, str] = {}


def _stem(token: str) -> str:
    stemmed = _stem_cache.get(token)
    if stemmed is None:
        stemmed = _stemmer.stem(token)
        _stem_cache[token] = stemmed
    return stemmed


def tokenize(text: str) -> list[str]:
    """Lowercase and split into alphanumeric tokens."""
    return TOKEN_RE.findall((text or "").lower())


def preprocess(
    text: str,
    *,
    remove_stopwords: bool = True,
    stem: bool = True,
) -> list[str]:
    """Run the full pipeline: tokenize -> drop stopwords -> stem.

    The flags exist so later stages (e.g. evaluation ablations, snippet
    display) can switch off individual steps, but documents and queries
    must always use the *same* settings.
    """
    tokens = tokenize(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if stem:
        tokens = [_stem(t) for t in tokens]
    return tokens

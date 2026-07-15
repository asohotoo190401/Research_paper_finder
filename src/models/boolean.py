"""Boolean retrieval: the oldest IR model, and the baseline for ours.

A Boolean query combines terms with the operators AND, OR, NOT and
parentheses; a document either matches or it doesn't -- there is no
ranking. The model is precise and fully explainable, which is why it
survives in legal/systematic-review search, but its all-or-nothing
matching is exactly what motivates the ranked models we add on Day 3.

Query language (operators must be UPPERCASE to distinguish them from
ordinary words -- lowercase "and"/"or"/"not" are stopwords and simply
disappear during preprocessing):

    graph AND neural
    bert OR roberta
    diffusion AND NOT image
    (privacy OR federated) AND learning
    transformer efficient          <- adjacent terms imply AND

Implementation: the query string is tokenized, then parsed by a small
recursive-descent parser into an abstract syntax tree following the
usual precedence NOT > AND > OR (so "a OR b AND c" means "a OR (b AND
c)"). Each leaf term is passed through the SAME preprocessing pipeline
as the documents -- so a query for "networks" finds documents indexed
under the stem "network". Evaluation then maps leaves to posting sets
and combines them with set intersection, union and complement.
"""

import re

from ..indexing import InvertedIndex
from ..preprocess import preprocess

OPERATORS = {"AND", "OR", "NOT"}

# A query token is a parenthesis or a run of anything that isn't
# whitespace or a parenthesis.
_QUERY_TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")


class QuerySyntaxError(ValueError):
    """Raised when a query cannot be parsed (e.g. unbalanced parentheses)."""


def _lex(query: str) -> list[str]:
    return _QUERY_TOKEN_RE.findall(query)


class _Parser:
    """Recursive-descent parser producing an AST of nested tuples.

    Node shapes: ("term", str) | ("and", [nodes]) | ("or", [nodes])
    | ("not", node) | None. A None node means the atom vanished during
    preprocessing (it was a stopword or pure punctuation); AND/OR simply
    ignore None operands, so "the transformer" behaves like "transformer".
    """

    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _next(self) -> str:
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def parse(self):
        if not self.tokens:
            return None
        node = self._parse_or()
        if self._peek() is not None:
            raise QuerySyntaxError(f"unexpected {self._peek()!r} in query")
        return node

    def _parse_or(self):
        children = [self._parse_and()]
        while self._peek() == "OR":
            self._next()
            children.append(self._parse_and())
        return _combine("or", children)

    def _parse_and(self):
        children = [self._parse_not()]
        while True:
            nxt = self._peek()
            if nxt == "AND":
                self._next()
                children.append(self._parse_not())
            elif nxt is not None and nxt not in ("OR", ")"):
                # Adjacent atom (term, "(" or NOT) -> implicit AND.
                children.append(self._parse_not())
            else:
                return _combine("and", children)

    def _parse_not(self):
        if self._peek() == "NOT":
            self._next()
            child = self._parse_not()
            return ("not", child) if child is not None else None
        return self._parse_atom()

    def _parse_atom(self):
        token = self._peek()
        if token is None or token in OPERATORS or token == ")":
            raise QuerySyntaxError(f"expected a term, found {token!r}")
        self._next()
        if token == "(":
            node = self._parse_or()
            if self._peek() != ")":
                raise QuerySyntaxError("missing closing parenthesis")
            self._next()
            return node
        # An ordinary word: run it through the document pipeline. It may
        # come back empty (stopword) or as several terms ("state-of-the-art").
        terms = preprocess(token)
        return _combine("and", [("term", t) for t in terms])


def _combine(kind: str, children: list):
    children = [c for c in children if c is not None]
    if not children:
        return None
    if len(children) == 1:
        return children[0]
    return (kind, children)


def parse_query(query: str):
    """Parse a query string into an AST (None if nothing survives)."""
    return _Parser(_lex(query)).parse()


def _evaluate(node, index: InvertedIndex) -> set[int]:
    kind = node[0]
    if kind == "term":
        return index.docs_containing(node[1])
    if kind == "not":
        return set(index.all_docs) - _evaluate(node[1], index)
    child_sets = [_evaluate(child, index) for child in node[1]]
    if kind == "and":
        # Intersect smallest-first: the result can never exceed the
        # smallest set, so this minimizes the work done.
        child_sets.sort(key=len)
        result = child_sets[0]
        for s in child_sets[1:]:
            result &= s
            if not result:
                break
        return result
    return set().union(*child_sets)  # "or"


def search(index: InvertedIndex, query: str) -> set[int]:
    """All doc ids matching the Boolean query (unranked, by definition).

    Raises QuerySyntaxError on malformed queries. A query that vanishes
    entirely during preprocessing matches nothing.
    """
    ast = parse_query(query)
    if ast is None:
        return set()
    return _evaluate(ast, index)

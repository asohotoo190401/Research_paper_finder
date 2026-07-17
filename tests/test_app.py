"""Integration tests for the Flask app via Flask's built-in test client.

Importing src.app builds the real index once (a couple of seconds);
after that, every request is served in-process with no network.
"""

import pytest

from src.app import app


@pytest.fixture(scope="module")
def client():
    app.config["TESTING"] = True
    return app.test_client()


def test_homepage_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"PaperFinder" in response.data


def test_ranked_search_returns_results(client):
    response = client.get("/?q=transformer&model=bm25")
    assert response.status_code == 200
    assert b"match" in response.data
    assert b"score" in response.data


def test_boolean_syntax_error_is_friendly(client):
    response = client.get("/?q=%28transformer&model=boolean")
    assert response.status_code == 200          # no stack trace, no 500
    assert b"Query error" in response.data


def test_unknown_model_falls_back_to_bm25(client):
    response = client.get("/?q=transformer&model=nonsense")
    assert response.status_code == 200
    assert b"match" in response.data

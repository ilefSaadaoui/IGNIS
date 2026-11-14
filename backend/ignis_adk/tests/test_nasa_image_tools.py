"""NASA Image and Video Library tool — offline (HTTP mocked)."""

import json
import pytest

from ignis.tools import nasa_tools
from ignis.tools.nasa_tools import search_nasa_images


class _Resp:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")
        self.headers = {}

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


SAMPLE = {
    "collection": {
        "metadata": {"total_hits": 2},
        "items": [
            {"href": "https://images.nasa.gov/search?id=100",
             "data": [{"nasa_id": "100", "title": "FLEX MDCA fuel reservoir", "description": "Multi-user droplet combustion apparatus aboard ISS.",
                        "date_created": "2013-05-01T00:00:00Z", "center": "GRC", "keywords": ["combustion", "FLEX"]}],
             "links": [{"href": "https://images.nasa.gov/100~preview.jpg", "rel": "preview", "render": "image"}]},
            {"href": "https://images.nasa.gov/search?id=101",
             "data": [{"nasa_id": "101", "title": "BASS-II flame spread", "date_created": "2015-09-01T00:00:00Z", "center": "GRC"}],
             "links": []},
        ],
    }
}


@pytest.fixture()
def mock_images_http(monkeypatch):
    def fake_get(url, params=None, headers=None, timeout=None):
        assert "images-api.nasa.gov" in url
        return _Resp(200, SAMPLE)
    monkeypatch.setattr(nasa_tools.requests, "get", fake_get)
    nasa_tools._NASA_IMAGES_CACHE.clear()


def test_search_nasa_images_maps_fields(mock_images_http):
    r = search_nasa_images("FLEX droplet", page_size=5)
    assert r["status"] == "completed"
    assert r["total_hits"] == 2
    first = r["items"][0]
    assert first["title"] == "FLEX MDCA fuel reservoir"
    assert first["preview_url"].endswith("~preview.jpg")
    assert first["collection_url"].startswith("https://images.nasa.gov/")
    assert "NASA Image and Video Library" in r["provenance"]


def test_search_nasa_images_validates_media_type(mock_images_http):
    r = search_nasa_images("fire", media_type=" hologram ")
    assert r["status"] == "failed"
    assert r["items"] == []


def test_search_nasa_images_http_error_is_honest(monkeypatch):
    def boom(url, params=None, headers=None, timeout=None):
        return _Resp(500, None, "oops")
    monkeypatch.setattr(nasa_tools.requests, "get", boom)
    nasa_tools._NASA_IMAGES_CACHE.clear()
    r = search_nasa_images("fire")
    assert r["status"] == "failed" and r["items"] == [] and "500" in r["errors"][0]


def test_search_nasa_images_network_failure_is_honest(monkeypatch):
    def boom(url, params=None, headers=None, timeout=None):
        import requests as real_requests
        raise real_requests.RequestException("boom")
    monkeypatch.setattr(nasa_tools.requests, "get", boom)
    nasa_tools._NASA_IMAGES_CACHE.clear()
    r = search_nasa_images("fire")
    assert r["status"] == "failed" and r["items"] == []

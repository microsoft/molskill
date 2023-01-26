import requests

from molskill.paths import DEFAULT_CHECKPOINT_REMOTE


def test_default_model_link_isup():
    r = requests.head(DEFAULT_CHECKPOINT_REMOTE)
    assert r.ok

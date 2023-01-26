import requests

from molskill.paths import DEFAULT_CHECKPOINT_REMOTE, DEFAULT_MOMENTS_REMOTE

_ALL_REMOTES = [DEFAULT_CHECKPOINT_REMOTE, DEFAULT_MOMENTS_REMOTE]


def test_default_model_link_isup():
    for remote in _ALL_REMOTES:
        r = requests.head(remote)
        assert r.ok

import hashlib
import runpy
import sys
import tarfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import pytorch_lightning as pl
import requests
import torch

import molskill.scorer as scorer_module
from molskill.helpers.download import download
from molskill.models.ranknet import LitRankNet
from molskill.scorer import MolSkillScorer

_CHECKPOINT_MEMBER = "models/default/checkpoints/last.ckpt"


@pytest.fixture
def serve_download(monkeypatch):
    def serve(data, status=200):
        def get_response(url, *, stream, timeout):
            assert stream
            assert timeout == (10, 60)
            response = requests.Response()
            response.status_code = status
            response.headers["content-length"] = str(len(data))
            response.raw = BytesIO(data)
            return response

        monkeypatch.setattr(requests, "get", get_response)

    return serve


def _model_archive(
    data, member_name=_CHECKPOINT_MEMBER, member_type=tarfile.REGTYPE, duplicate=False
):
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        checkpoint = tarfile.TarInfo(member_name)
        checkpoint.type = member_type
        if checkpoint.isfile():
            checkpoint.size = len(data)
            archive.addfile(checkpoint, BytesIO(data))
        else:
            checkpoint.linkname = "../../outside.ckpt"
            archive.addfile(checkpoint)
        if duplicate:
            archive.addfile(checkpoint, BytesIO(data))
        for name in ("../../outside.ckpt", "/outside.ckpt"):
            unrelated = tarfile.TarInfo(name)
            unrelated.size = len(data)
            archive.addfile(unrelated, BytesIO(data))
    return buffer.getvalue()


@pytest.mark.parametrize(
    "status",
    [pytest.param(202, id="challenge"), pytest.param(404, id="not-found")],
)
def test_download_rejects_unsuccessful_responses(tmp_path, serve_download, status):
    serve_download(b"", status=status)
    destination = tmp_path / "model"
    with pytest.raises(requests.HTTPError):
        download("https://example.com/model", str(destination))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "data",
    [pytest.param(b"", id="empty"), pytest.param(b"tampered", id="wrong-digest")],
)
def test_download_preserves_existing_file_on_failure(tmp_path, serve_download, data):
    destination = tmp_path / "model"
    destination.write_bytes(b"existing")
    serve_download(data)
    with pytest.raises(ValueError):
        download(
            "https://example.com/model",
            str(destination),
            sha256=hashlib.sha256(b"trusted").hexdigest(),
        )
    assert destination.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [destination]


@pytest.fixture
def default_checkpoint_path(tmp_path, monkeypatch):
    path = tmp_path / "models" / "default" / "checkpoints" / "last.ckpt"
    monkeypatch.setattr(scorer_module, "DEFAULT_CHECKPOINT_PATH", str(path))
    return path


def test_download_copies_only_verified_checkpoint(
    default_checkpoint_path, monkeypatch, serve_download
):
    data = b"trusted checkpoint"
    archive = _model_archive(data)
    serve_download(archive)
    monkeypatch.setattr(
        scorer_module,
        "DEFAULT_CHECKPOINT_ARCHIVE_SHA256",
        hashlib.sha256(archive).hexdigest(),
    )
    monkeypatch.setattr(
        scorer_module, "DEFAULT_CHECKPOINT_SHA256", hashlib.sha256(data).hexdigest()
    )

    MolSkillScorer._download_default_model()

    assert default_checkpoint_path.read_bytes() == data
    assert list(default_checkpoint_path.parent.iterdir()) == [default_checkpoint_path]
    root = default_checkpoint_path.parents[3]
    assert list(root.rglob("*ckpt")) == [default_checkpoint_path]


@pytest.mark.parametrize(
    "member_name,member_type,duplicate",
    [
        pytest.param("../last.ckpt", tarfile.REGTYPE, False, id="missing"),
        pytest.param(_CHECKPOINT_MEMBER, tarfile.SYMTYPE, False, id="symlink"),
        pytest.param(_CHECKPOINT_MEMBER, tarfile.LNKTYPE, False, id="hardlink"),
        pytest.param(_CHECKPOINT_MEMBER, tarfile.DIRTYPE, False, id="directory"),
        pytest.param(_CHECKPOINT_MEMBER, tarfile.REGTYPE, True, id="duplicate"),
    ],
)
def test_download_rejects_invalid_checkpoint_member(
    default_checkpoint_path,
    monkeypatch,
    serve_download,
    member_name,
    member_type,
    duplicate,
):
    archive = _model_archive(b"checkpoint", member_name, member_type, duplicate)
    serve_download(archive)
    monkeypatch.setattr(
        scorer_module,
        "DEFAULT_CHECKPOINT_ARCHIVE_SHA256",
        hashlib.sha256(archive).hexdigest(),
    )
    with pytest.raises(ValueError, match="one regular default checkpoint"):
        MolSkillScorer._download_default_model()
    assert list(default_checkpoint_path.parent.iterdir()) == []


@pytest.mark.parametrize(
    "invalid_digest",
    [
        pytest.param("archive", id="archive"),
        pytest.param("checkpoint", id="checkpoint"),
    ],
)
def test_download_rejects_changed_artifact(
    default_checkpoint_path, monkeypatch, serve_download, invalid_digest
):
    data = b"checkpoint"
    archive = _model_archive(data)
    serve_download(archive)
    monkeypatch.setattr(
        scorer_module,
        "DEFAULT_CHECKPOINT_ARCHIVE_SHA256",
        hashlib.sha256(
            archive if invalid_digest != "archive" else b"trusted"
        ).hexdigest(),
    )
    monkeypatch.setattr(
        scorer_module,
        "DEFAULT_CHECKPOINT_SHA256",
        hashlib.sha256(
            data if invalid_digest != "checkpoint" else b"trusted"
        ).hexdigest(),
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        MolSkillScorer._download_default_model()
    assert list(default_checkpoint_path.parent.iterdir()) == []


def test_cached_checkpoint_is_verified_before_loading(
    default_checkpoint_path, monkeypatch
):
    default_checkpoint_path.parent.mkdir(parents=True)
    default_checkpoint_path.write_bytes(b"unverified checkpoint")

    def unexpected_load(*args, **kwargs):
        pytest.fail("Unverified bytes reached the deserializer")

    monkeypatch.setattr(LitRankNet, "load_from_checkpoint", unexpected_load)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        MolSkillScorer(num_workers=0, verbose=False)


def test_verified_loader_uses_exact_verified_bytes(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "model.ckpt"
    data = b"trusted checkpoint"
    checkpoint_path.write_bytes(data)
    model = LitRankNet(input_size=4)

    def load(*, checkpoint_path: BytesIO, input_size, map_location):
        (tmp_path / "model.ckpt").write_bytes(b"replacement")
        assert checkpoint_path.read() == data
        assert input_size == 4
        assert map_location == "cpu"
        return model

    monkeypatch.setattr(LitRankNet, "load_from_checkpoint", load)
    loaded = LitRankNet.load_verified_checkpoint(
        checkpoint_path,
        hashlib.sha256(data).hexdigest(),
        input_size=4,
        map_location="cpu",
    )
    assert loaded is model


def test_verified_checkpoint_preserves_model_predictions(tmp_path):
    model = LitRankNet(input_size=4)
    checkpoint_path = tmp_path / "model.ckpt"
    torch.save(
        {"state_dict": model.state_dict(), "pytorch-lightning_version": pl.__version__},
        checkpoint_path,
    )
    loaded = LitRankNet.load_verified_checkpoint(
        checkpoint_path,
        hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        input_size=4,
        map_location="cpu",
    )
    inputs = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    torch.testing.assert_close(
        loaded.score(inputs), model.score(inputs), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "options",
    [
        pytest.param(["--model_ckpt", "model.ckpt"], id="no-trust-or-digest"),
        pytest.param(
            ["--model_ckpt", "model.ckpt", "--trust_model_ckpt"], id="no-digest"
        ),
        pytest.param(
            ["--model_ckpt", "model.ckpt", "--model_ckpt_sha256", "a" * 64],
            id="no-trust",
        ),
        pytest.param(["--trust_model_ckpt"], id="trust-without-model"),
        pytest.param(["--model_ckpt_sha256", "a" * 64], id="digest-without-model"),
    ],
)
def test_cli_rejects_unauthenticated_custom_checkpoints(monkeypatch, options):
    script = Path(__file__).parents[1] / "score.py"
    monkeypatch.setattr(
        sys,
        "argv",
        [str(script), "--compound_csv", "unused.csv", "--output_csv", "unused.csv"]
        + options,
    )
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(script), run_name="__main__")
    assert error.value.code == 2


def test_cli_verifies_trusted_custom_checkpoint(tmp_path, monkeypatch):
    script = Path(__file__).parents[1] / "score.py"
    compounds = tmp_path / "compounds.csv"
    compounds.write_text("smiles\nCCO\n")
    output = tmp_path / "scores.csv"
    digest = "a" * 64
    calls = []
    model = LitRankNet()

    def load(checkpoint_path, *, sha256, input_size):
        calls.append((checkpoint_path, sha256, input_size))
        return model

    def make_scorer(*, model, featurizer):
        return SimpleNamespace(score=lambda molrpr: np.array([1.25]))

    monkeypatch.setattr(LitRankNet, "load_verified_checkpoint", load)
    monkeypatch.setattr(scorer_module, "MolSkillScorer", make_scorer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--compound_csv",
            str(compounds),
            "--output_csv",
            str(output),
            "--model_ckpt",
            "trusted.ckpt",
            "--trust_model_ckpt",
            "--model_ckpt_sha256",
            digest,
        ],
    )
    runpy.run_path(str(script), run_name="__main__")
    assert calls == [("trusted.ckpt", digest, 2220)]
    assert output.read_text() == "smiles,score\nCCO,1.25\n"

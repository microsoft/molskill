import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import requests
from tqdm import tqdm


def verify_sha256(data: bytes, expected_sha256: str) -> None:
    """Reject content that does not match a SHA-256 from a trusted source."""
    if hashlib.sha256(data).hexdigest() != expected_sha256.lower():
        raise ValueError("SHA-256 mismatch. Do not load this file.")


def download(src: str, dest: str, sha256: Optional[str] = None) -> None:
    """Simple GET request with progress bar
    Args:
        src (str): Source link to download from
        dest (str): Destination file
        sha256 (Optional[str]): Expected digest from a trusted source.
    """
    destination = Path(dest)
    with requests.get(src, stream=True, timeout=(10, 60)) as response:
        response.raise_for_status()
        if response.status_code != 200:
            raise requests.HTTPError(
                f"Expected HTTP 200 for {src}, got {response.status_code}.",
                response=response,
            )
        tsize = int(response.headers.get("content-length", 0))
        with TemporaryDirectory(dir=destination.parent) as temp_dir:
            temporary_path = Path(temp_dir) / "download"
            with temporary_path.open("wb") as handle, tqdm(
                total=tsize, unit="iB", unit_scale=True, position=0, leave=False
            ) as progress:
                progress.set_description(os.path.basename(dest))
                for chunk in response.iter_content(chunk_size=1024):
                    handle.write(chunk)
                    progress.update(len(chunk))
            if temporary_path.stat().st_size == 0:
                raise ValueError(f"Empty download from {src}.")
            if sha256 is not None:
                verify_sha256(temporary_path.read_bytes(), sha256)
            temporary_path.replace(destination)

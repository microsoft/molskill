import os

import requests
from tqdm import tqdm


def download(src: str, dest: str) -> None:
    """Simple GET request with progress bar
    Args:
        src (str): Source link to download from
        dest (str): Destination file
    """
    r = requests.get(src, stream=True)
    tsize = int(r.headers.get("content-length", 0))
    progress = tqdm(total=tsize, unit="iB", unit_scale=True, position=0, leave=False)

    with open(dest, "wb") as handle:
        progress.set_description(os.path.basename(dest))
        for chunk in r.iter_content(chunk_size=1024):
            handle.write(chunk)
            progress.update(len(chunk))

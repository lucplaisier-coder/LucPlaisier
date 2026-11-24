"""
raw_data.py: find the raw Lending Club file, check its fingerprint, load it.

Plain-language summary
----------------------
1. The raw file is NOT in this repository (it is 375 MB and we may not
   redistribute it). You put it in data/raw/ yourself, or let
   `fetch_raw_file()` download it from Kaggle.
2. Before anything else we compute its SHA-256 fingerprint and compare it
   with the one recorded in config.py. A different file = stop.
3. We read only the ~25 columns we need, all as TEXT first. Reading as text
   means nothing is silently converted; the next modules convert each
   column with strict rules and stop on anything unexpected.
4. The raw file is opened read-only and never written to.
"""

import hashlib
import shutil
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

from . import config


def sha256_of(path: Path) -> str:
    """Return the SHA-256 fingerprint of a file (read in 1 MB pieces)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_raw_file(source: str | None = None) -> Path:
    """
    Make sure data/raw/accepted_2007_to_2018Q4.csv.gz exists.

    `source` can be:
      * None      -> the file must already be in data/raw/
      * a path    -> copy it from there (e.g. a Google Drive folder in Colab)
      * "kaggle"  -> download with the Kaggle API (needs a kaggle.json key,
                     see README "Getting the data")
    """
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = config.RAW_PATH
    if target.exists():
        return target

    if source is None:
        raise FileNotFoundError(
            f"Raw file not found at {target}. Put it there, or call "
            "fetch_raw_file('kaggle') / fetch_raw_file('<path to file>')."
        )

    if source == "kaggle":
        # Kaggle dataset wordsforthewise/lending-club (control document F.2).
        subprocess.run(
            ["kaggle", "datasets", "download", "wordsforthewise/lending-club",
             "-f", config.RAW_FILENAME, "-p", str(config.RAW_DIR)],
            check=True,
        )
        # Kaggle sometimes wraps a single file in a .zip; unwrap if so.
        zipped = config.RAW_DIR / (config.RAW_FILENAME + ".zip")
        if zipped.exists():
            with zipfile.ZipFile(zipped) as z:
                z.extractall(config.RAW_DIR)
            zipped.unlink()
    else:
        shutil.copyfile(source, target)

    if not target.exists():
        raise FileNotFoundError(f"Download/copy finished but {target} is missing.")
    return target


def verify_raw_fingerprint(path: Path = config.RAW_PATH) -> str:
    """Stop if the raw file is not exactly the file used in the project."""
    actual = sha256_of(path)
    if actual != config.RAW_SHA256:
        raise ValueError(
            "The raw file's fingerprint does not match the project's file.\n"
            f"  expected {config.RAW_SHA256}\n  found    {actual}\n"
            "Do not continue: results would not be comparable."
        )
    return actual


def load_raw_columns(path: Path = config.RAW_PATH) -> pd.DataFrame:
    """
    Read only the columns listed in config.RAW_COLUMNS, as text.

    keep_default_na=False + na_values=[""] means: only a truly EMPTY cell
    becomes missing. Words like "NONE" (home ownership) or "n/a" stay as
    text, so the later rules can decide what they mean.
    """
    return pd.read_csv(
        path,
        usecols=config.RAW_COLUMNS,
        dtype=str,
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    )

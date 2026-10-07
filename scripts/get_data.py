"""Fetch the Myntra listings scrape (526,564 rows) into data/myn.csv.

    python scripts/get_data.py

The dataset is published on Kaggle ("Myntra Fashion Clothing"); this script downloads a public copy
of the same file re-hosted on GitHub, which needs no account, and checks the archive's SHA-256.
The notebook reproduces v1's cluster sizes exactly from it. The data is not redistributed here.
"""
from __future__ import annotations

import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://raw.githubusercontent.com/1303-harshu/fashion_ecommerce_analysis/main/Myntra%20Fasion%20Clothing.csv.zip"
SHA256 = "9b3fb2c62c559def1bc9f13912dd6c37be2c51a3c5c7553895c81e8233842464"
TARGET = Path(__file__).resolve().parents[1] / "data" / "myn.csv"


def main() -> None:
    if TARGET.exists():
        print("myn.csv already present")
        return
    blob = urllib.request.urlopen(URL, timeout=600).read()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != SHA256:
        sys.exit("checksum mismatch: %s" % digest)
    TARGET.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        TARGET.write_bytes(z.read("Myntra Fasion Clothing.csv"))
    print("wrote", TARGET)


if __name__ == "__main__":
    main()

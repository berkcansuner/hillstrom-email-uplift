"""Download the Hillstrom MineThatData email CSV into data/raw/ and verify its checksum.

Usage:
    python scripts/download_data.py

The raw file is never modified after download. If it already exists and the
checksum matches, nothing is downloaded.
"""

import hashlib
import sys
import urllib.request

from hillstrom.paths import RAW_CSV

URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
EXPECTED_SHA256 = "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"


def sha256_of(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if RAW_CSV.exists() and sha256_of(RAW_CSV) == EXPECTED_SHA256:
        print(f"Already present and verified: {RAW_CSV}")
        return 0

    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = RAW_CSV.with_suffix(".csv.part")
    print(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, tmp_path)

    actual = sha256_of(tmp_path)
    if actual != EXPECTED_SHA256:
        tmp_path.unlink()
        print(f"Checksum mismatch: expected {EXPECTED_SHA256}, got {actual}", file=sys.stderr)
        return 1

    tmp_path.replace(RAW_CSV)
    print(f"Saved and verified: {RAW_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

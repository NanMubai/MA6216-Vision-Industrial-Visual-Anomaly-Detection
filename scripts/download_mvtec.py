"""Download and extract MVTec AD categories.

The download links are obtained after filling in the form at
https://www.mvtec.com/company/research/datasets/mvtec-ad . Licence: CC BY-NC-SA 4.0.

Usage:
    python scripts/download_mvtec.py --urls URL [URL ...]
    python scripts/download_mvtec.py --url-file links.txt          # one URL per line
    python scripts/download_mvtec.py --urls <full mvtec_anomaly_detection.tar.xz URL> \
        --categories bottle capsule screw carpet tile leather       # extract a subset

Archives go to data/raw/, extracted categories to data/mvtec_ad/<category>/.
Extraction uses tarfile's "data" filter (no absolute paths, links outside the target, devices).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def download(url: str, dest_dir: Path) -> Path:
    name = url.split("?")[0].rstrip("/").split("/")[-1] or "download.tar.xz"
    dest = dest_dir / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest} exists")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[get ] {url} -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "mvtec-downloader/1.0"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r       {done / 2**20:7.1f} / {total / 2**20:.1f} MiB", end="", flush=True)
    print()
    tmp.rename(dest)
    return dest


def extract(archive: Path, out_dir: Path, categories: set[str] | None) -> None:
    with tarfile.open(archive) as tf:
        members = tf.getmembers()
        if categories:
            members = [m for m in members if m.name.lstrip("./").split("/")[0] in categories]
        tops = sorted({m.name.lstrip("./").split("/")[0] for m in members})
        print(f"[xtr ] {archive.name}: {', '.join(tops)}")
        tf.extractall(out_dir, members=members, filter="data")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urls", nargs="*", default=[])
    ap.add_argument("--url-file")
    ap.add_argument("--categories", nargs="*", help="only extract these top-level folders")
    ap.add_argument("--raw-dir", default=str(REPO / "data" / "raw"))
    ap.add_argument("--out", default=str(REPO / "data" / "mvtec_ad"))
    ap.add_argument("--keep-archives", action="store_true")
    a = ap.parse_args()

    urls = list(a.urls)
    if a.url_file:
        urls += [l.strip() for l in Path(a.url_file).read_text().splitlines() if l.strip() and not l.startswith("#")]
    if not urls:
        sys.exit("no URLs given (--urls / --url-file)")
    raw, out = Path(a.raw_dir), Path(a.out)
    raw.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    cats = set(a.categories) if a.categories else None
    for url in urls:
        arc = download(url, raw)
        extract(arc, out, cats)
        if not a.keep_archives:
            arc.unlink()
    for d in sorted(p for p in out.iterdir() if p.is_dir()):
        n_train = len(list((d / "train" / "good").glob("*.png")))
        n_test = len(list((d / "test").glob("*/*.png")))
        print(f"  {d.name:12s} train/good={n_train:4d} test={n_test:4d}")
    if not a.keep_archives and raw.exists() and not any(raw.iterdir()):
        shutil.rmtree(raw)


if __name__ == "__main__":
    main()

"""
Download a real MOT Challenge sequence (frames + ground truth) without the
multi-gigabyte archive.

``MOT16.zip`` (~1.9 GB) bundles seven training sequences. This script only
streams the bytes it needs: it reads the ZIP end-of-central-directory from the
tail of the file, locates the ``MOT16-02`` entries, and pulls the per-frame
JPEGs plus ``gt.txt`` via HTTP ``Range`` requests::

    uv run python scripts/fetch_mot16_02.py --frames 90     # -> data/MOT16-02

The resulting folder is consumed by ``scripts/mot16_demo.py`` (renders
``docs/assets/gif/mot16_tracking.gif``) and the ``examples/mot16_*.py``
real-video tracking demos.

Data source: MOT16 (Dendorfer et al., 2021), https://motchallenge.net
"""

import argparse
import os
import struct
import urllib.request
import zlib

MOT16_URL = "https://motchallenge.net/data/MOT16.zip"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUT = os.path.join(REPO_ROOT, "data", "MOT16-02")


def _fetch(url, headers):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()


def _remote_central_directory(url, total_size):
    """Parse the central directory by reading only the file tail."""
    tail = _fetch(url, {"Range": f"bytes={total_size - 1_000_000}-{total_size - 1}"})
    idx = tail.rfind(b"\x50\x4b\x05\x06")
    if idx < 0:
        raise RuntimeError("could not locate the ZIP end-of-central-directory")
    cd_size = struct.unpack("<I", tail[idx + 12 : idx + 16])[0]
    cd_off = struct.unpack("<I", tail[idx + 16 : idx + 20])[0]
    cd = _fetch(url, {"Range": f"bytes={cd_off}-{cd_off + cd_size - 1}"})

    entries = {}
    pos = 0
    while pos + 46 <= len(cd) and cd[pos : pos + 4] == b"\x50\x4b\x01\x02":
        nlen = struct.unpack("<H", cd[pos + 28 : pos + 30])[0]
        elen = struct.unpack("<H", cd[pos + 30 : pos + 32])[0]
        clen = struct.unpack("<H", cd[pos + 32 : pos + 34])[0]
        name = cd[pos + 46 : pos + 46 + nlen].decode()
        comp = struct.unpack("<I", cd[pos + 20 : pos + 24])[0]
        lho = struct.unpack("<I", cd[pos + 42 : pos + 46])[0]
        entries[name] = (comp, lho)
        pos += 46 + nlen + elen + clen
    return entries


def _extract(url, entries, entry):
    comp, lho = entries[entry]
    hdr = _fetch(url, {"Range": f"bytes={lho}-{lho + 30 + 2}"})
    method = struct.unpack("<H", hdr[8:10])[0]
    nlen = struct.unpack("<H", hdr[26:28])[0]
    elen = struct.unpack("<H", hdr[28:30])[0]
    data = _fetch(url, {"Range": f"bytes={lho}-{lho + 30 + nlen + elen + comp - 1}"})
    payload = data[30 + nlen + elen :]
    if method == 8:
        dobj = zlib.decompressobj(-zlib.MAX_WBITS)
        payload = dobj.decompress(payload) + dobj.flush()
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--frames",
        type=int,
        default=90,
        help="how many leading frames of MOT16-02 to fetch",
    )
    parser.add_argument("--url", default=MOT16_URL)
    args = parser.parse_args(argv)

    print(f"Reading central directory of {args.url} ...")
    with urllib.request.urlopen(
        urllib.request.Request(args.url, method="HEAD"), timeout=60
    ) as resp:
        total = int(resp.headers["Content-Length"])
    entries = _remote_central_directory(args.url, total)

    prefix = "train/MOT16-02/"
    gt_key = f"{prefix}gt/gt.txt"
    if gt_key not in entries:
        raise RuntimeError(f"{gt_key} not found in archive")

    os.makedirs(os.path.join(args.out, "gt"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "img1"), exist_ok=True)

    with open(os.path.join(args.out, "gt", "gt.txt"), "wb") as f:
        f.write(_extract(args.url, entries, gt_key))
    print("wrote gt/gt.txt")

    for i in range(1, args.frames + 1):
        name = f"{prefix}img1/{i:06d}.jpg"
        with open(os.path.join(args.out, "img1", f"{i:06d}.jpg"), "wb") as f:
            f.write(_extract(args.url, entries, name))
    print(f"wrote {args.frames} frames to {args.out}")

    print(
        "\nNext:  uv run python scripts/mot16_demo.py --gif   "
        "(renders docs/assets/gif/mot16_tracking.gif)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

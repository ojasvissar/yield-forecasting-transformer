#!/usr/bin/env python
"""
Download the subset of Tiny CropNet referenced by the data index files.

Only files listed in the index JSONs are fetched from HuggingFace
(https://huggingface.co/datasets/fudong03/Tiny-CropNet). The index uses short
folder names ("WRF-HRRR", "USDA") while HuggingFace uses long ones
("WRF-HRRR Computed Dataset", "USDA Crop Dataset"), so symlinks are created to
make both resolve under the same root directory.

Sentinel-2 files cover a whole state-quarter (~4 GB each). By default only the
indexed counties and the 1st-of-month images the loader uses are copied out of
them, reading the remote HDF5 files with HTTP range requests (~1 GB for the
default 10 counties instead of ~59 GB). Use --full_sentinel for whole files.

Usage:
    python download_data.py                                  # USDA + HRRR (~1.9 GB)
    python download_data.py --modalities sentinel            # Sentinel-2 subset (~1 GB)
    python download_data.py --modalities sentinel --full_sentinel
"""

import argparse
import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor

HF_BASE = "https://huggingface.co/datasets/fudong03/Tiny-CropNet/resolve/main/"

# index folder name -> HuggingFace folder name
FOLDER_ALIASES = {
    "WRF-HRRR": "WRF-HRRR Computed Dataset",
    "USDA": "USDA Crop Dataset",
}


def to_hf_path(path):
    top, _, rest = path.partition("/")
    return FOLDER_ALIASES.get(top, top) + "/" + rest


def hf_url(hf_path):
    return HF_BASE + urllib.parse.quote(hf_path)


def read_index(index_files):
    """Relative data paths per modality, plus the counties needed from each Sentinel-2 file."""
    paths = {"usda": set(), "hrrr": set(), "sentinel": set()}
    sentinel_fips = defaultdict(set)
    for index_file in index_files:
        with open(index_file) as f:
            for obj in json.load(f):
                data = obj["data"]
                paths["usda"].add(data["USDA"])
                paths["hrrr"].update(data["HRRR"]["short_term"])
                for year_paths in data["HRRR"]["long_term"]:
                    paths["hrrr"].update(year_paths)
                paths["sentinel"].update(data["sentinel"])
                for p in data["sentinel"]:
                    sentinel_fips[p].add(obj["FIPS"])
    return paths, sentinel_fips


def download(rel_path, root_dir):
    """Download one whole file, skipping files that already exist."""
    dest = os.path.join(root_dir, to_hf_path(rel_path))
    if os.path.exists(dest):
        return "exists"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with urllib.request.urlopen(hf_url(to_hf_path(rel_path))) as resp, open(tmp, "wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    os.replace(tmp, dest)
    return "downloaded"


class RemoteFile(io.RawIOBase):
    """Read-only, seekable view of a remote file using HTTP range requests, with an LRU block cache."""

    def __init__(self, url, block_size=1 << 20, cache_blocks=64, retries=4):
        self.url, self.block_size, self.cache_blocks, self.retries = url, block_size, cache_blocks, retries
        self.cache = OrderedDict()
        self.pos = 0
        self._resolve()

    def _resolve(self):
        # follow the HuggingFace redirect once and keep the CDN URL for range requests
        req = urllib.request.Request(self.url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req) as resp:
            self.cdn_url = resp.geturl()
            self.size = int(resp.headers["Content-Range"].rsplit("/", 1)[1])

    def _fetch(self, start, end):
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.cdn_url, headers={"Range": f"bytes={start}-{end - 1}"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    return resp.read()
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 ** attempt)
                self._resolve()  # signed CDN URLs expire

    def _block(self, i):
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        start = i * self.block_size
        block = self._fetch(start, min(start + self.block_size, self.size))
        self.cache[i] = block
        if len(self.cache) > self.cache_blocks:
            self.cache.popitem(last=False)
        return block

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence] + offset
        return self.pos

    def readinto(self, b):
        n = max(0, min(len(b), self.size - self.pos))
        out = memoryview(b).cast("B")
        if n >= 4 * self.block_size:  # large reads (image arrays) bypass the cache
            out[:n] = self._fetch(self.pos, self.pos + n)
        else:
            done = 0
            while done < n:
                i, offset = divmod(self.pos + done, self.block_size)
                block = self._block(i)
                k = min(n - done, len(block) - offset)
                out[done:done + k] = block[offset:offset + k]
                done += k
        self.pos += n
        return n


def extract_sentinel(rel_path, fips_codes, root_dir):
    """Copy the given counties' 1st-of-month images out of a remote state-quarter HDF5 file."""
    import h5py

    dest = os.path.join(root_dir, rel_path)
    if os.path.exists(dest):
        with h5py.File(dest, "r") as hf:
            if set(fips_codes) <= set(hf.keys()):
                return "exists"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with h5py.File(RemoteFile(hf_url(rel_path)), "r") as src, h5py.File(tmp, "w") as dst:
        for fips in sorted(fips_codes):
            for date, group in src[fips].items():
                if not date.endswith("-01"):  # the loader only uses the 1st of each month
                    continue
                out = dst.create_group(f"{fips}/{date}")
                for name, ds in group.items():
                    out[name] = ds[()]
    os.replace(tmp, dest)
    return f"extracted {len(fips_codes)} counties"


def link_aliases(root_dir):
    for alias, target in FOLDER_ALIASES.items():
        link = os.path.join(root_dir, alias)
        if os.path.isdir(os.path.join(root_dir, target)) and not os.path.lexists(link):
            os.symlink(target, link)


def main():
    parser = argparse.ArgumentParser("Download Tiny CropNet subset")
    parser.add_argument("--root_dir", default="./data/TinyCropNet")
    parser.add_argument("--index_files", nargs="+",
                        default=["./data/soybean_train.json", "./data/soybean_val.json"])
    parser.add_argument("--modalities", nargs="+", default=["usda", "hrrr"],
                        choices=["usda", "hrrr", "sentinel"])
    parser.add_argument("--full_sentinel", action="store_true",
                        help="Download whole Sentinel-2 state files (~4 GB each) instead of extracting counties")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    paths, sentinel_fips = read_index(args.index_files)
    jobs = []
    for modality in args.modalities:
        for p in sorted(paths[modality]):
            if modality == "sentinel" and not args.full_sentinel:
                jobs.append((p, lambda p=p: extract_sentinel(p, sentinel_fips[p], args.root_dir)))
            else:
                jobs.append((p, lambda p=p: download(p, args.root_dir)))
    print(f"{len(jobs)} files for {args.modalities} -> {args.root_dir}")

    workers = min(args.workers, 4) if "sentinel" in args.modalities else args.workers
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [(p, pool.submit(job)) for p, job in jobs]
        for i, (p, future) in enumerate(futures, 1):
            print(f"[{i}/{len(jobs)}] {future.result()}: {p}", flush=True)

    link_aliases(args.root_dir)
    print("Done.")


if __name__ == "__main__":
    main()

"""SensEURCity: turn the 5.7 GB Zenodo archive into one compact parquet file per unit.

Source: Zenodo record 17858205, senseurcity_data_v03.zip (CC-BY-4.0), 85 AirSensEUR units
in Antwerp, Oslo and Zagreb, one-minute records, April 2020 to April 2021. Each unit file
carries the raw low-cost sensor channels, quality flags and, while the unit stood next to
a reference station, the reference instruments (Ref.* columns).

The archive is 22 GB unpacked, so nothing is unpacked to disk. Entries are decompressed
as a stream and written straight to parquet (float32, zstd). The script also works on a
PARTIAL download: the entry table is read from the remote archive once and cached, and
only entries whose bytes are already on disk are processed, so it can be rerun while the
download is still going.
"""
import io
import json
import os
import struct
import sys
import zlib

import numpy as np
import pandas as pd

DATA = os.environ.get('SEC_DATA', os.path.join(os.environ.get('DQ_DATA', os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')), 'senseurcity'))
ZIP = os.path.join(DATA, 'senseurcity_data_v03.zip')
URL = 'https://zenodo.org/api/records/17858205/files/senseurcity_data_v03.zip/content'
INDEX = os.path.join(DATA, 'zip_index.json')
OUT = os.path.join(DATA, 'units')
TEXT = ('date', 'Location.ID')


def index():
    if os.path.exists(INDEX):
        return json.load(open(INDEX))
    from remotezip import RemoteZip
    with RemoteZip(URL) as z:
        idx = [dict(name=i.filename, offset=i.header_offset, csize=i.compress_size, size=i.file_size,
                    method=i.compress_type) for i in z.infolist()]
    json.dump(idx, open(INDEX, 'w'))
    return idx


class Inflate(io.RawIOBase):
    """File-like view of one deflated zip entry, read from the raw archive bytes."""

    def __init__(self, path, entry):
        self.f = open(path, 'rb')
        self.f.seek(entry['offset'])
        h = self.f.read(30)
        assert h[:4] == b'PK' + bytes([3, 4]), 'no local header at offset'
        nlen, xlen = struct.unpack('<HH', h[26:30])
        self.f.seek(nlen + xlen, 1)
        self.left = entry['csize']
        self.d = zlib.decompressobj(-15) if entry['method'] == 8 else None
        self.buf = b''

    def readable(self):
        return True

    def readinto(self, b):
        while not self.buf and self.left > 0:
            raw = self.f.read(min(1 << 20, self.left))
            self.left -= len(raw)
            self.buf = self.d.decompress(raw) if self.d else raw
        if not self.buf and self.d:
            self.buf, self.d = self.d.flush(), None
        n = min(len(b), len(self.buf))
        b[:n] = self.buf[:n]
        self.buf = self.buf[n:]
        return n


def convert(entry):
    name = os.path.basename(entry['name'])[:-4]
    dst = os.path.join(OUT, name + '.parquet')
    if os.path.exists(dst):
        return 'have'
    parts = []
    reader = pd.read_csv(io.BufferedReader(Inflate(ZIP, entry), 1 << 20), chunksize=200000, low_memory=False)
    for ch in reader:
        ch['date'] = pd.to_datetime(ch['date'], utc=True).dt.tz_localize(None)
        for c in ch.columns:
            if c in TEXT:
                continue
            if c.endswith('_flag') or ch[c].dtype == object:
                ch[c] = ch[c].astype('string')
            else:
                ch[c] = ch[c].astype(np.float32)
        parts.append(ch)
    d = pd.concat(parts, ignore_index=True)
    for c in d.columns:
        if d[c].dtype == 'string' or c == 'Location.ID':
            d[c] = d[c].astype('category')
    d.to_parquet(dst + '.tmp', compression='zstd', index=False)
    os.replace(dst + '.tmp', dst)
    return '%d rows, %d cols, %.1f MB' % (len(d), d.shape[1], os.path.getsize(dst) / 1e6)


def main():
    os.makedirs(OUT, exist_ok=True)
    have = os.path.getsize(ZIP)
    todo = [e for e in index() if e['name'].startswith('dataset/') and e['name'].endswith('.csv')]
    only = sys.argv[1:]
    done = 0
    for e in todo:
        if only and not any(o in e['name'] for o in only):
            continue
        if e['offset'] + 30 + 400 + e['csize'] > have:
            continue
        print('%-34s %s' % (e['name'], convert(e)), flush=True)
        done += 1
    print('%d of %d entries on disk and converted; archive %.0f of 5704 MB' % (done, len(todo), have / 1048576))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Render tag36h11 textures for the four workstation tags.

Uses cv2.aruco, and cross-checks against libapriltag when it is installed.
The black square is 8/10 of the 0.10 m plate, so apriltag_ros size is 0.080.
"""
import ctypes
import ctypes.util
import os
import sys

import cv2
import numpy as np
from PIL import Image

CELL_PX = 40            # 40 px per tag cell -> 400x400 texture
TOTAL_CELLS = 10
BORDER_CELLS = 8
IDS = [1, 2, 3, 4]      # tag id per workstation WS01..WS04
OUT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'models', 'robocup_tags', 'materials', 'textures'))


# ----------------------------------------------------------------- opencv ---
def cells_opencv(tag_id):
    """10x10 cell grid (0=black, 1=white) for tag36h11 `tag_id`."""
    try:
        d = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)
        draw = cv2.aruco.drawMarker
    except AttributeError:                       # OpenCV >= 4.7 renamed these
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        draw = cv2.aruco.generateImageMarker

    # borderBits=1 -> 6 data + 1 black ring = 8 cells, one pixel per cell
    marker = draw(d, tag_id, BORDER_CELLS, borderBits=1)
    grid = (np.asarray(marker) > 127).astype(np.uint8)

    out = np.ones((TOTAL_CELLS, TOTAL_CELLS), dtype=np.uint8)   # white quiet zone
    out[1:1 + BORDER_CELLS, 1:1 + BORDER_CELLS] = grid
    return out


# --------------------------------------------------------------- apriltag ---
class AprilTagFamily(ctypes.Structure):
    """Mirrors struct apriltag_family in apriltag.h (apriltag3)."""
    _fields_ = [
        ('ncodes',           ctypes.c_uint32),
        ('codes',            ctypes.POINTER(ctypes.c_uint64)),
        ('width_at_border',  ctypes.c_int),
        ('total_width',      ctypes.c_int),
        ('reversed_border',  ctypes.c_bool),
        ('nbits',            ctypes.c_uint32),
        ('bit_x',            ctypes.POINTER(ctypes.c_uint32)),
        ('bit_y',            ctypes.POINTER(ctypes.c_uint32)),
        ('h',                ctypes.c_uint32),
        ('name',             ctypes.c_char_p),
        ('impl',             ctypes.c_void_p),
    ]


def load_family():
    """Return the tag36h11 family, or None if libapriltag isn't installed."""
    for cand in ('libapriltag.so', 'libapriltag.so.3',
                 ctypes.util.find_library('apriltag')):
        if not cand:
            continue
        try:
            lib = ctypes.CDLL(cand)
        except OSError:
            continue
        lib.tag36h11_create.restype = ctypes.POINTER(AprilTagFamily)
        return lib.tag36h11_create().contents
    return None


def cells_apriltag(fam, tag_id):
    """Reimplementation of apriltag_to_image() from apriltag.c."""
    tw = fam.total_width
    im = np.zeros((tw, tw), dtype=np.uint8)              # starts black

    wbw = fam.width_at_border + (0 if fam.reversed_border else 2)
    wbs = (tw - wbw) // 2
    for i in range(wbw - 1):                             # 1 cell white ring
        im[wbs, wbs + i] = 1
        im[wbs + i, tw - wbs - 1] = 1
        im[tw - wbs - 1, wbs + i + 1] = 1
        im[wbs + 1 + i, wbs] = 1

    code = fam.codes[tag_id]
    bs = (tw - fam.width_at_border) // 2
    for i in range(fam.nbits):                           # data bits
        if code & (1 << (fam.nbits - i - 1)):
            im[fam.bit_y[i] + bs, fam.bit_x[i] + bs] = 1
    return im


# ------------------------------------------------------------------- main ---
def main():
    fam = load_family()
    if fam is None:
        print('libapriltag not installed - using OpenCV only '
              '(install ros-humble-apriltag to enable the cross-check)')
    else:
        print(f'cross-check available: family={fam.name.decode()} '
              f'ncodes={fam.ncodes} nbits={fam.nbits} '
              f'total_width={fam.total_width} width_at_border={fam.width_at_border}')

    os.makedirs(OUT, exist_ok=True)
    failures = []
    for tid in IDS:
        grid = cells_opencv(tid)

        if fam is not None:
            ref = cells_apriltag(fam, tid)
            if not np.array_equal(grid, ref):
                failures.append(tid)
                print(f'  id {tid}: MISMATCH between OpenCV and libapriltag')
                continue

        big = np.kron(grid * 255, np.ones((CELL_PX, CELL_PX), dtype=np.uint8))
        path = os.path.join(OUT, f'tag36_11_{tid:05d}.png')
        Image.fromarray(big.astype(np.uint8), mode='L').convert('RGB').save(path)

        # check the written file decodes to the same id
        found, ids, _ = cv2.aruco.detectMarkers(
            np.array(Image.open(path).convert('L')),
            cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11))
        ok = ids is not None and int(ids.flatten()[0]) == tid
        print(f'  id {tid}: {big.shape[1]}px  redetect={"OK" if ok else "FAILED"}'
              f'  -> {os.path.relpath(path)}')
        if not ok:
            failures.append(tid)

    if failures:
        sys.exit(f'FAILED for ids {failures}')
    print(f'\nall {len(IDS)} textures written to {OUT}')
    print(f'black square = {BORDER_CELLS}/{TOTAL_CELLS} of plate '
          f'= 0.10 * 0.8 = 0.080 m  (apriltag.yaml `size`)')


if __name__ == '__main__':
    main()

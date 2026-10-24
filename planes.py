"""
planes.py - Miller planes clipped to the unit cell, as Plotly traces.
"""
import itertools

import numpy as np
import plotly.graph_objects as go

_CORNERS = np.array(list(itertools.product((0, 1), repeat=3)), dtype=float)
_EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8)
          if np.abs(_CORNERS[i] - _CORNERS[j]).sum() == 1]


def _cut(f, n):
    #Points where the plane  h x + k y + l z = n  meets the unit cube (fractional).
    pts = []
    for i, j in _EDGES:
        fi, fj = f[i] - n, f[j] - n
        if abs(fi) < 1e-9:
            pts.append(_CORNERS[i])
        if abs(fj) < 1e-9:
            pts.append(_CORNERS[j])
        if fi * fj < 0:
            t = fi / (fi - fj)
            pts.append(_CORNERS[i] + t * (_CORNERS[j] - _CORNERS[i]))
    if not pts:
        return None
    return np.unique(np.round(pts, 9), axis=0)


def _order_polygon(P):
    #Sort coplanar points around their centre. Returns (points, area) or None
    if len(P) < 3:
        return None
    c = P.mean(axis=0)
    u_, s, vt = np.linalg.svd(P - c)
    if s[1] < 1e-9:                     # all on a line
        return None
    u, v = vt[0], vt[1]
    ang = np.arctan2((P - c) @ v, (P - c) @ u)
    P = P[np.argsort(ang)]
    xy = np.c_[(P - c) @ u, (P - c) @ v]
    area = 0.5 * abs(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - xy[:, 1] * np.roll(xy[:, 0], -1)))
    return (P, area) if area > 1e-9 else None


def plane_polygons(h, k, l, lattice_matrix, all_planes=True):
    """Cartesian polygons where the planes (hkl) cut the unit cell.

    all_planes=True: every plane  h x + k y + l z = n  that crosses the cell.
    all_planes=False: just the one with the largest cross-section.
    """
    if (h, k, l) == (0, 0, 0):
        raise ValueError("(0 0 0) is not a plane")
    f = _CORNERS @ np.array([h, k, l], dtype=float)
    found = []
    for n in range(int(np.ceil(f.min() - 1e-9)), int(np.floor(f.max() + 1e-9)) + 1):
        pts = _cut(f, n)
        if pts is None:
            continue
        cart = pts @ lattice_matrix     # do the sorting in real space, not fractional
        res = _order_polygon(cart)
        if res is not None:
            found.append((n, res[0], res[1]))
    if not all_planes and found:
        found = [max(found, key=lambda t: (round(t[2], 9), t[0]))]
    return [P for _, P, _ in found]


def plane_traces(polygons, color="#d62728", name="plane", opacity=0.45):
    #One translucent Mesh3d plus an outline for a set of polygons.
    x, y, z, I, J, K = [], [], [], [], [], []
    ox, oy, oz = [], [], []
    offset = 0
    for P in polygons:
        x += list(P[:, 0]); y += list(P[:, 1]); z += list(P[:, 2])
        for t in range(1, len(P) - 1):          # triangle fan
            I.append(offset); J.append(offset + t); K.append(offset + t + 1)
        offset += len(P)
        closed = np.vstack([P, P[:1]])
        ox += list(closed[:, 0]) + [None]
        oy += list(closed[:, 1]) + [None]
        oz += list(closed[:, 2]) + [None]
    mesh = go.Mesh3d(x=x, y=y, z=z, i=I, j=J, k=K, color=color, opacity=opacity,
                     name=name, showlegend=True, flatshading=True, hoverinfo="name")
    edge = go.Scatter3d(x=ox, y=oy, z=oz, mode="lines", line=dict(color=color, width=4),
                        hoverinfo="skip", showlegend=False)
    return [mesh, edge]
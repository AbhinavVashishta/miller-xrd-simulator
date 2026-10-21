"""
engine.py - crystal maths and powder XRD simulation.
"""
import itertools
from dataclasses import dataclass

import gemmi
import numpy as np

CU_KA1 = 1.5406  # angstrom


@dataclass
class Crystal:
    cell: tuple            # a, b, c (angstrom), alpha, beta, gamma (degrees)
    elements: list         # one symbol per atom in the unit cell
    frac: np.ndarray       # (N, 3) fractional coordinates
    occ: np.ndarray        # (N,) site occupancies
    biso: np.ndarray       # (N,) isotropic B in A^2 (0 if the CIF has none)
    spacegroup: str = "P1"
    name: str = ""


@dataclass
class Pattern:
    two_theta: np.ndarray  # peak positions (deg)
    d: np.ndarray          # d-spacings (angstrom)
    intensity: np.ndarray  # relative, strongest peak = 100
    hkl: list              # one representative (h, k, l) per peak
    mult: np.ndarray       # number of hkl merged into each peak
    x: np.ndarray          # broadened profile, 2theta axis
    y: np.ndarray          # broadened profile, intensity


# lattice

def lattice_matrix(a, b, c, alpha, beta, gamma):
    # Rows are the a, b, c vectors in Cartesian coords (a along x, b in xy-plane).
    al, be, ga = np.radians([alpha, beta, gamma])
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    cz2 = c**2 - cx**2 - cy**2
    if min(a, b, c) <= 0 or cz2 <= 0:
        raise ValueError("these lengths/angles don't make a valid unit cell")
    cz = np.sqrt(cz2)
    return np.array([
        [a, 0.0, 0.0],
        [b * np.cos(ga), b * np.sin(ga), 0.0],
        [cx, cy, cz],
    ])


def reciprocal_matrix(M):
    # Rows are a*, b*, c* (no 2*pi), so that 1/d = |h a* + k b* + l c*|.
    return np.linalg.inv(M).T


def d_spacing(hkl, recip):
    G = np.atleast_2d(hkl) @ recip
    return 1.0 / np.linalg.norm(G, axis=1)


# CIF

def _element_of(site):
    el = site.element.name
    if el != "X":
        return el
    # AMCSD labels like OH or Ow aren't element symbols; use the leading letters
    letters = "".join(ch for ch in site.label if ch.isalpha())
    for n in (2, 1):
        guess = letters[:n].capitalize()
        if len(guess) == n and gemmi.Element(guess).name != "X":
            return guess
    raise ValueError(f"can't work out the element for site '{site.label}'")


def load_cif(path):
    small = gemmi.read_small_structure(str(path))
    # gemmi applies the space group operations and drops duplicate sites for us
    sites = small.get_all_unit_cell_sites()
    if not sites:
        raise ValueError(f"no atom sites found in {path}")

    frac = np.array([[s.fract.x, s.fract.y, s.fract.z] for s in sites])
    frac = np.round(frac, 6) % 1.0
    c = small.cell
    return Crystal(
        cell=(c.a, c.b, c.c, c.alpha, c.beta, c.gamma),
        elements=[_element_of(s) for s in sites],
        frac=frac,
        occ=np.array([s.occ for s in sites]),
        biso=np.array([8 * np.pi**2 * s.u_iso for s in sites]),
        spacegroup=small.spacegroup_hm,
        name=str(path),
    )


def build_crystal(cell, atoms, spacegroup="P 1", name=""):
    """Make a Crystal from an asymmetric unit.

    atoms: list of (element, x, y, z) or (element, x, y, z, occ) tuples.
    spacegroup: any symbol gemmi understands ("C -1", "Fm-3m", "P 21/c", ...).
    """
    sg = gemmi.find_spacegroup_by_name(spacegroup)
    if sg is None:
        raise ValueError(f"unknown space group '{spacegroup}'")
    ops = list(sg.operations())

    elements, frac, occ = [], [], []
    for atom in atoms:
        el, xyz = atom[0], np.array(atom[1:4], dtype=float)
        o = float(atom[4]) if len(atom) > 4 else 1.0
        images = []
        for op in ops:
            p = np.round(np.array(op.apply_to_xyz(list(xyz))), 6) % 1.0
            # skip images that land on an atom we already have from this site
            if not any(np.allclose((p - q + 0.5) % 1.0, 0.5, atol=1e-4) for q in images):
                images.append(p)
        for p in images:
            elements.append(el)
            frac.append(p)
            occ.append(o)
    return Crystal(cell=tuple(cell), elements=elements, frac=np.array(frac),
                   occ=np.array(occ), biso=np.zeros(len(elements)),
                   spacegroup=sg.hm, name=name)


# drawing helpers

def display_sites(crystal, tol=1e-4):
    """Markers for the 3D view.

    Atoms sharing a position (Al/Si, K/Na mixed sites) become one marker, and
    atoms on a face, edge or corner are repeated on the far side of the cell.
    Returns Cartesian positions, dominant element and a hover label per marker.
    """
    M = lattice_matrix(*crystal.cell)
    groups = {}
    for f, el, o in zip(crystal.frac, crystal.elements, crystal.occ):
        key = tuple(np.round(f, 3) % 1.0)
        groups.setdefault(key, []).append((el, o, f))

    pos, main, label = [], [], []
    for members in groups.values():
        members = sorted(members, key=lambda m: -m[1])
        f = members[0][2]
        if len(members) == 1:
            text = members[0][0]
        else:
            text = " ".join(f"{el}{o:.2f}" for el, o, _ in members)
        options = [(0, 1) if x < tol else (0,) for x in f]
        for shift in itertools.product(*options):
            pos.append(f + np.array(shift))
            main.append(members[0][0])
            label.append(f"{text}  ({f[0]:.3f}, {f[1]:.3f}, {f[2]:.3f})")
    return np.array(pos) @ M, main, label


def cell_edges(crystal):
    #12 cell edges as (start, end) Cartesian points
    M = lattice_matrix(*crystal.cell)
    corners = np.array(list(itertools.product((0, 1), repeat=3)))
    edges = []
    for i, p in enumerate(corners):
        for j, q in enumerate(corners):
            if i < j and np.abs(p - q).sum() == 1:
                edges.append((p @ M, q @ M))
    return edges


# structure factor

def scattering_factor(element, s2):
    #f0 from the International Tables (Cromer-Mann style) coefficients. s2 = (sin(theta)/lambda)^2
    co = gemmi.Element(element).it92
    s2 = np.asarray(s2)
    f = np.full(s2.shape, co.c)
    for a, b in zip(co.a, co.b):
        f = f + a * np.exp(-b * s2)
    return f


def structure_factors(crystal, hkl):
    hkl = np.atleast_2d(hkl)
    recip = reciprocal_matrix(lattice_matrix(*crystal.cell))
    s2 = (hkl @ recip) ** 2
    s2 = s2.sum(axis=1) / 4.0          # (sin(theta)/lambda)^2 = 1 / (4 d^2)

    phase = np.exp(2j * np.pi * (hkl @ crystal.frac.T))   # (n_hkl, n_atoms)
    F = np.zeros(len(hkl), dtype=complex)
    for el in set(crystal.elements):
        m = np.array([e == el for e in crystal.elements])
        f = scattering_factor(el, s2)
        dw = np.exp(-np.outer(s2, crystal.biso[m]))       # Debye-Waller
        F += f * (crystal.occ[m] * dw * phase[:, m]).sum(axis=1)
    return F


def is_extinct(crystal, hkl, rel_tol=1e-8):
    """True where F(hkl) vanishes. Used to cross-check the analytic rules.

    Compared against |F(000)|^2 (every atom in phase), so the answer for a
    reflection doesn't depend on which other reflections are in the list.
    """
    I = np.abs(structure_factors(crystal, hkl)) ** 2
    scale = np.abs(structure_factors(crystal, [0, 0, 0])[0]) ** 2
    return I < rel_tol * scale


# XRD

def generate_hkl(crystal, dmin):
    M = lattice_matrix(*crystal.cell)
    recip = reciprocal_matrix(M)
    # h = G . a and |G| <= 1/dmin, so |h| <= |a| / dmin
    hmax = np.floor(np.linalg.norm(M, axis=1) / dmin).astype(int)
    rng = [np.arange(-n, n + 1) for n in hmax]
    grid = np.stack(np.meshgrid(*rng, indexing="ij"), axis=-1).reshape(-1, 3)
    grid = grid[np.any(grid != 0, axis=1)]
    return grid[d_spacing(grid, recip) >= dmin]


def _pick_label(group):
    # prefer indices without minus signs, e.g. (1 0 0) over (-1 0 0)
    return min(group, key=lambda t: (sum(x < 0 for x in t), -t[0], -t[1], -t[2]))


def simulate_xrd(crystal, tt_min=5.0, tt_max=80.0, wavelength=CU_KA1,
                 fwhm=0.1, eta=0.5, step=0.01):
    recip = reciprocal_matrix(lattice_matrix(*crystal.cell))
    dmin = wavelength / (2 * np.sin(np.radians(tt_max / 2)))

    hkl = generate_hkl(crystal, dmin)
    d = d_spacing(hkl, recip)
    theta = np.arcsin(wavelength / (2 * d))
    keep = np.degrees(2 * theta) >= tt_min
    hkl, d, theta = hkl[keep], d[keep], theta[keep]

    if len(hkl) == 0:
        raise ValueError("no reflections in this 2theta range")

    F = structure_factors(crystal, hkl)
    lp = (1 + np.cos(2 * theta) ** 2) / (np.sin(theta) ** 2 * np.cos(theta))
    I = np.abs(F) ** 2 * lp

    # remove systematic absences first, so peak labels and multiplicities
    # only count reflections that actually diffract
    alive = I > 1e-6 * I.max()
    hkl, d, theta, I = hkl[alive], d[alive], theta[alive], I[alive]

    # reflections with the same d (symmetry equivalents) add up into one peak
    order = np.argsort(d)
    hkl, d, theta, I = hkl[order], d[order], theta[order], I[order]
    starts = np.concatenate([[0], np.flatnonzero(np.diff(d) > 1e-6) + 1])
    ends = np.append(starts[1:], len(d))
    I_sum = np.add.reduceat(I, starts)
    labels = [_pick_label([tuple(int(v) for v in r) for r in hkl[a:b]])
              for a, b in zip(starts, ends)]

    rel = 100 * I_sum / I_sum.max()
    tt = np.degrees(2 * theta[starts])

    x = np.arange(tt_min, tt_max + step, step)
    y = np.zeros_like(x)
    for pos, h in zip(tt, rel):
        dx = x - pos
        lor = 1 / (1 + 4 * dx**2 / fwhm**2)
        gau = np.exp(-4 * np.log(2) * dx**2 / fwhm**2)
        y += h * (eta * lor + (1 - eta) * gau)
    y *= 100 / y.max()

    return Pattern(two_theta=tt, d=d[starts], intensity=rel, hkl=labels,
                   mult=ends - starts, x=x, y=y)
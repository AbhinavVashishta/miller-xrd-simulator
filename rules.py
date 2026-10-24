# -*- coding: utf-8 -*-
"""
rules.py - crystal system classification and lattice-centering extinction rules.
"""

import numpy as np

SYSTEM_CONSTRAINTS = {
    "cubic": "a = b = c, alpha = beta = gamma = 90",
    "tetragonal": "a = b != c, alpha = beta = gamma = 90",
    "orthorhombic": "a != b != c, alpha = beta = gamma = 90",
    "hexagonal": "a = b != c, alpha = beta = 90, gamma = 120",
    "trigonal (rhombohedral axes)": "a = b = c, alpha = beta = gamma != 90",
    "monoclinic": "a != b != c, alpha = gamma = 90, beta != 90",
    "triclinic": "a != b != c, alpha != beta != gamma, no angle constraints",
}


def _len_eq(x, y, tol):
    """Relative length equality."""
    return abs(x - y) <= tol * max(abs(x), abs(y))


def _ang_is(angle, target, tol):
    return abs(angle - target) <= tol


def classify_crystal_system(a, b, c, alpha, beta, gamma,
                            len_tol=0.005, ang_tol=0.05):
    """Classify a unit cell into one of the keys of SYSTEM_CONSTRAINTS."""
    ab = _len_eq(a, b, len_tol)
    bc = _len_eq(b, c, len_tol)
    ac = _len_eq(a, c, len_tol)
    angles = (alpha, beta, gamma)
    lengths_equal = ab and bc and ac

    right = [_ang_is(x, 90.0, ang_tol) for x in angles]

    # All angles 90
    if all(right):
        if lengths_equal:
            return "cubic"
        if ab or bc or ac:
            return "tetragonal"
        return "orthorhombic"

    # Rhombohedral axes: a = b = c, alpha = beta = gamma (not 90)
    if (lengths_equal
            and _ang_is(alpha, beta, ang_tol)
            and _ang_is(beta, gamma, ang_tol)
            and _ang_is(alpha, gamma, ang_tol)):
        return "trigonal (rhombohedral axes)"

    # Hexagonal: two equal axes with 120 between them, other two angles 90.
    # Axis pairs: (a, b) <-> gamma, (a, c) <-> beta, (b, c) <-> alpha.
    pairs = (
        (ab, gamma, (alpha, beta)),
        (ac, beta, (alpha, gamma)),
        (bc, alpha, (beta, gamma)),
    )
    for equal, between, others in pairs:
        if (equal and _ang_is(between, 120.0, ang_tol)
                and all(_ang_is(x, 90.0, ang_tol) for x in others)):
            return "hexagonal"

    # Exactly two angles 90
    if sum(right) == 2:
        return "monoclinic"

    return "triclinic"


def centering_from_spacegroup(symbol):
    """Return the lattice centering letter from a space group symbol."""
    if not symbol:
        return "P"
    s = str(symbol).strip()
    if s and s[0].upper() in "PABCIFR":
        return s[0].upper()
    return "P"


def is_allowed(h, k, l, centering="P"):
    """True where the centering does not systematically cancel (h, k, l).

    Works on plain ints and numpy arrays (returns a bool array of the same
    shape for arrays, including for "P").
    """
    h_arr = np.asarray(h)
    k_arr = np.asarray(k)
    l_arr = np.asarray(l)

    if centering == "P":
        result = np.ones(np.broadcast(h_arr, k_arr, l_arr).shape, dtype=bool)
    elif centering == "A":
        result = (k_arr + l_arr) % 2 == 0
    elif centering == "B":
        result = (h_arr + l_arr) % 2 == 0
    elif centering == "C":
        result = (h_arr + k_arr) % 2 == 0
    elif centering == "I":
        result = (h_arr + k_arr + l_arr) % 2 == 0
    elif centering == "F":
        result = ((h_arr + k_arr) % 2 == 0) & ((h_arr + l_arr) % 2 == 0)
    elif centering == "R":
        result = (-h_arr + k_arr + l_arr) % 3 == 0
    else:
        raise ValueError(f"Unknown centering letter: {centering!r}")

    # Plain-int inputs give a plain bool back.
    if np.ndim(result) == 0:
        return bool(result)
    return result


RULE_TEXT = {
    "P": ("no condition", "Primitive lattice; no extra lattice points."),
    "A": ("k + l = 2n", "Extra lattice point at (0, 1/2, 1/2)."),
    "B": ("h + l = 2n", "Extra lattice point at (1/2, 0, 1/2)."),
    "C": ("h + k = 2n", "Extra lattice point at (1/2, 1/2, 0)."),
    "I": ("h + k + l = 2n", "Extra lattice point at (1/2, 1/2, 1/2)."),
    "F": ("h, k, l all even or all odd",
          "Extra lattice points at (0, 1/2, 1/2), (1/2, 0, 1/2), (1/2, 1/2, 0)."),
    "R": ("-h + k + l = 3n",
          "Obverse setting on hexagonal axes: extra lattice points at "
          "(2/3, 1/3, 1/3) and (1/3, 2/3, 2/3)."),
}

"""
Testing the ruling.

if __name__ == "__main__":
    assert classify_crystal_system(5, 5, 5, 90, 90, 90) == "cubic"
    assert classify_crystal_system(5, 5, 7, 90, 90, 90) == "tetragonal"
    assert classify_crystal_system(5, 6, 7, 90, 90, 90) == "orthorhombic"
    assert classify_crystal_system(5, 5, 5, 60, 60, 60) == "trigonal (rhombohedral axes)"
    assert classify_crystal_system(4, 4, 7, 90, 90, 120) == "hexagonal"
    assert classify_crystal_system(4, 7, 4, 90, 120, 90) == "hexagonal"
    assert classify_crystal_system(5, 6, 7, 90, 100, 90) == "monoclinic"
    assert classify_crystal_system(8.58, 12.96, 7.22, 90.53, 115.9, 87.7) == "triclinic"
    assert centering_from_spacegroup("C -1") == "C"
    assert centering_from_spacegroup("F d -3 m:2") == "F"
    assert centering_from_spacegroup("") == "P"
    assert is_allowed(1, 1, 0, "C") is True
    assert is_allowed(1, 0, 0, "C") is False
    arr = np.array([0, 1, 2])
    assert is_allowed(arr, arr, arr, "P").shape == (3,)
    assert list(is_allowed(arr, 0 * arr, 0 * arr, "F")) == [True, False, True]

    
"""

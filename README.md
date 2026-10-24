# xtalview

Interactive Python interface for Miller planes and powder XRD simulation
(Solid State Physics, Assignment 1, Question 2).

Give it a crystal (CIF file, preset mineral, or lattice parameters + space group + atoms typed in)
and it will:

- classify the crystal system from a, b, c, alpha, beta, gamma and draw the unit cell with all atoms
- work out which reflections the lattice centering allows, and check that against a direct
  structure-factor calculation
- draw any (hkl) planes inside the 3D cell
- simulate the powder XRD pattern with the hkl indices marked on the peaks


## Structures

| Mineral | AMCSD code | Space group | Reference |
|---|---|---|---|
| Kaolinite | 0012237 | C 1 | Bish (1993), Clays Clay Miner. 41, 738 |
| Microcline | 0006649 | C -1 | Allan and Angel (1997), Eur. J. Mineral. 9, 263 |
| Labradorite | 0000757 | C -1 | Wenk et al. (1980), Am. Mineral. 65, 81 |

Data from the American Mineralogist Crystal Structure Database (rruff.geo.arizona.edu/AMS).

## Notes for typing a structure in

- Enter the asymmetric unit in fractional coordinates, plus the space group. The space group
  generates the rest of the cell.
- The atom table takes B in square angstroms. If a source lists U, use B = 8 pi^2 U.
- Some space groups need an origin choice, e.g. `F d -3 m:2` for silicon.
- Simulated with Cu K-alpha1 (1.5406 A) unless you pick another radiation.

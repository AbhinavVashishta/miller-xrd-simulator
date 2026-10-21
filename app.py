"""
app.py - Streamlit front end for xtalview.
"""
import re
import tempfile
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import gemmi
import numpy as np
import pandas as pd
import streamlit as st

import engine
import plots
import rules

CIF_DIR = Path(__file__).parent / "data" / "cif"

RADIATION = {
    "Cu Kα1 (1.5406 Å)": 1.5406,
    "Cu Kα average (1.5418 Å)": 1.5418,
    "Co Kα1 (1.7890 Å)": 1.7890,
    "Mo Kα1 (0.7093 Å)": 0.7093,
}

# metric symmetry rank, used to warn when the cell is too low-symmetry for the space group
RANK = {"triclinic": 0, "monoclinic": 1, "orthorhombic": 2, "tetragonal": 3,
        "trigonal": 3, "hexagonal": 3, "cubic": 4}

st.set_page_config(page_title="xtalview", layout="wide")


# helpers

# streamlit can't hash a Crystal by itself, so say what identifies one
HASH = {engine.Crystal: lambda c: (c.cell, tuple(c.elements), c.frac.tobytes(),
                                   c.occ.tobytes(), c.biso.tobytes())}


def show_chart(fig):
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:                      # older streamlit
        st.plotly_chart(fig, use_container_width=True)


@st.cache_data(show_spinner=False)
def read_cif(path, mtime):
    return engine.load_cif(path)


@st.cache_data(show_spinner=False)
def read_cif_bytes(data):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "upload.cif"
        p.write_bytes(data)
        return engine.load_cif(p)


@st.cache_data(show_spinner=False, hash_funcs=HASH)
def run_xrd(crystal, tt_min, tt_max, wavelength, fwhm):
    return engine.simulate_xrd(crystal, tt_min, tt_max, wavelength, fwhm)


@st.cache_data(show_spinner=False, hash_funcs=HASH)
def reflection_table(crystal, n, centering):
    r = np.arange(-n, n + 1)
    grid = np.stack(np.meshgrid(r, r, r, indexing="ij"), axis=-1).reshape(-1, 3)
    grid = grid[np.any(grid != 0, axis=1)]
    recip = engine.reciprocal_matrix(engine.lattice_matrix(*crystal.cell))

    by_lattice = rules.is_allowed(grid[:, 0], grid[:, 1], grid[:, 2], centering)
    absent = engine.is_extinct(crystal, grid)
    F2 = np.abs(engine.structure_factors(crystal, grid)) ** 2

    status = np.select(
        [by_lattice & ~absent, by_lattice & absent, ~by_lattice & absent],
        ["allowed", "absent: atomic arrangement", "absent: lattice centering"],
        default="MISMATCH",
    )
    df = pd.DataFrame({
        "hkl": [plots.fmt_hkl(tuple(int(v) for v in g)) for g in grid],
        "d (Å)": engine.d_spacing(grid, recip),
        "centering rule": np.where(by_lattice, "allowed", "forbidden"),
        "|F|² (% of max)": 100 * F2 / F2.max(),
        "status": status,
    })
    return df.sort_values("d (Å)", ascending=False, ignore_index=True)


def parse_hkl(text):
    #Pull integers out of the text and group them in threes.
    nums = [int(v) for v in re.findall(r"-?\d+", text)]
    if not nums:
        return [], None
    if len(nums) % 3:
        return [], "Enter each plane as three integers separated by spaces, e.g. 1 0 -1; 2 1 0"
    out = []
    for i in range(0, len(nums), 3):
        t = tuple(nums[i:i + 3])
        if t == (0, 0, 0):
            return [], "(0 0 0) is not a plane."
        if max(abs(v) for v in t) > 10:
            return [], "Keep the indices to |h|, |k|, |l| <= 10."
        if t not in out:
            out.append(t)
    if len(out) > 8:
        return [], "At most 8 planes at a time."
    return out, None


def intercepts(hkl):
    return ", ".join("∞" if v == 0 else str(Fraction(1, v)) for v in hkl)


def cell_inputs(default_cell, tag):
    labels = ["a (Å)", "b (Å)", "c (Å)", "α (°)", "β (°)", "γ (°)"]
    values = []
    for i, (lab, v) in enumerate(zip(labels, default_cell)):
        values.append(st.sidebar.number_input(
            lab, value=float(round(v, 4)), min_value=0.1 if i < 3 else 1.0,
            max_value=200.0 if i < 3 else 179.0, step=0.01, format="%.4f", key=f"{tag}_{lab}"))
    return tuple(values)


# sidebar

st.sidebar.title("xtalview")
source = st.sidebar.radio("Structure source", ["Preset mineral", "Upload CIF", "Manual entry"])

crystal, tag = None, None

if source == "Preset mineral":
    files = sorted(CIF_DIR.glob("*.cif"))
    if not files:
        st.error(f"No CIF files found in {CIF_DIR}")
        st.stop()
    choice = st.sidebar.selectbox("Mineral", files, format_func=lambda p: p.stem.capitalize())
    crystal = read_cif(str(choice), choice.stat().st_mtime)
    tag = choice.stem

elif source == "Upload CIF":
    up = st.sidebar.file_uploader("CIF file", type=["cif"])
    if up is None:
        st.info("Upload a CIF file in the sidebar, or switch to a preset mineral.")
        st.stop()
    try:
        crystal = replace(read_cif_bytes(up.getvalue()), name=up.name)
    except Exception as e:
        st.error(f"Couldn't read that CIF: {e}")
        st.stop()
    tag = f"{up.name}_{up.size}"

else:
    tag = "manual"

default_cell = crystal.cell if crystal is not None else (5.64, 5.64, 5.64, 90.0, 90.0, 90.0)
st.sidebar.subheader("Lattice parameters")
cell = cell_inputs(default_cell, tag)

if source == "Manual entry":
    sg_symbol = st.sidebar.text_input("Space group", "F m -3 m",
                                      help="Hermann-Mauguin symbol, e.g. C -1, P 21/c, Fm-3m. Add :1 or :2 for the origin choice, e.g. F d -3 m:2")
    with st.expander("Atoms in the asymmetric unit", expanded=True):
        st.caption("Fractional coordinates. The space group generates the rest of the cell.")
        atoms_df = st.data_editor(
            pd.DataFrame({"element": ["Na", "Cl"], "x": [0.0, 0.5], "y": [0.0, 0.5],
                          "z": [0.0, 0.5], "occ": [1.0, 1.0]}),
            num_rows="dynamic", key="manual_atoms")
    atoms = []
    for _, r in atoms_df.dropna(subset=["element", "x", "y", "z"]).iterrows():
        el = str(r["element"]).strip().capitalize()
        if gemmi.Element(el).name == "X":
            st.error(f"'{el}' isn't an element symbol.")
            st.stop()
        occ = 1.0 if pd.isna(r.get("occ")) else float(r["occ"])
        atoms.append((el, r["x"], r["y"], r["z"], occ))
    if not atoms:
        st.info("Add at least one atom.")
        st.stop()
    try:
        crystal = engine.build_crystal(cell, atoms, sg_symbol, name="manual")
    except ValueError as e:
        st.error(str(e))
        st.stop()
else:
    crystal = replace(crystal, cell=cell)

st.sidebar.subheader("Diffraction")
radiation = st.sidebar.selectbox("Radiation", list(RADIATION))
wavelength = RADIATION[radiation]
tt_min, tt_max = st.sidebar.slider("2θ range (°)", 2.0, 120.0, (5.0, 80.0), step=1.0)
fwhm = st.sidebar.slider("Peak width, FWHM (°)", 0.02, 0.60, 0.10, step=0.01)
n_labels = st.sidebar.slider("Peaks to label", 0, 30, 12)
min_sep = st.sidebar.slider("Minimum label spacing (°)", 0.0, 1.0, 0.3, step=0.05)

# header

try:
    M = engine.lattice_matrix(*crystal.cell)
except ValueError as e:
    st.error(str(e))
    st.stop()

system = rules.classify_crystal_system(*crystal.cell)
centering = rules.centering_from_spacegroup(crystal.spacegroup)

st.title(crystal.name and Path(crystal.name).stem.capitalize() or "Crystal")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Crystal system", system.split(" (")[0].capitalize())
c2.metric("Space group", crystal.spacegroup)
c3.metric("Atoms in cell", len(crystal.elements), help="Sites counted separately, so a mixed Al/Si site counts twice.")
c4.metric("Cell volume", f"{abs(np.linalg.det(M)):.2f} Å³")
st.caption(f"{system.capitalize()}: {rules.SYSTEM_CONSTRAINTS[system]}   |   lattice centering: {centering}")

sg = gemmi.find_spacegroup_by_name(crystal.spacegroup)
if sg is not None and RANK[system.split(" (")[0]] < RANK[sg.crystal_system_str()]:
    st.warning(f"These cell parameters only allow a {system} lattice, but space group "
               f"{crystal.spacegroup} is {sg.crystal_system_str()}. Check the cell or the space group.")

tab_cell, tab_rules, tab_xrd = st.tabs(["Unit cell and planes", "Selection rules", "XRD pattern"])

# tab 1

with tab_cell:
    left, right = st.columns([1, 3])
    with left:
        text = st.text_input("Miller planes (hkl)", "0 0 1; 1 1 -1",
                             help="Separate indices with spaces and planes with semicolons.")
        all_planes = st.checkbox("Every parallel plane that cuts the cell", True,
                                 help="Off: only the plane with the largest cross-section.")
        show_atoms = st.checkbox("Show atoms", True)
        hkls, err = parse_hkl(text)
        if err:
            st.error(err)

        if hkls:
            arr = np.array(hkls)
            recip = engine.reciprocal_matrix(M)
            absent = engine.is_extinct(crystal, arr)
            st.dataframe(pd.DataFrame({
                "hkl": [plots.fmt_hkl(t) for t in hkls],
                "d (Å)": engine.d_spacing(arr, recip).round(4),
                "intercepts a, b, c": [intercepts(t) for t in hkls],
                "diffracts": np.where(absent, "no", "yes"),
            }), hide_index=True)
            st.caption("A plane with no diffraction is still a lattice plane; its reflection is "
                       "cancelled by symmetry or centering.")
    with right:
        show_chart(plots.structure_figure(crystal, hkls, all_planes, show_atoms))

    with st.expander("Atom list"):
        st.dataframe(pd.DataFrame({
            "element": crystal.elements, "x": crystal.frac[:, 0].round(4),
            "y": crystal.frac[:, 1].round(4), "z": crystal.frac[:, 2].round(4),
            "occupancy": crystal.occ, "B (Å²)": crystal.biso.round(3)}), hide_index=True)

# tab 2

with tab_rules:
    condition, origin = rules.RULE_TEXT[centering]
    st.markdown(f"**Lattice centering {centering}**: reflections are present only if **{condition}**. {origin}")
    st.markdown(
        "A centering translation **t** repeats every atom, so the structure factor picks up a factor "
        "(1 + e^(2πi **h**·**t**)). It cancels whenever **h**·**t** is a half-integer. Beyond the lattice, "
        "glide planes, screw axes and the atom positions themselves can cancel more reflections. "
        "The table below compares the centering rule with a direct calculation of "
        "F(hkl) = Σ f<sub>j</sub> exp[2πi(hx<sub>j</sub> + ky<sub>j</sub> + lz<sub>j</sub>)] over every atom in the cell.",
        unsafe_allow_html=True)

    n = st.slider("Largest index |h|, |k|, |l|", 1, 6, 3)
    table = reflection_table(crystal, n, centering)
    counts = table["status"].value_counts()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Reflections tested", len(table))
    m2.metric("Allowed", int(counts.get("allowed", 0)))
    m3.metric("Absent (centering)", int(counts.get("absent: lattice centering", 0)))
    m4.metric("Absent (atoms/symmetry)", int(counts.get("absent: atomic arrangement", 0)))
    if counts.get("MISMATCH", 0):
        st.error(f"{counts['MISMATCH']} reflections break the centering rule but have non-zero F. "
                 "The space group or its setting probably doesn't match the atom positions.")
    else:
        st.success("The centering rule and the structure factor agree on every reflection.")

    show = st.selectbox("Show", ["all", "allowed", "absent: lattice centering",
                                 "absent: atomic arrangement", "MISMATCH"])
    view = table if show == "all" else table[table["status"] == show]
    st.dataframe(view.assign(**{"d (Å)": view["d (Å)"].round(4),
                                "|F|² (% of max)": view["|F|² (% of max)"].round(3)}), hide_index=True)

# tab 3

with tab_xrd:
    try:
        pattern = run_xrd(crystal, tt_min, tt_max, wavelength, fwhm)
    except ValueError as e:
        st.warning(str(e))
        st.stop()

    show_chart(plots.xrd_figure(pattern, wavelength, n_labels, min_sep))
    st.caption("Peak position from Bragg's law, λ = 2d sinθ, with d from the reciprocal lattice. "
               "Intensity ∝ m |F|² (1 + cos²2θ) / (sin²θ cosθ). Equivalent reflections are merged into one peak "
               "(m = number merged).")

    peaks = pd.DataFrame({
        "2θ (°)": pattern.two_theta.round(3), "d (Å)": pattern.d.round(4),
        "I (%)": pattern.intensity.round(2),
        "hkl": [plots.fmt_hkl(t) for t in pattern.hkl], "m": pattern.mult})
    top = peaks.sort_values("I (%)", ascending=False).head(25)
    st.subheader("Strongest reflections")
    st.dataframe(top, hide_index=True)
    st.download_button("Download all peaks (CSV)", peaks.to_csv(index=False),
                       file_name="peaks.csv", mime="text/csv")
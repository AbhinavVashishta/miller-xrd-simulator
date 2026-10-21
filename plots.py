"""
plots.py - Plotly figures for the app (kept apart from Streamlit so they can be tested).
"""
import numpy as np
import plotly.graph_objects as go

import engine
import planes

# colour, marker size
STYLE = {
    "H": ("#bdbdbd", 5), "O": ("#e8403a", 9), "Si": ("#4f6fd0", 9), "Al": ("#e0a458", 10),
    "K": ("#8f40d4", 15), "Na": ("#1fb5c9", 13), "Ca": ("#3dbf3d", 14), "Mg": ("#0f9d58", 11),
    "Fe": ("#a0522d", 11), "Ti": ("#7f8c8d", 11), "Cl": ("#1fd11f", 13), "C": ("#333333", 8),
}
DEFAULT_STYLE = ("#8c8c8c", 9)
PLANE_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#e377c2", "#7f7f7f"]


def fmt_hkl(t):
    return "({} {} {})".format(*t)


def structure_figure(crystal, hkl_list=(), all_planes=True, show_atoms=True):
    M = engine.lattice_matrix(*crystal.cell)
    fig = go.Figure()

    # unit cell edges
    xs, ys, zs = [], [], []
    for p, q in engine.cell_edges(crystal):
        xs += [p[0], q[0], None]
        ys += [p[1], q[1], None]
        zs += [p[2], q[2], None]
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", hoverinfo="skip",
                               line=dict(color="#333333", width=4), showlegend=False))

    # a, b, c labels just past the end of each axis
    tips = 1.08 * M
    fig.add_trace(go.Scatter3d(x=tips[:, 0], y=tips[:, 1], z=tips[:, 2], mode="text",
                               text=["a", "b", "c"], textfont=dict(size=16, color="#333333"),
                               hoverinfo="skip", showlegend=False))

    if show_atoms:
        pos, main, label = engine.display_sites(crystal)
        main = np.array(main)
        for el in sorted(set(main)):
            m = main == el
            color, size = STYLE.get(el, DEFAULT_STYLE)
            fig.add_trace(go.Scatter3d(
                x=pos[m, 0], y=pos[m, 1], z=pos[m, 2], mode="markers", name=el,
                marker=dict(size=size, color=color, line=dict(color="white", width=0.5)),
                text=[label[i] for i in np.flatnonzero(m)], hoverinfo="text"))

    for n, (h, k, l) in enumerate(hkl_list):
        polys = planes.plane_polygons(h, k, l, M, all_planes=all_planes)
        if polys:
            fig.add_traces(planes.plane_traces(polys, PLANE_COLORS[n % len(PLANE_COLORS)], fmt_hkl((h, k, l))))

    hidden = dict(visible=False)
    fig.update_layout(
        scene=dict(aspectmode="data", xaxis=hidden, yaxis=hidden, zaxis=hidden,
                   camera=dict(eye=dict(x=1.5, y=-1.7, z=0.9))),
        margin=dict(l=0, r=0, t=0, b=0), height=620, template="plotly_white",
        legend=dict(itemsizing="constant"),
    )
    return fig


def pick_labels(pattern, n_labels, min_sep):
    #Strongest peaks first, skipping any that sit closer than min_sep degrees to one already chosen.
    chosen = []
    for i in np.argsort(-pattern.intensity):
        if len(chosen) >= n_labels:
            break
        if all(abs(pattern.two_theta[i] - pattern.two_theta[j]) >= min_sep for j in chosen):
            chosen.append(int(i))
    return chosen


def xrd_figure(pattern, wavelength, n_labels=12, min_sep=0.3):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pattern.x, y=pattern.y, mode="lines", name="simulated",
                             line=dict(color="#1f4e79", width=1.4)))

    # a tick for every allowed reflection, with hkl and d on hover
    fig.add_trace(go.Scatter(
        x=pattern.two_theta, y=np.full(len(pattern.two_theta), -3.0), mode="markers", name="reflections",
        marker=dict(symbol="line-ns", size=9, line=dict(color="#c0392b", width=1)),
        text=[f"{fmt_hkl(t)}  d = {d:.4f} A" for t, d in zip(pattern.hkl, pattern.d)], hoverinfo="text+x"))

    for i in pick_labels(pattern, n_labels, min_sep):
        top = float(np.interp(pattern.two_theta[i], pattern.x, pattern.y))
        fig.add_annotation(x=pattern.two_theta[i], y=top, text=fmt_hkl(pattern.hkl[i]), textangle=-90,
                           showarrow=False, yanchor="bottom", yshift=3, font=dict(size=11))

    fig.update_layout(
        template="plotly_white", height=540, margin=dict(l=60, r=20, t=20, b=50),
        xaxis=dict(title=f"2θ (degrees, λ = {wavelength:g} Å)"),
        yaxis=dict(title="Relative intensity (%)", range=[-8, 140]),
        legend=dict(orientation="h", y=1.02, x=0),
    )
    return fig
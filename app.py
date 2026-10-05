"""
TSP Solver - Streamlit app (Advanced Operations Research assignment)

Run locally:   streamlit run app.py
Algorithm:     Nearest Neighbour  ->  Local search (2-opt, Or-opt, 3-opt, Swap)
               ->  Iterated Local Search.  All moves hand-written in tsp_core.py.
"""
import json
import os
import itertools
import time

import plotly.graph_objects as go
import streamlit as st

from tsp_core import (LocalSearch, best_nearest_neighbour, distance_matrix,
                      iterated_local_search, parse_tsp, quadrant_neighbour_lists,
                      tour_length, is_valid_tour)

BEST_KNOWN = {"a280": 2579, "d198": 15780, "d493": 35002, "fl417": 11861,
              "lin318": 42029, "pcb442": 50778, "pr152": 73682, "pr226": 80369,
              "pr439": 107217, "ts225": 126643}
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

_KEY = itertools.count()


def key():
    return f"chart{next(_KEY)}"


st.set_page_config(page_title="TSP Solver", page_icon="🧭", layout="wide")


# ------------------------------------------------------------------ helpers
@st.cache_data(show_spinner=False)
def prepare(text):
    info = parse_tsp(text)
    coords = info["coords"]
    D = distance_matrix(coords)
    neigh = quadrant_neighbour_lists(D, coords, 6, 3)
    return info, coords, D, neigh


def tour_figure(coords, tour, title, color="#2563eb"):
    xs = [coords[i][0] for i in tour] + [coords[tour[0]][0]]
    ys = [coords[i][1] for i in tour] + [coords[tour[0]][1]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                             line=dict(color=color, width=1.6), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[c[0] for c in coords], y=[c[1] for c in coords],
                             mode="markers", marker=dict(size=4, color="#dc2626"),
                             text=[f"city {i + 1}" for i in range(len(coords))],
                             hoverinfo="text"))
    fig.add_trace(go.Scatter(x=[coords[tour[0]][0]], y=[coords[tour[0]][1]],
                             mode="markers", marker=dict(size=11, color="#111827",
                                                         symbol="square"),
                             hoverinfo="skip"))
    fig.update_layout(title=title, showlegend=False, height=560,
                      margin=dict(l=10, r=10, t=50, b=10),
                      xaxis=dict(visible=False),
                      yaxis=dict(visible=False, scaleanchor="x", scaleratio=1))
    return fig


def gap(length, name):
    b = BEST_KNOWN.get(name)
    return None if b is None else 100.0 * (length - b) / b


def fmt_gap(length, name):
    g = gap(length, name)
    return "–" if g is None else f"{g:+.2f}% vs best known"


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🧭 TSP Solver")
st.sidebar.caption("Nearest Neighbour → Local Search → Iterated Local Search. "
                   "No optimisation libraries – every move is hand-coded.")

files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".tsp"))
source = st.sidebar.radio("Instance", ["Choose from list", "Upload .tsp file"])
if source == "Choose from list":
    fname = st.sidebar.selectbox("TSPLIB instance", files)
    with open(os.path.join(DATA_DIR, fname)) as f:
        text = f.read()
else:
    up = st.sidebar.file_uploader("Upload TSPLIB file (EUC_2D)", type=["tsp", "txt"])
    if up is None:
        st.info("Upload a .tsp file in the sidebar to begin.")
        st.stop()
    text = up.read().decode("utf-8", errors="ignore")

st.sidebar.subheader("Improvement moves")
use_2opt = st.sidebar.checkbox("2-opt (invert a section)", True)
use_oropt = st.sidebar.checkbox("Or-opt (relocate 1–3 cities)", True)
use_3opt = st.sidebar.checkbox("3-opt segment exchange", True)
use_swap = st.sidebar.checkbox("Swap two cities", True)

st.sidebar.subheader("Metaheuristic")
use_ils = st.sidebar.checkbox("Iterated Local Search (escape local minima)", True)
time_limit = st.sidebar.slider("ILS time limit (seconds)", 5, 120, 20, 5)
seed = st.sidebar.number_input("Random seed", 1, 9999, 1)
run = st.sidebar.button("▶ Solve", type="primary")

# ------------------------------------------------------------------ main
info, coords, D, neigh = prepare(text)
name = info["name"] or "uploaded"
n = len(coords)
st.title(f"Travelling Salesman Problem – {name}")
st.caption(f"{n} cities · {info.get('comment', '')} · distances = TSPLIB EUC_2D (rounded)")

tab_solve, tab_hof, tab_method = st.tabs(["Solve", "Hall of Fame results", "Method"])

with tab_solve:
    m1, m2, m3, m4 = st.columns(4)
    plot_ph = st.empty()
    status_ph = st.empty()
    chart_ph = st.empty()

    if not run:
        plot_ph.plotly_chart(key=key(), figure_or_data=tour_figure(coords, list(range(n)), "Cities (press ▶ Solve)",
                                         color="rgba(0,0,0,0)"))
        bk = BEST_KNOWN.get(name)
        if bk:
            m4.metric("Best known (TSPLIB)", f"{bk:,}")
    else:
        if not (use_2opt or use_oropt or use_3opt or use_swap):
            st.warning("Switch on at least one improvement move.")
            st.stop()
        # Step 1 – Nearest Neighbour from every start city
        t0 = time.time()
        status_ph.info("Step 1/3 – Nearest Neighbour from every start city…")
        nn_tour, nn_len, nn_start, _ = best_nearest_neighbour(D)
        m1.metric("1 · Nearest Neighbour", f"{nn_len:,}", fmt_gap(nn_len, name),
                  delta_color="off")
        plot_ph.plotly_chart(key=key(), figure_or_data=tour_figure(coords, nn_tour,
                                         f"Nearest Neighbour tour: {nn_len:,}", "#9ca3af"))

        # Step 2 – Local search (hill climbing)
        status_ph.info("Step 2/3 – Local search (hill climbing) …")
        ls = LocalSearch(D, nn_tour, neigh, use_2opt, use_oropt, use_swap, use_3opt)
        ls.run()
        ls_len = ls.length
        m2.metric("2 · Local search", f"{ls_len:,}", fmt_gap(ls_len, name),
                  delta_color="off")
        plot_ph.plotly_chart(key=key(), figure_or_data=tour_figure(coords, ls.tour,
                                         f"After local search: {ls_len:,}", "#f59e0b"))
        final_tour, final_len, history = list(ls.tour), ls_len, [(0.0, ls_len)]

        # Step 3 – Iterated Local Search
        if use_ils:
            last = {"t": 0.0}

            def cb(tour, length, kicks, elapsed):
                if elapsed - last["t"] < 1.5:
                    return
                last["t"] = elapsed
                status_ph.info(f"Step 3/3 – Iterated Local Search … {elapsed:.0f}/"
                               f"{time_limit}s · {kicks:,} kicks · best {length:,}")
                plot_ph.plotly_chart(key=key(), figure_or_data=tour_figure(coords, tour, f"ILS best so far: {length:,}"))

            res = iterated_local_search(D, ls.tour, neigh, time_limit=time_limit,
                                        seed=int(seed), use_2opt=use_2opt,
                                        use_oropt=use_oropt, use_swap=use_swap,
                                        use_3opt=use_3opt, spatial_kick_prob=0.3,
                                        callback=cb, callback_every=1.5)
            final_tour, final_len = res["tour"], res["length"]
            history = [(h[0], h[1]) for h in res["history"]]
            kicks = res["kicks"]
        else:
            kicks = 0

        assert is_valid_tour(final_tour, n)
        assert tour_length(final_tour, D) == final_len
        m3.metric("3 · Final tour", f"{final_len:,}", fmt_gap(final_len, name),
                  delta_color="off")
        bk = BEST_KNOWN.get(name)
        m4.metric("Best known (TSPLIB)", f"{bk:,}" if bk else "–")
        plot_ph.plotly_chart(key=key(), figure_or_data=tour_figure(coords, final_tour, f"Final tour: {final_len:,}",
                                         "#16a34a"))
        status_ph.success(f"Done in {time.time() - t0:.1f}s · improvement over NN: "
                          f"{100 * (nn_len - final_len) / nn_len:.1f}% · ILS kicks: {kicks:,}")
        if len(history) > 1:
            fig = go.Figure(go.Scatter(x=[h[0] for h in history], y=[h[1] for h in history],
                                       mode="lines+markers", line_shape="hv"))
            fig.update_layout(title="Convergence: best tour length over time",
                              xaxis_title="seconds", yaxis_title="tour length", height=300,
                              margin=dict(l=10, r=10, t=40, b=10))
            chart_ph.plotly_chart(key=key(), figure_or_data=fig)
        st.download_button("Download tour (city order, 1-based)",
                           "\n".join(str(c + 1) for c in final_tour),
                           file_name=f"{name}_{final_len}.tour")

with tab_hof:
    path = os.path.join(HERE, "results.json")
    if os.path.exists(path):
        res = json.load(open(path))
        rows = []
        for k in sorted(res):
            r = res[k]
            rows.append({"Instance": k, "Cities": r["n"], "Nearest Neighbour": r["nn"],
                         "After local search": r["ls"], "Final (km)": r["final"],
                         "Best known": r["best_known"],
                         "Gap %": round(100 * (r["final"] - r["best_known"]) / r["best_known"], 2)})
        st.subheader("Our results on the 10 challenge instances")
        st.dataframe(rows, hide_index=True)
        pick = st.selectbox("Show saved tour", sorted(res))
        st.plotly_chart(key=key(), figure_or_data=tour_figure(prepare(open(os.path.join(DATA_DIR, pick + ".tsp")).read())[1],
                                    res[pick]["tour"], f"{pick}: {res[pick]['final']:,}",
                                    "#16a34a"))
    else:
        st.info("results.json not found – run `python run_benchmark.py`.")

with tab_method:
    st.markdown("""
**1. Construction – Nearest Neighbour.** From a start city always visit the closest unvisited
city. Run from *every* start city and keep the shortest (restart idea).

**2. Improvement – Neighbourhood search (hill climbing).** Apply a move only if the tour gets
shorter, until no move helps (a *local minimum*). Moves:
- **2-opt** – remove two edges, reverse the section between them (the "invert" move).
- **Or-opt** – relocate a chain of 1–3 cities to a better place (the "relocate" move).
- **3-opt segment exchange** – cut three edges and reconnect the two middle pieces in a new
  order (a "multi-exchange" move).
- **Swap** – exchange two cities.

Switching between moves when one gets stuck = **change the neighbourhood (escape 1)**.

**3. Metaheuristic – Iterated Local Search.** Give the tour a random *kick* (double-bridge),
run local search again, keep it if not worse. Repeating this escapes local minima.

**Speed-ups (from class):** delta evaluation – only the changed edges are re-computed;
candidate (neighbour) lists – only nearby cities are tried, including nearest cities in each
of 4 directions; a work queue so only cities near the last change are re-checked.

No optimisation library is used – all code is in `tsp_core.py`.
""")

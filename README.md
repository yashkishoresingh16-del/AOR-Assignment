# TSP Solver – Advanced OR assignment

Nearest Neighbour -> Local search (2-opt, Or-opt, 3-opt, Swap) -> Iterated Local Search.
Pure Python, no optimisation library. All moves are in `tsp_core.py`.

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Re-run the benchmark (writes res_*.json; merge into results.json)
    python run_benchmark.py 20

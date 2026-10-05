"""
tsp_core.py  -  Travelling Salesman Problem solver written from scratch in pure Python.

No optimisation library is used. Every move is implemented by hand.

Pipeline (course terminology in brackets):
    1. Read TSPLIB file, build rounded distance matrix (EUC_2D, TSPLIB "nint" rounding)
    2. Nearest Neighbour tour                      [construction heuristic]
    3. Local search with 2-opt, Or-opt, Swap        [neighbourhood search / hill climbing,
                                                     change the neighbourhood = VND]
    4. Iterated Local Search                        [metaheuristic: escape local minima by
                                                     a random "kick" + local search again]

Speed ideas from class:
    * Delta evaluation  - only the edges a move touches are added/subtracted
                          ("evaluate only what changed").
    * Neighbour lists   - a city is only connected to one of its k nearest cities
                          ("restrict the choices").
    * Don't-look queue  - only cities near the last change are re-examined
                          ("only the impacted part of the network is re-evaluated").
"""

import math
import random
import time
from collections import deque


# --------------------------------------------------------------------------------------
# 1. Reading the data
# --------------------------------------------------------------------------------------
def parse_tsp(text):
    """Parse a TSPLIB .tsp file (EUC_2D). Returns dict with name, comment, coords."""
    info = {"name": "", "comment": "", "coords": []}
    in_coords = False
    coords = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("EOF"):
            break
        if upper.startswith("NODE_COORD_SECTION"):
            in_coords = True
            continue
        if in_coords:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    coords.append((float(parts[1]), float(parts[2])))
                except ValueError:
                    in_coords = False
            continue
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip().upper()
            val = val.strip()
            if key == "NAME":
                info["name"] = val
            elif key == "COMMENT":
                info["comment"] = val
            elif key == "EDGE_WEIGHT_TYPE":
                info["edge_weight_type"] = val
    info["coords"] = coords
    return info


def load_tsp(path):
    with open(path, "r") as f:
        return parse_tsp(f.read())


def nint(x):
    """TSPLIB rounding: nearest integer (0.5 rounds up)."""
    return int(x + 0.5)


def distance_matrix(coords):
    """Full matrix of rounded Euclidean distances (TSPLIB EUC_2D)."""
    n = len(coords)
    D = [[0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = coords[i]
        row = D[i]
        for j in range(i + 1, n):
            xj, yj = coords[j]
            d = nint(math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2))
            row[j] = d
            D[j][i] = d
    return D


def neighbour_lists(D, k=10):
    """For every city, its k nearest other cities (closest first)."""
    n = len(D)
    k = min(k, n - 1)
    out = []
    for i in range(n):
        row = D[i]
        order = sorted((j for j in range(n) if j != i), key=row.__getitem__)
        out.append(order[:k])
    return out


def quadrant_neighbour_lists(D, coords, k_near=6, k_quad=3):
    """Candidate list = the k_near nearest cities PLUS the k_quad nearest cities in each
    of the 4 compass quadrants around the city. In clustered maps the nearest cities are
    all in the same cluster; the quadrant cities give the search a way to fix the
    links between clusters. Sorted closest first."""
    n = len(D)
    out = []
    for i in range(n):
        row = D[i]
        xi, yi = coords[i]
        order = sorted((j for j in range(n) if j != i), key=row.__getitem__)
        chosen = list(order[:k_near])
        seen = set(chosen)
        quad_count = [0, 0, 0, 0]
        for j in order:
            xj, yj = coords[j]
            q = (0 if xj >= xi else 1) + (0 if yj >= yi else 2)
            if quad_count[q] < k_quad:
                quad_count[q] += 1
                if j not in seen:
                    seen.add(j)
                    chosen.append(j)
            if min(quad_count) >= k_quad:
                break
        chosen.sort(key=row.__getitem__)
        out.append(chosen)
    return out


def tour_length(tour, D):
    """Total length of a closed tour, computed from scratch."""
    total = 0
    prev = tour[-1]
    for c in tour:
        total += D[prev][c]
        prev = c
    return total


def is_valid_tour(tour, n):
    return len(tour) == n and set(tour) == set(range(n))


# --------------------------------------------------------------------------------------
# 2. Construction: Nearest Neighbour
# --------------------------------------------------------------------------------------
def nearest_neighbour(D, start=0):
    """Start at `start`, always go to the closest unvisited city."""
    n = len(D)
    remaining = [c for c in range(n) if c != start]
    tour = [start]
    cur = start
    while remaining:
        row = D[cur]
        nxt = min(remaining, key=row.__getitem__)
        remaining.remove(nxt)
        tour.append(nxt)
        cur = nxt
    return tour


def best_nearest_neighbour(D, starts=None, time_limit=None):
    """Run Nearest Neighbour from many start cities, keep the shortest
    ("restart from different starting points")."""
    n = len(D)
    if starts is None:
        starts = range(n)
    t0 = time.time()
    best_tour, best_len, best_start = None, float("inf"), None
    tried = 0
    for s in starts:
        t = nearest_neighbour(D, s)
        L = tour_length(t, D)
        tried += 1
        if L < best_len:
            best_tour, best_len, best_start = t, L, s
        if time_limit is not None and time.time() - t0 > time_limit:
            break
    return best_tour, best_len, best_start, tried


# --------------------------------------------------------------------------------------
# 3. Improvement: local search with 2-opt, Or-opt and Swap
# --------------------------------------------------------------------------------------
class LocalSearch:
    """Holds a tour and improves it with hand-written moves.

    tour[i] = city visited at position i ; pos[c] = position of city c.
    """

    def __init__(self, D, tour, neigh, use_2opt=True, use_oropt=True, use_swap=True,
                 use_3opt=True):
        self.D = D
        self.n = len(tour)
        self.tour = list(tour)
        self.pos = [0] * self.n
        for i, c in enumerate(self.tour):
            self.pos[c] = i
        self.neigh = neigh
        self.length = tour_length(self.tour, D)
        self.use_2opt = use_2opt
        self.use_oropt = use_oropt
        self.use_swap = use_swap
        self.use_3opt = use_3opt
        self.move_count = {"2-opt": 0, "Or-opt": 0, "3-opt": 0, "Swap": 0}

    # ---- helpers ---------------------------------------------------------------------
    def succ(self, c):
        p = self.pos[c] + 1
        return self.tour[0 if p == self.n else p]

    def pred(self, c):
        return self.tour[self.pos[c] - 1]

    def set_tour(self, tour, length=None):
        self.tour[:] = tour
        pos = self.pos
        for i, c in enumerate(self.tour):
            pos[c] = i
        self.length = tour_length(self.tour, self.D) if length is None else length

    def _reverse(self, i, j):
        """Reverse the tour between positions i..j (inclusive, going forward, wrapping).
        Reverses the shorter side - for a symmetric TSP both give the same cycle."""
        n, tour, pos = self.n, self.tour, self.pos
        L = (j - i) % n + 1
        if 2 * L > n:
            i, j = (j + 1) % n, (i - 1) % n
            L = n - L
        for _ in range(L // 2):
            a = tour[i]
            b = tour[j]
            tour[i] = b
            pos[b] = i
            tour[j] = a
            pos[a] = j
            i += 1
            if i == n:
                i = 0
            j -= 1
            if j < 0:
                j = n - 1

    # ---- 2-opt (course: "invert") ----------------------------------------------------
    def try_2opt(self, a):
        """Remove two edges and reconnect by reversing the section between them.
        Delta = d(a,c) + d(b,d) - d(a,b) - d(c,d)   (only 4 edges evaluated)."""
        D = self.D
        Da = D[a]
        # direction 1: edges a->b and c->d  (b = successor of a)
        b = self.succ(a)
        dab = Da[b]
        for c in self.neigh[a]:
            dac = Da[c]
            if dac >= dab:
                break
            d = self.succ(c)
            if c == b or d == a:
                continue
            delta = dac + D[b][d] - dab - D[c][d]
            if delta < 0:
                self._reverse(self.pos[b], self.pos[c])
                self.length += delta
                self.move_count["2-opt"] += 1
                return (a, b, c, d)
        # direction 2: edges b->a and d->c  (b = predecessor of a)
        b = self.pred(a)
        dab = Da[b]
        for c in self.neigh[a]:
            dac = Da[c]
            if dac >= dab:
                break
            d = self.pred(c)
            if c == b or d == a:
                continue
            delta = dac + D[b][d] - dab - D[c][d]
            if delta < 0:
                self._reverse(self.pos[a], self.pos[d])
                self.length += delta
                self.move_count["2-opt"] += 1
                return (a, b, c, d)
        return None

    # ---- Or-opt (course: "relocate" a city or a chain of 2-3 cities) -----------------
    def try_oropt(self, a, max_seg=3):
        """Take a segment of 1..3 consecutive cities containing `a` (at one end) out of
        the tour and insert it between two other neighbouring cities, possibly reversed."""
        D = self.D
        n = self.n
        tour, pos = self.tour, self.pos
        for L in range(1, max_seg + 1):
            if L >= n - 2:
                break
            for start_pos in (pos[a], (pos[a] - L + 1) % n):  # a first, or a last
                s1 = tour[start_pos]
                s2 = tour[(start_pos + L - 1) % n]
                p = tour[start_pos - 1]
                nx = tour[(start_pos + L) % n]
                remove_gain = D[p][s1] + D[s2][nx] - D[p][nx]
                if remove_gain <= 0:
                    continue
                # cities inside the segment
                seg = [tour[(start_pos + k) % n] for k in range(L)]
                inseg = set(seg)
                best = None  # (delta, x, y, reversed)
                for end, other in ((s1, s2), (s2, s1)):
                    Dend = D[end]
                    for c in self.neigh[end]:
                        g = remove_gain - Dend[c]
                        if g <= 0:
                            break
                        if c in inseg:
                            continue
                        # c adjacent to `end` in the new tour; two possible gaps around c
                        for x, y in ((c, self.succ(c)), (self.pred(c), c)):
                            if x in inseg or y in inseg:
                                continue
                            dxy = D[x][y]
                            if x == c:   # x=c -> end ... other -> y
                                delta = Dend[c] + D[other][y] - dxy - remove_gain
                                rev = (end == s2)
                            else:        # x -> other ... end -> c=y
                                delta = D[x][other] + Dend[c] - dxy - remove_gain
                                rev = (end == s1)
                            if delta < 0 and (best is None or delta < best[0]):
                                best = (delta, x, y, rev)
                if best is not None:
                    delta, x, y, rev = best
                    self._apply_oropt(seg, inseg, nx, x, rev)
                    self.length += delta
                    self.move_count["Or-opt"] += 1
                    return (p, nx, x, y, s1, s2)
        return None

    def _apply_oropt(self, seg, inseg, nx, x, rev):
        tour = self.tour
        start = self.pos[nx]
        rest = [c for c in tour[start:] + tour[:start] if c not in inseg]
        k = rest.index(x)
        piece = seg[::-1] if rev else seg
        new = rest[:k + 1] + piece + rest[k + 1:]
        self.tour[:] = new
        pos = self.pos
        for i, c in enumerate(new):
            pos[c] = i

    # ---- 3-opt segment exchange (course: "multi-exchange") ---------------------------
    def try_3opt(self, a):
        """Remove three edges (a,b), (c,d), (e,f) and reconnect the two middle pieces
        S1 = b..c and S2 = d..e in a new order / orientation:
            case 1:  a -> S2 -> S1 -> f              (pure segment exchange)
            case 2:  a -> S2 reversed -> S1 -> f
            case 3:  a -> S2 -> S1 reversed -> f
            case 4:  a -> S1 reversed -> S2 reversed -> f
        Searched with neighbour lists from both directions around a; gain is built up
        edge by edge and the search stops as soon as it cannot be positive."""
        for forward in (True, False):
            res = self._try_3opt_dir(a, forward)
            if res is not None:
                return res
        return None

    def _try_3opt_dir(self, a, forward):
        D, n, tour, pos, neigh = self.D, self.n, self.tour, self.pos, self.neigh
        if n < 8:
            return None
        pa = pos[a]
        if forward:
            def nxt(x):
                p = pos[x] + 1
                return tour[0 if p == n else p]

            def prv(x):
                return tour[pos[x] - 1]

            def rel(x):
                return (pos[x] - pa) % n
        else:
            def nxt(x):
                return tour[pos[x] - 1]

            def prv(x):
                p = pos[x] + 1
                return tour[0 if p == n else p]

            def rel(x):
                return (pa - pos[x]) % n
        Da = D[a]
        b = nxt(a)
        dab = Da[b]
        Db = D[b]
        for x in neigh[a]:
            g1 = dab - Da[x]
            if g1 <= 0:
                break
            rx = rel(x)
            if rx < 2:
                continue
            # --- x plays d (cases 1 and 3): c = prv(d)
            d = x
            c = prv(d)
            rd = rx
            G = g1 + D[c][d]
            # case 1: e -> b
            for e in neigh[b]:
                g2 = G - Db[e]
                if g2 <= 0:
                    break
                re_ = rel(e)
                if re_ < rd:
                    continue
                f = nxt(e)
                gain = g2 + D[e][f] - D[c][f]
                if gain > 0:
                    return self._apply_3opt(a, forward, rel(c), re_, 1, gain, (a, b, c, d, e, f))
            # case 3: e -> c
            Dc = D[c]
            if c != b:
                for e in neigh[c]:
                    g2 = G - Dc[e]
                    if g2 <= 0:
                        break
                    re_ = rel(e)
                    if re_ < rd:
                        continue
                    f = nxt(e)
                    gain = g2 + D[e][f] - Db[f]
                    if gain > 0:
                        return self._apply_3opt(a, forward, rel(c), re_, 3, gain, (a, b, c, d, e, f))
            # --- x plays e (case 2): f = nxt(e); new edges a-e, d-b, c-f
            e = x
            re_ = rx
            if re_ >= 2:
                f = nxt(e)
                G = g1 + D[e][f]
                for d in neigh[b]:
                    g2 = G - Db[d]
                    if g2 <= 0:
                        break
                    rd = rel(d)
                    if rd < 2 or rd > re_:
                        continue
                    c = prv(d)
                    gain = g2 + D[c][d] - D[c][f]
                    if gain > 0:
                        return self._apply_3opt(a, forward, rd - 1, re_, 2, gain, (a, b, c, d, e, f))
            # --- x plays c (case 4): d = nxt(c); new edges a-c, b-e, d-f
            c = x
            rc = rx
            d = nxt(c)
            G = g1 + D[c][d]
            for e in neigh[b]:
                g2 = G - Db[e]
                if g2 <= 0:
                    break
                re_ = rel(e)
                if re_ < rc + 1:
                    continue
                f = nxt(e)
                gain = g2 + D[e][f] - D[d][f]
                if gain > 0:
                    return self._apply_3opt(a, forward, rc, re_, 4, gain, (a, b, c, d, e, f))
        return None

    def _apply_3opt(self, a, forward, rc, re_, case, gain, touched):
        """Rebuild the tour: r = tour read from a (in the search direction),
        S1 = r[1..rc], S2 = r[rc+1..re], rest = r[re+1..]."""
        n, tour, pos = self.n, self.tour, self.pos
        pa = pos[a]
        if forward:
            r = tour[pa:] + tour[:pa]
        else:
            r = [tour[(pa - k) % n] for k in range(n)]
        S1 = r[1:rc + 1]
        S2 = r[rc + 1:re_ + 1]
        rest = r[re_ + 1:]
        if case == 1:
            new = [a] + S2 + S1 + rest
        elif case == 2:
            new = [a] + S2[::-1] + S1 + rest
        elif case == 3:
            new = [a] + S2 + S1[::-1] + rest
        else:
            new = [a] + S1[::-1] + S2[::-1] + rest
        tour[:] = new
        for i, c in enumerate(new):
            pos[c] = i
        self.length -= gain
        self.move_count["3-opt"] += 1
        return touched

    # ---- Swap (course: "swap two stops") ---------------------------------------------
    def try_swap(self, a):
        """Exchange the positions of city a and a nearby city c."""
        D = self.D
        n = self.n
        if n < 5:
            return None
        tour, pos = self.tour, self.pos
        pa, na = self.pred(a), self.succ(a)
        for c in self.neigh[a]:
            pc, nc = self.pred(c), self.succ(c)
            if na == c:      # ... pa a c nc ...  ->  pa c a nc
                delta = D[pa][c] + D[a][nc] - D[pa][a] - D[c][nc]
            elif nc == a:    # ... pc c a na ...  ->  pc a c na
                delta = D[pc][a] + D[c][na] - D[pc][c] - D[a][na]
            else:
                delta = (D[pa][c] + D[c][na] + D[pc][a] + D[a][nc]
                         - D[pa][a] - D[a][na] - D[pc][c] - D[c][nc])
            if delta < 0:
                i, j = pos[a], pos[c]
                tour[i], tour[j] = c, a
                pos[a], pos[c] = j, i
                self.length += delta
                self.move_count["Swap"] += 1
                return (a, c, pa, na, pc, nc)
        return None

    # ---- the search loop -------------------------------------------------------------
    def run(self, active=None, deadline=None, callback=None):
        """Hill climbing: keep applying improving moves until no neighbour is better
        (a local minimum for all switched-on neighbourhoods).

        `active` = cities to examine first (all cities if None). After a move, only the
        cities touched by it are put back in the queue (don't-look bits)."""
        n = self.n
        if active is None:
            active = list(self.tour)
        queue = deque()
        inq = [False] * n
        for c in active:
            if not inq[c]:
                inq[c] = True
                queue.append(c)
        steps = 0
        while queue:
            a = queue.popleft()
            inq[a] = False
            touched = None
            if self.use_2opt:
                touched = self.try_2opt(a)
            if touched is None and self.use_oropt:
                touched = self.try_oropt(a)
            if touched is None and self.use_3opt:
                touched = self.try_3opt(a)
            if touched is None and self.use_swap:
                touched = self.try_swap(a)
            if touched is not None:
                for c in touched:
                    if not inq[c]:
                        inq[c] = True
                        queue.append(c)
                if not inq[a]:
                    inq[a] = True
                    queue.append(a)
                steps += 1
                if callback is not None and steps % 50 == 0:
                    callback(self)
                if deadline is not None and steps % 50 == 0 and time.time() > deadline:
                    break
        return self.length


# --------------------------------------------------------------------------------------
# 4. Metaheuristic: Iterated Local Search
# --------------------------------------------------------------------------------------
def double_bridge_kick(tour, rng, window):
    """Random 'kick': cut three edges close together and reconnect segments B and C
    in swapped order (A B C D -> A C B D). Local search cannot undo this in one step,
    so it moves the search to a new valley."""
    n = len(tour)
    if n < 8:
        return list(tour), []
    w = max(8, min(window, n))
    i = rng.randrange(n)
    t = tour[i:] + tour[:i]
    p1, p2, p3 = sorted(rng.sample(range(1, w), 3))
    if p2 - p1 < 1 or p3 - p2 < 1:
        return list(tour), []
    new = t[:p1] + t[p2:p3] + t[p1:p2] + t[p3:]
    touched = [t[p1 - 1], t[p1], t[p2 - 1], t[p2], t[p3 - 1], t[p3 % n]]
    return new, touched


def ls_pos(tour, n):
    pos = [0] * n
    for i, c in enumerate(tour):
        pos[c] = i
    return pos


def spatial_double_bridge_kick(tour, pos, near, rng):
    """Same double-bridge kick, but the 4 cut points are cities that are close together
    ON THE MAP (a random city + 3 of its nearby cities), even if they are far apart in the
    visiting order. This lets the search rearrange how different parts of the tour link
    up - something a kick of consecutive positions cannot do."""
    n = len(tour)
    if n < 8:
        return list(tour), []
    a = rng.randrange(n)
    others = rng.sample(near[a], 3)
    ps = sorted({pos[a]} | {pos[c] for c in others})
    if len(ps) < 4:
        return list(tour), []
    r = ps[0] + 1                         # rotate: first cut becomes the end of the list
    t = tour[r:] + tour[:r]
    p1, p2, p3 = (p - r for p in ps[1:])  # cut after positions p1, p2, p3 (rotated)
    p1 += 1
    p2 += 1
    p3 += 1
    if not (0 < p1 < p2 < p3 < n):
        return list(tour), []
    new = t[:p1] + t[p2:p3] + t[p1:p2] + t[p3:]
    touched = [t[p1 - 1], t[p1], t[p2 - 1], t[p2], t[p3 - 1], t[p3 % n], t[-1], t[0]]
    return new, touched


def iterated_local_search(D, start_tour, neigh, time_limit=30.0, seed=1,
                          use_2opt=True, use_oropt=True, use_swap=True, use_3opt=True,
                          kick_window=50, global_kick_prob=0.0, temperature=0.0,
                          spatial_kick_prob=0.5, restart_after=0, restart_mode="random",
                          callback=None, callback_every=0.5):
    """Iterated Local Search.

    repeat until time is up:
        kick the current tour a little   (perturbation)
        run local search again           (hill climbing)
        accept the new tour if it is not worse; if it IS worse, still accept it
        with probability exp(-worsening / T), where T shrinks to 0 over time
        (simulated-annealing acceptance = "accept a worse move now and then")
    The best tour ever seen is always remembered.

    restart_after > 0: if that many kicks in a row bring no improvement, the search is
    stuck in a deep valley -> restart from a new starting tour ("restart" escape).
    restart_mode = "random" (random tour) or "bigkick" (best tour + many random kicks).
    """
    rng = random.Random(seed)
    t0 = time.time()
    deadline = t0 + time_limit
    ls = LocalSearch(D, start_tour, neigh, use_2opt, use_oropt, use_swap, use_3opt)
    ls.run(deadline=deadline)
    cur_tour = list(ls.tour)
    cur_len = ls.length
    best_tour, best_len = list(cur_tour), cur_len
    n = len(cur_tour)
    T0 = temperature * best_len / n          # relative to an average edge length
    near = neighbour_lists(D, min(30, n - 1)) if spatial_kick_prob > 0 else None
    history = [(time.time() - t0, best_len)]
    kicks = improvements = restarts = stall = 0
    last_cb = 0.0
    while time.time() < deadline:
        if restart_after and stall >= restart_after:
            stall = 0
            restarts += 1
            if restart_mode == "bigkick":
                new = list(best_tour)
                for _ in range(max(1, n // 10)):
                    new, _t = double_bridge_kick(new, rng, n)
            else:
                new = list(range(n))
                rng.shuffle(new)
            ls.set_tour(new)
            ls.run(deadline=deadline)
            cur_tour, cur_len = list(ls.tour), ls.length
            if cur_len < best_len:
                best_len, best_tour = cur_len, list(cur_tour)
                history.append((time.time() - t0, best_len))
            continue
        if near is not None and rng.random() < spatial_kick_prob:
            new, touched = spatial_double_bridge_kick(cur_tour, ls_pos(cur_tour, n), near, rng)
        else:
            w = n if rng.random() < global_kick_prob else kick_window
            new, touched = double_bridge_kick(cur_tour, rng, w)
        if not touched:
            continue
        ls.set_tour(new)
        ls.run(active=touched, deadline=deadline)
        kicks += 1
        new_len = ls.length
        accept = new_len <= cur_len
        if not accept and T0 > 0:
            T = T0 * max(0.0, 1.0 - (time.time() - t0) / time_limit)
            if T > 0 and rng.random() < math.exp(-(new_len - cur_len) / T):
                accept = True
        stall = 0 if new_len < cur_len else stall + 1
        if accept:
            cur_tour = list(ls.tour)
            cur_len = new_len
            if cur_len < best_len:
                best_len = cur_len
                best_tour = list(cur_tour)
                improvements += 1
                history.append((time.time() - t0, best_len))
        if callback is not None and time.time() - last_cb > callback_every:
            last_cb = time.time()
            callback(best_tour, best_len, kicks, time.time() - t0)
    history.append((time.time() - t0, best_len))
    return {
        "tour": best_tour,
        "length": best_len,
        "kicks": kicks,
        "improvements": improvements,
        "restarts": restarts,
        "history": history,
        "moves": dict(ls.move_count),
        "time": time.time() - t0,
    }


# --------------------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------------------
def solve(coords, time_limit=30.0, seed=1, k_neigh=10, nn_starts="all",
          use_2opt=True, use_oropt=True, use_swap=True, use_3opt=True, kick_window=50,
          global_kick_prob=0.0, temperature=0.0, callback=None):
    D = distance_matrix(coords)
    neigh = quadrant_neighbour_lists(D, coords, 6, 3)
    n = len(coords)
    t0 = time.time()
    starts = range(n) if nn_starts == "all" else [0]
    nn_tour, nn_len, nn_start, _ = best_nearest_neighbour(D, starts)
    t_nn = time.time() - t0

    ls = LocalSearch(D, nn_tour, neigh, use_2opt, use_oropt, use_swap, use_3opt)
    ls.run()
    ls_len = ls.length
    t_ls = time.time() - t0

    res = iterated_local_search(D, ls.tour, neigh, time_limit=time_limit, seed=seed,
                                use_2opt=use_2opt, use_oropt=use_oropt,
                                use_swap=use_swap, use_3opt=use_3opt,
                                kick_window=kick_window,
                                global_kick_prob=global_kick_prob,
                                temperature=temperature, callback=callback)
    final = res["tour"]
    assert is_valid_tour(final, n), "invalid tour"
    true_len = tour_length(final, D)
    assert true_len == res["length"], (true_len, res["length"])
    res.update({"nn_length": nn_len, "nn_start": nn_start, "nn_time": t_nn,
                "ls_length": ls_len, "ls_time": t_ls - t_nn, "D": D})
    return res

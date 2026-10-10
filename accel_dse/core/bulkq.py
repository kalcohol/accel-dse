"""Greedy bulk-service queue M/G^[b]/1 for a static prefill batch cap (0.65).

The prefill replica (``pdsim.PrefillReplica``) is greedy: whenever it is free and requests wait it starts a batch
of k = min(queue, b) of them at once, all leave together after T(k) (the batch wall, which depends on k); an
arrival to an idle server starts a batch of 1 immediately.  0.51–0.64 modelled this as an M/G/1 with per-request
service τ_b = T(b)/b and own latency T(b) — exact only when batches are always full.  At partial loads batches are
mostly small, the server is busy for T(1) ≈ T(b) per *request*, so the true wait is larger (and the own latency
smaller) than the fluid model's; p90 TTFT was optimistic (Qwen3-Next cap 2: 79.4 vs DES ≈ 86 ms).

Exact numerical solution (no approximation beyond truncation / quadrature):

* Embedded chain at service starts, state n ≥ 1 = requests waiting at the start (incl. the batch taken):
  k = min(n, b), r = n − k stay, A ~ mixed Poisson(λ·T) arrive during the service (T ~ D_k), next n' = r + A, or 1
  when r + A = 0 (idle, then the next arrival starts alone).  States 1..N are solved exactly (banded elimination,
  π₁ = 1 fixed, one balance equation dropped); for n > N the chain is homogeneous (k = b) and π_n = π_N·z^{n−N}, z = 1/x₀,
  x₀ > 1 the real root of A_b(x) = x^b (A_b the PGF of arrivals during a full-batch service).
* Tagged request (PASTA): it finds the server idle with the idle time fraction (→ served alone, T(1)); otherwise it
  arrives at age u (uniform) of a service started in state n with duration t: j ~ Poisson(λu) arrived before it in
  that service, p = r + j requests are ahead, F = ⌊p/b⌋ full batches (T(b) each) precede its own batch, which holds
  p mod b + 1 + L requests (capped at b), L ~ Poisson(λ·(t − u + F·T(b))) arrive behind it before that batch starts.
  TTFT core = (t − u) + Σ_F T(b) + T(k_own).  u is integrated on ``N_U`` sub-intervals with W uniform inside each, so the
  law is a mixture of uniform segments (piecewise-linear CDF) and point masses.

Service laws D_k are discrete [(t, w)]; with a prompt mix the batch wall is the mean of TTFT(k, S_i) over the batch
(pdsim.Costs.batch_prefill_wall), D_k = law of that mean (k-fold convolution, merged to ``MAX_ATOMS`` atoms).  The
sum of F i.i.d. full-batch walls is moment-matched by a 3-point law when D_b has spread 「近似」.

Pure Python, memoised; cost ~ N·(J + b)·N_U.
"""
from __future__ import annotations

import math
from functools import lru_cache

GRID = 160          # bins across the mean TTFT scale (bulk_law grid floor)
N_U = 8             # sub-intervals of the age u within a service
MAX_ATOMS = 24      # atoms kept for a batch-wall law D_k (0.65.1: 6 clipped the service tail)
PER_ATOMS = 48      # atoms of the per-request wall law fed to batch_wall_law
TOL = 1e-12


def _pois(mu: float, tol: float = TOL) -> list[float]:
    """Poisson(mu) pmf until the remaining tail < tol."""
    if mu <= 0:
        return [1.0]
    p = math.exp(-mu) if mu < 700 else 0.0
    if p == 0.0:                                 # large mu: start from the mode in log space
        out, j = [], 0
        lg = -mu
        acc = 0.0
        while True:
            v = math.exp(lg) if lg > -745 else 0.0
            out.append(v)
            acc += v
            j += 1
            lg += math.log(mu) - math.log(j)
            if j > mu and (1 - acc < tol or v < tol * 1e-3):
                break
        return out
    out, acc, j = [p], p, 0
    while 1 - acc > tol and j < 100000:
        j += 1
        p *= mu / j
        out.append(p)
        acc += p
        if j > mu and p < tol * 1e-3:
            break
    return out


def _mixpois(lam: float, atoms) -> list[float]:
    """Arrivals during a service whose duration has law atoms [(t, w)]."""
    out: list[float] = []
    for t, w in atoms:
        pm = _pois(lam * t)
        if len(pm) > len(out):
            out.extend([0.0] * (len(pm) - len(out)))
        for j, v in enumerate(pm):
            out[j] += w * v
    return out


def merge_atoms(atoms, n: int = MAX_ATOMS):
    """Reduce a discrete law to ≤ n atoms keeping the mass and mean of each quantile slice."""
    atoms = sorted((t, w) for t, w in atoms if w > 0)
    tot = sum(w for _, w in atoms)
    if len(atoms) <= n:
        return [(t, w / tot) for t, w in atoms]
    out, acc, cur_w, cur_m, edge = [], 0.0, 0.0, 0.0, 1
    for t, w in atoms:
        w /= tot
        while w > 1e-18:
            room = edge / n - acc
            take = min(w, room) if edge < n else w
            cur_w += take; cur_m += take * t; acc += take; w -= take
            if edge < n and acc >= edge / n - 1e-15:
                out.append((cur_m / cur_w, cur_w)); cur_w = cur_m = 0.0; edge += 1
    if cur_w > 1e-15:
        out.append((cur_m / cur_w, cur_w))
    return out


_GH = ((-2.0201828704560856, 0.019953242059045913), (-0.9585724646138185, 0.39361932315224116),
       (0.0, 0.9453087204829419), (0.9585724646138185, 0.39361932315224116),
       (2.0201828704560856, 0.019953242059045913))       # 5-point Gauss–Hermite (physicists'), Σw = √π
EXACT_K = 8         # batch_wall_law: exact convolution up to this many draws, moment-matched beyond


def batch_wall_law(k: int, per_req, n_atoms: int = MAX_ATOMS) -> list[tuple[float, float]]:
    """Law of the mean of k i.i.d. draws from per_req [(t, w)] (t = TTFT(k, S_i)).  Exact convolution (merged to
    ``n_atoms`` quantile slices) for k ≤ EXACT_K; beyond, 5-point Gauss–Hermite with the exact mean and variance / k
    (skewness ∝ 1/√k ignored 「近似」)."""
    per = merge_atoms(per_req, PER_ATOMS)
    if len(per) == 1 or k <= 0:
        return [(per[0][0], 1.0)] if per else [(0.0, 1.0)]
    if k > EXACT_K:
        mu = sum(t * w for t, w in per)
        sd = math.sqrt(max(0.0, sum(w * (t - mu) ** 2 for t, w in per)) / k)
        rp = math.sqrt(math.pi)
        lo = min(t for t, _ in per)
        return [(max(lo, mu + math.sqrt(2) * sd * x), w / rp) for x, w in _GH]
    cur = [(0.0, 1.0)]
    for _ in range(k):
        nxt: dict = {}
        for s_, a in cur:
            for t, w in per:
                key = s_ + t / k
                nxt[key] = nxt.get(key, 0.0) + a * w
        cur = merge_atoms(nxt.items(), 24)
    return merge_atoms(cur, n_atoms)


def _root(ab: list[float], b: int) -> float:
    """z = 1/x₀, x₀ > 1 the real root of Σ a_j x^j = x^b (decay rate of the stationary tail)."""
    def f(x: float) -> float:
        lx = math.log(x)
        return sum(a * math.exp((j - b) * lx) for j, a in enumerate(ab) if a > 0) - 1.0
    lo, hi = 1.0 + 1e-12, 2.0
    while f(hi) < 0:
        lo, hi = hi, hi * 2
        if hi > 1e6:
            return 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-14 * hi:
            break
    return 1.0 / (0.5 * (lo + hi))


def _stationary(lam: float, b: int, D: tuple) -> tuple[list[float], float, list[list[float]]]:
    """π over service starts n = 1..N (normalised incl. the geometric tail), tail ratio z, arrival pmfs a_k."""
    a = [None] + [_mixpois(lam, D[k - 1]) for k in range(1, b + 1)]
    ab = a[b]
    J = len(ab) - 1
    z = _root(ab, b)
    N = max(b + 2 * J + 16, 4 * b + 16)
    # unknowns x[i] = π_{i+2}, i = 0..N−2; equation for m = 2..N (row m−2); π₁ = 1 on the RHS
    n_u = N - 1
    rows: list[dict] = [dict() for _ in range(n_u)]
    rhs = [0.0] * n_u

    def add(m: int, n: int, v: float):
        if m < 2 or m > N or v == 0.0:
            return
        if n == 1:
            rhs[m - 2] += v
        else:
            rows[m - 2][n - 2] = rows[m - 2].get(n - 2, 0.0) - v
    for n in range(1, N + b + 1):
        k = min(n, b)
        r = n - k
        ak = a[k]
        tgt = n if n <= N else N
        mult = 1.0 if n <= N else z ** (n - N)
        for j, v in enumerate(ak):
            m = r + j
            if m == 0:
                m = 1
            if m > N:
                break
            add(m, tgt, v * mult)
    for i in range(n_u):
        rows[i][i] = rows[i].get(i, 0.0) + 1.0
    # banded elimination without pivoting (column diagonally dominant M-matrix)
    lower: list[list[int]] = [[] for _ in range(n_u)]      # rows r > i with an entry in column i
    for rI, row in enumerate(rows):
        for c in row:
            if c < rI:
                lower[c].append(rI)
    for i in range(n_u):
        piv = rows[i][i]
        prow = [(c, v) for c, v in rows[i].items() if c > i]
        for rI in sorted(set(lower[i])):
            if rI <= i:
                continue
            row = rows[rI]
            f = row.pop(i, 0.0) / piv
            if f == 0.0:
                continue
            for c, v in prow:
                nv = row.get(c, 0.0) - f * v
                if c not in row and c < rI:
                    lower[c].append(rI)
                row[c] = nv
            rhs[rI] -= f * rhs[i]
    x = [0.0] * n_u
    for i in range(n_u - 1, -1, -1):
        s = rhs[i] - sum(v * x[c] for c, v in rows[i].items() if c > i)
        x[i] = s / rows[i][i]
    pi = [1.0] + x
    pi = [max(0.0, v) for v in pi]
    tail = pi[-1] * z / (1 - z) if 0 < z < 1 else 0.0
    tot = sum(pi) + tail
    return [v / tot for v in pi], z, a


def _sum3(F: int, mu: float, var: float):
    """3-point law of a sum of F i.i.d. (mean mu, variance var): mean ± √(3F·var), weights 1/6, 2/3, 1/6."""
    if F == 0:
        return [(0.0, 1.0)]
    if var <= 0:
        return [(F * mu, 1.0)]
    d = math.sqrt(3 * F * var)
    return [(F * mu - d, 1 / 6), (F * mu, 2 / 3), (F * mu + d, 1 / 6)]


@lru_cache(maxsize=1 << 15)
def _pois_c(mu: float, tol: float = TOL) -> tuple:
    return tuple(_pois(mu, tol))


@lru_cache(maxsize=1 << 16)
def _pois_head(mu: float, tol: float, n: int) -> tuple:
    """The first min(n, len) terms of ``_pois(mu, tol)`` — same recurrence, same floating-point operations, so the
    terms are bit-identical — without building the rest of the pmf (0.66 perf: ``_tagged`` uses ≤ b − 1 terms of a
    pmf that is hundreds of terms long)."""
    if n <= 0:
        return ()
    if mu <= 0:
        return (1.0,)
    p = math.exp(-mu) if mu < 700 else 0.0
    out = []
    if p == 0.0:
        # all n head terms underflow to 0.0 exactly when j < mu (lg rises monotonically) and the largest exponent
        # bound −mu + n·ln mu stays below −746 (1 of margin ≫ the accumulated rounding); the loop would emit n zeros
        if n < mu and -mu + n * math.log(mu) < -746:
            return (0.0,) * n
        j, lg, acc = 0, -mu, 0.0
        while True:
            v = math.exp(lg) if lg > -745 else 0.0
            out.append(v)
            if len(out) >= n:
                return tuple(out)
            acc += v
            j += 1
            lg += math.log(mu) - math.log(j)
            if j > mu and (1 - acc < tol or v < tol * 1e-3):
                return tuple(out)
    out, acc, j = [p], p, 0
    while len(out) < n and 1 - acc > tol and j < 100000:
        j += 1
        p *= mu / j
        out.append(p)
        acc += p
        if j > mu and p < tol * 1e-3:
            break
    return tuple(out)


@lru_cache(maxsize=512)
def _chain(lam: float, b: int, D: tuple):
    pi, z, a = _stationary(lam, b, D)
    ext = []
    if 0 < z < 1:
        v = pi[-1] * z
        while v > 1e-14 and len(ext) < 20 * len(pi):
            ext.append(v)
            v *= z
    pis = pi + ext
    means = [sum(t * w for t, w in D[k - 1]) for k in range(1, b + 1)]
    m2s = [sum(t * t * w for t, w in D[k - 1]) for k in range(1, b + 1)]
    Es = Lq = served = own = 0.0
    for n0, p in enumerate(pis):
        n = n0 + 1
        k = min(n, b)
        Es += p * means[k - 1]
        Lq += p * ((n - k) * means[k - 1] + lam * m2s[k - 1] / 2)
        served += p * k
        own += p * k * means[k - 1]
    Pidle = sum(pis[n - 1] * a[n][0] for n in range(1, min(b, len(pi)) + 1))
    cyc = Es + Pidle / lam
    return pis, a, means, cyc, Pidle, Lq / cyc / lam, own / served, served / sum(pis)


def bulk_mean(lam: float, b: int, D: tuple) -> dict | None:
    """Mean TTFT core of the greedy bulk server from the embedded chain alone (Little: W_q = L_q / λ, L_q = the
    time-average count waiting outside the running batch; own = the request-average batch wall).  None if
    unstable (λ·E[T(b)] ≥ b)."""
    mb = sum(t * w for t, w in D[b - 1])
    if lam * mb >= b * (1 - 1e-12):
        return None
    if lam <= 0:
        m = sum(t * w for t, w in D[0])
        return {"mean": m, "wait_mean": 0.0, "own_mean": m, "p_wait": 0.0, "batch_mean": 1.0}
    pis, a, means, cyc, Pidle, wq, own, bm = _chain(lam, b, D)
    return {"mean": wq + own, "wait_mean": wq, "own_mean": own, "p_wait": 1 - (Pidle / lam) / cyc,
            "batch_mean": bm}


def bulk_busy(lam: float, b: int, D: tuple) -> float:
    """Time-busy fraction of the greedy bulk server (E[service] per cycle / cycle length)."""
    if lam <= 0:
        return 0.0
    pis, a, means, cyc, Pidle, *_ = _chain(lam, b, D)
    return 1 - (Pidle / lam) / cyc


def bulk_own_k(lam: float, b: int, D: tuple) -> list:
    """Request-average law of the size of the batch a request is served in: [(k, P)]."""
    if lam <= 0:
        return [(1, 1.0)]
    pis = _chain(lam, b, D)[0]
    acc = [0.0] * (b + 1)
    for n0, p in enumerate(pis):
        k = min(n0 + 1, b)
        acc[k] += p * k
    tot = sum(acc)
    return [(k, v / tot) for k, v in enumerate(acc) if v > 0]


def _tagged(lam: float, b: int, D: tuple):
    """Tagged-request decomposition: {(t, u-slot, ΣF, own batch size): mass} over busy arrivals + the idle fraction."""
    mb = sum(t * w for t, w in D[b - 1])
    pis, a, means, cyc, Pidle, wq, own, bm = _chain(lam, b, D)
    f_idle = (Pidle / lam) / cyc
    mb_var = sum(w * (t - mb) ** 2 for t, w in D[b - 1])
    rlaw: dict = {}
    for n0, p in enumerate(pis):
        n = n0 + 1
        k = min(n, b)
        d = rlaw.setdefault(k, {})
        d[n - k] = d.get(n - k, 0.0) + p
    # services started with k < b take everyone (r = 0): what follows depends only on their duration, so those
    # classes are pooled into one duration law (π_k-weighted D_k, merged — mass and Σ w·t, the length-biased weight,
    # are kept); 0.65 perf: b classes → 2
    classes = []
    small = [(t, rlaw[k][0] * w) for k in rlaw if k < b for t, w in D[k - 1]]
    if small:
        m0 = sum(w for _, w in small)
        classes.append(([(0, m0)], merge_atoms(small, 2 * MAX_ATOMS)))   # merge_atoms normalises
    if b in rlaw:
        classes.append(([(r, pr) for r, pr in rlaw[b].items() if pr > 1e-13], list(D[b - 1])))
    keys: dict = {}
    for rd, atoms in classes:
        asc = all(rd[i][0] < rd[i + 1][0] for i in range(len(rd) - 1)) and bool(rd)
        for t, w in atoms:
            base_w = w * t / cyc / N_U
            for ui in range(N_U):
                u = t * (ui + 0.5) / N_U
                pj = _pois_c(round(lam * u, 12), 1e-11)
                if asc:
                    # 0.66 perf: dense list instead of a dict — per index the same products added in the same r order
                    # (0.0 + x is exact), and with rd ascending the dict's insertion order is ascending index; untouched
                    # slots stay 0.0 and fall under the mass cut below exactly like absent keys
                    n_j = len(pj)
                    acc = [0.0] * (rd[-1][0] + n_j)
                    for r, pr in rd:
                        acc[r:r + n_j] = [a + pr * v for a, v in zip(acc[r:r + n_j], pj)]
                    pl = enumerate(acc)
                else:
                    pd_: dict = {}
                    for r, pr in rd:
                        for j, v in enumerate(pj):
                            pd_[r + j] = pd_.get(r + j, 0.0) + pr * v
                    pl = pd_.items()
                # 0.66 perf: per own-batch block F the three ΣF atoms and their arrival pmf heads are shared by every
                # m0 (heads of one pmf are prefixes of each other, cdf prefix sums are the same sequential sums), so
                # they are built once per (t, u, F) — every number is bit-identical to the per-pp construction
                fc: dict = {}
                for pp, pv in pl:
                    mass = base_w * pv
                    if mass < 1e-13:
                        continue
                    F, m0 = divmod(pp, b)
                    need = b - m0 - 1
                    blk = fc.get(F)
                    if blk is None:
                        blk = []
                        for sF, wF in _sum3(F, mb, mb_var):
                            if b > 1:
                                pL = _pois_head(round(lam * (t - u + sF), 9), 1e-11, b - 1)
                                cd, c = [], 0.0
                                for v in pL:
                                    c += v
                                    cd.append(c)
                            else:
                                pL, cd = (), []
                            blk.append((sF, wF, pL, cd))
                        fc[F] = blk
                    for sF, wF, pL, cd in blk:
                        if need == 0:
                            kd = ((b, 1.0),)
                        else:
                            n = min(need, len(pL))
                            kd = [(m0 + 1 + L, pL[L]) for L in range(n)]
                            cdf = cd[n - 1] if n else 0.0
                            if 1 - cdf > 1e-12:
                                kd.append((m0 + 1 + n, 1 - cdf))
                        for ko, pk in kd:
                            key = (t, ui, sF, ko)
                            keys[key] = keys.get(key, 0.0) + mass * wF * pk
    return keys, f_idle, wq, own, bm


@lru_cache(maxsize=128)
def _wait_hists(lam: float, b: int, D: tuple):
    """Per own-batch-size k: the wait part (time until the tagged request's own batch starts) on the grid of width
    h = E[T(1)]/(4·N_U) — {k: (first bin, [mass …], Σ mass·mean)} — plus the idle fraction (zero wait)."""
    keys, f_idle, wq, own, bm = _tagged(lam, b, D)
    # grid: E[T(1)]/(4·N_U), coarsened at high load to ~1/GRID of the mean TTFT scale (wait + full-batch wall) so
    # the shift-add below stays O(GRID·atoms) per batch size 「近似」(quantiles interpolate linearly inside a bin)
    mb = sum(t * w for t, w in D[b - 1])
    h = max(sum(t * w for t, w in D[0]) / (4 * N_U), (wq + mb) / GRID)
    acc: dict = {}
    for (t, ui, sF, ko), ms in keys.items():
        hist, mom = acc.setdefault(ko, ({}, [0.0]))
        wd = t / N_U
        lo = t - t * (ui + 1) / N_U + sF
        mom[0] += ms * (lo + wd / 2)
        x0, x1 = lo / h, (lo + wd) / h
        i0, i1 = int(x0), int(x1)
        if i0 == i1:
            hist[i0] = hist.get(i0, 0.0) + ms
            continue
        dens = ms / (x1 - x0)
        hist[i0] = hist.get(i0, 0.0) + dens * (i0 + 1 - x0)
        for i in range(i0 + 1, i1):
            hist[i] = hist.get(i, 0.0) + dens
        hist[i1] = hist.get(i1, 0.0) + dens * (x1 - i1)
    out = {}
    for ko, (hist, mom) in acc.items():
        tot = sum(hist.values())
        if tot < 1e-12:
            continue
        a, z = min(hist), max(hist)
        out[ko] = (a, [hist.get(i, 0.0) for i in range(a, z + 1)], mom[0], tot)
    return out, f_idle, wq, own, bm, h


@lru_cache(maxsize=256)
def bulk_law(lam: float, b: int, D: tuple, O: tuple | None = None) -> dict | None:
    """TTFT law of the greedy bulk server.  D[k−1] = ((t, w), …) = batch-wall law for k = 1..b.  ``O`` (optional)
    O[k−1] = law of the tagged request's own part when it is served in a batch of k — its batch wall plus anything
    after it that depends only on its own prompt (KV exposure); default O = D.  The own part is independent of the
    wait given k (the request's own prompt is independent of the queue it finds, PASTA), so typed own walls
    (T(k, S_i) + (k − 1)·others)/k with their exposure are exact here, not an independence approximation.
    → {"mean", "p_wait", "hist": (h, ((bin, mass), …) sorted), "atoms": ((t, w), …) point masses} or None if
    unstable.  Wait part per own batch size on a grid of width h = E[T(1)]/(4·N_U) (uniform segments, one per age
    sub-interval), shifted by each own-part atom (linear split between two bins), plus point masses for idle
    arrivals (O[0])."""
    mb = sum(t * w for t, w in D[b - 1])
    if lam * mb >= b * (1 - 1e-12):
        return None
    O = D if O is None else O
    o1 = sum(t * w for t, w in O[0])
    if lam <= 0:
        return {"mean": o1, "p_wait": 0.0, "hist": (sum(t * w for t, w in D[0]) / (4 * N_U), ()),
                "atoms": tuple(O[0]), "wait_mean": 0.0}
    W, f_idle, wq, own, bm, h = _wait_hists(lam, b, D)
    mean = f_idle * o1
    lo_i = hi_i = None
    for ko, (a, arr, mom, tot) in W.items():
        Ok = O[ko - 1]
        x0 = a + int(min(t for t, _ in Ok) / h)
        x1 = a + len(arr) + int(max(t for t, _ in Ok) / h) + 2
        lo_i = x0 if lo_i is None else min(lo_i, x0)
        hi_i = x1 if hi_i is None else max(hi_i, x1)
    hist = [0.0] * ((hi_i - lo_i) if lo_i is not None else 0)
    for ko, (a, arr, mom, tot) in W.items():
        Ok = O[ko - 1]
        mean += mom + tot * sum(t * w for t, w in Ok)
        n = len(arr)
        for to, wo in Ok:
            x = to / h
            s0 = int(x)
            f = x - s0
            base = a + s0 - lo_i
            g0, g1 = wo * (1 - f), wo * f
            seg = hist[base:base + n + 1]
            hist[base:base + n + 1] = [c + (arr[i] * g0 if i < n else 0.0) + (arr[i - 1] * g1 if i > 0 else 0.0)
                                       for i, c in enumerate(seg)]
    atoms = tuple((t, f_idle * w) for t, w in O[0])
    hz = tuple((lo_i + i, v) for i, v in enumerate(hist) if v > 0)
    return {"mean": mean, "p_wait": 1 - f_idle, "wait_mean": wq, "own_mean": own, "batch_mean": bm,
            "hist": (h, hz), "atoms": atoms}


def law_quantile(law: dict, q: float, shift=((0.0, 1.0),)) -> float:
    """q-quantile of a bulk_law (histogram, uniform within a bin, + point masses) ⊕ an independent discrete shift."""
    h, hist = law["hist"]
    pts = []                                   # (x_lo, x_hi, mass); x_lo == x_hi → point mass
    for s, ws in shift:
        for i, m in hist:
            pts.append((i * h + s, (i + 1) * h + s, m * ws))
        for t, w in law["atoms"]:
            pts.append((t + s, t + s, w * ws))
    pts.sort()
    tot = sum(m for _, _, m in pts)
    target = q * tot
    # walk: CDF is piecewise linear; overlapping bins only arise from the shift mixture → bisection on x
    lo_all, hi_all = pts[0][0], max(p[1] for p in pts)

    def cdf(x: float) -> float:
        c = 0.0
        for a, bb, m in pts:
            if a > x:
                break
            if x >= bb:
                c += m
            else:
                c += m * (x - a) / (bb - a)
        return c
    if len(shift) == 1:                        # no overlap: direct walk
        acc = 0.0
        for a, bb, m in pts:
            if acc + m >= target:
                return a if bb == a else a + (bb - a) * (target - acc) / m
            acc += m
        return hi_all
    a, bb = lo_all, hi_all
    for _ in range(60):
        mid = 0.5 * (a + bb)
        if cdf(mid) < target:
            a = mid
        else:
            bb = mid
        if bb - a < 1e-9 * max(bb, 1e-9):
            break
    return bb


def seg_quantile(segs, q: float, shift=((0.0, 1.0),)) -> float:
    """q-quantile of a mixture of uniform segments (lo, hi, w) (+ an independent discrete shift law)."""
    pts = [(lo + s, hi + s, w * ws) for lo, hi, w in segs for s, ws in shift]
    lo_all = min(p[0] for p in pts)
    hi_all = max(p[1] for p in pts)

    def cdf(x: float) -> float:
        c = 0.0
        for lo, hi, w in pts:
            if x >= hi:
                c += w
            elif x > lo:
                c += w * (x - lo) / (hi - lo)
        return c
    a, bb = lo_all, hi_all
    for _ in range(60):
        mid = 0.5 * (a + bb)
        if cdf(mid) < q:
            a = mid
        else:
            bb = mid
        if bb - a < 1e-9 * max(bb, 1e-9):
            break
    return bb

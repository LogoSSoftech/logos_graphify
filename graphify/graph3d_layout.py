"""Deterministic 3D layouts whose *shape* is derived from the graph itself.

A force-directed layout applies uniform repulsion plus a weak central gravity, so
it minimizes energy isotropically and every project converges to the same ball —
the drawing carries no information beyond local adjacency.

``spectral_signature_positions`` instead embeds the graph with the eigenvectors of
its normalized Laplacian (Koren, *Drawing Graphs by Eigenvectors*) and — crucially —
does **not** rescale the axes afterwards. Each axis is scaled by ``1/sqrt(lambda)``
(the diffusion-map convention), so the natural anisotropy of the codebase survives
into the picture: a modular codebase with weak bridges stretches into lobes joined
by necks, a small tightly-knit one stays round, a layered one flattens into a ribbon.
The resulting silhouette is a stable fingerprint of the architecture — identical for
identical code, and visibly drifting as the coupling structure changes.

No SciPy: the normalized Laplacian is applied as an O(E) matrix-free product, and
the eigenvectors come from dense ``numpy.linalg.eigh`` on small components or
subspace iteration on large ones.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Hashable, List, Sequence, Tuple

import numpy as np

# Dense eigendecomposition is O(n^3); above this many nodes a component is solved
# with matrix-free subspace iteration instead (O(E) per iteration).
DENSE_MAX_NODES = 1200
_SUBSPACE_ITERS = 400
_EPS = 1e-12


def _normalized_adjacency_matvec(
    n: int, edges: Sequence[Tuple[int, int]], dinv: np.ndarray
):
    """Return ``f(X) -> M @ X`` for ``M = D^-1/2 A D^-1/2``, without building M.

    Operates on a whole block of vectors at once so subspace iteration costs one
    pass over the edge list per iteration regardless of how many vectors it tracks.
    """

    src = np.fromiter((u for u, _ in edges), dtype=np.int64, count=len(edges))
    dst = np.fromiter((v for _, v in edges), dtype=np.int64, count=len(edges))
    w = dinv[src] * dinv[dst]

    def matvec(X: np.ndarray) -> np.ndarray:
        Y = np.zeros_like(X)
        # undirected: each edge contributes in both directions
        np.add.at(Y, src, w[:, None] * X[dst])
        np.add.at(Y, dst, w[:, None] * X[src])
        return Y

    return matvec


def _embed_component(
    n: int, edges: Sequence[Tuple[int, int]], deg: np.ndarray, dim: int
) -> np.ndarray:
    """Embed one connected component; returns an ``(n, dim)`` array.

    Columns are the eigenvectors of the normalized Laplacian for the ``dim``
    smallest non-trivial eigenvalues, each divided by ``sqrt(lambda)``. Components
    too small to carry ``dim`` non-trivial eigenvectors fall back to a deterministic
    spherical spread, so a 2-node island still gets sane coordinates.
    """
    if n <= dim + 1:
        return _tiny_component_coords(n, dim)

    dinv = 1.0 / np.sqrt(np.maximum(deg, _EPS))

    if n <= DENSE_MAX_NODES:
        A = np.zeros((n, n), dtype=np.float64)
        for u, v in edges:
            A[u, v] += 1.0
            A[v, u] += 1.0
        M = (A * dinv).T * dinv
        lam_m, vecs = np.linalg.eigh(M)          # ascending
        order = np.argsort(lam_m)[::-1][1:dim + 1]   # drop the trivial top vector
        lam = 1.0 - lam_m[order]                 # eigenvalues of L = I - M
        V = vecs[:, order]
    else:
        lam, V = _subspace_iteration(n, edges, dinv, dim)

    scale = 1.0 / np.sqrt(np.maximum(lam, _EPS))
    return V * scale


def _subspace_iteration(
    n: int, edges: Sequence[Tuple[int, int]], dinv: np.ndarray, dim: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Matrix-free eigenvectors for the smallest non-trivial eigenvalues of L.

    The smallest eigenvalues of ``L = I - M`` are the largest of ``M``, so we run
    subspace iteration on ``(M + I) / 2`` — a monotone map of the spectrum onto
    ``[0, 1]`` that keeps the ordering while making every eigenvalue non-negative,
    so a strongly bipartite component's ``lambda ≈ -1`` mode can't outrank the ones
    we want. The known top eigenvector of ``M`` (``sqrt(deg)``, eigenvalue 1) is
    deflated at every step; it is the constant mode of L and carries no shape.
    """
    matvec = _normalized_adjacency_matvec(n, edges, dinv)
    trivial = np.sqrt(np.maximum(1.0 / np.maximum(dinv ** 2, _EPS), 0.0))
    trivial /= max(float(np.linalg.norm(trivial)), _EPS)

    rng = np.random.default_rng(0)               # deterministic: same code, same shape
    X = rng.standard_normal((n, dim))
    X -= np.outer(trivial, trivial @ X)
    X, _ = np.linalg.qr(X)

    for _ in range(_SUBSPACE_ITERS):
        X = 0.5 * (matvec(X) + X)                # (M + I)/2
        X -= np.outer(trivial, trivial @ X)      # deflate the constant mode
        X, _ = np.linalg.qr(X)

    # Rayleigh quotients on M give the L eigenvalues we scale by.
    MX = matvec(X)
    lam_m = np.einsum("ij,ij->j", X, MX)
    lam = np.clip(1.0 - lam_m, _EPS, None)
    order = np.argsort(lam)
    return lam[order], X[:, order]


def _tiny_component_coords(n: int, dim: int) -> np.ndarray:
    """Deterministic spread for a component with no usable spectrum."""
    out = np.zeros((n, dim), dtype=np.float64)
    if n == 1:
        return out
    # golden-angle spiral on a sphere: stable and evenly spaced for any n
    ga = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        y = 1.0 - (2.0 * i) / max(n - 1, 1)
        r = math.sqrt(max(0.0, 1.0 - y * y))
        out[i, 0] = math.cos(ga * i) * r
        if dim > 1:
            out[i, 1] = y
        if dim > 2:
            out[i, 2] = math.sin(ga * i) * r
    return out


def _components(nodes: List[Hashable], adj: Dict[Hashable, List[Hashable]]) -> List[List[Hashable]]:
    """Connected components, largest first; deterministic given the node order."""
    seen: set = set()
    comps: List[List[Hashable]] = []
    for start in nodes:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in adj.get(cur, ()):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        comps.append(comp)
    comps.sort(key=lambda c: (-len(c), str(c[0])))
    return comps


def spectral_signature_positions(
    nodes: List[Hashable],
    links: Sequence[Tuple[Hashable, Hashable]],
    *,
    dim: int = 3,
    world: float = 600.0,
) -> Dict[Hashable, Tuple[float, ...]]:
    """Fixed 3D coordinates whose global shape is intrinsic to this graph.

    The giant component is embedded at the origin; every other component is
    embedded on its own and parked on a golden-angle shell around it, ordered by
    size, so the islands read as satellites instead of polluting the main body.
    A single uniform scale maps the result into ``world`` — uniform on purpose,
    since a per-axis rescale is exactly what would flatten the signature back into
    a featureless ball.
    """
    adj: Dict[Hashable, List[Hashable]] = {n: [] for n in nodes}
    clean: List[Tuple[Hashable, Hashable]] = []
    for u, v in links:
        if u in adj and v in adj and u != v:
            adj[u].append(v)
            adj[v].append(u)
            clean.append((u, v))

    comps = _components(nodes, adj)
    if not comps:
        return {}

    embeddings: List[np.ndarray] = []
    for comp in comps:
        index = {nid: i for i, nid in enumerate(comp)}
        c_edges = [(index[u], index[v]) for u, v in clean if u in index and v in index]
        deg = np.zeros(len(comp), dtype=np.float64)
        for u, v in c_edges:
            deg[u] += 1.0
            deg[v] += 1.0
        embeddings.append(_embed_component(len(comp), c_edges, deg, dim))

    # One uniform scale for everything, taken from the giant component's own extent.
    main = embeddings[0]
    main_extent = float(np.abs(main).max()) or 1.0
    scale = (world * 0.5) / main_extent

    pos: Dict[Hashable, Tuple[float, ...]] = {}
    for i, nid in enumerate(comps[0]):
        pos[nid] = tuple(float(x) for x in main[i] * scale)

    # Satellites: golden-angle shell just outside the main body — but stretched by
    # the body's own per-axis extent. A plain spherical shell would re-round the
    # silhouette (a codebase with hundreds of islands would look like a ball again,
    # which is precisely the failure this layout exists to avoid), so the islands
    # hug the shape instead of hiding it.
    ga = math.pi * (3.0 - math.sqrt(5.0))
    body = np.abs(main * scale).max(axis=0)
    body_max = float(body.max())
    if body_max <= _EPS:
        # Degenerate main body (a lone node, or a graph with no edges at all): there
        # is no shape to follow, and dividing through would park every satellite at
        # the origin — all nodes stacked on one invisible point. Fall back to a
        # plain isotropic shell, which is the honest picture of a graph with no
        # structure to show.
        body = np.ones(dim, dtype=np.float64)
    else:
        # Floor each axis so a near-flat body still spreads its islands instead of
        # crushing them onto a line or a plane.
        body = np.maximum(body / body_max, 0.12)
    shell = world * 0.62
    for k, (comp, emb) in enumerate(zip(comps[1:], embeddings[1:]), start=1):
        y = 1.0 - (2.0 * k) / max(len(comps) - 1, 1) if len(comps) > 2 else 0.0
        r = math.sqrt(max(0.0, 1.0 - y * y))
        cx = math.cos(ga * k) * r * shell * float(body[0])
        cy = y * shell * float(body[1] if dim > 1 else 1.0)
        cz = math.sin(ga * k) * r * shell * float(body[2] if dim > 2 else 1.0)
        extent = float(np.abs(emb).max()) or 1.0
        local = (world * 0.045) / extent
        for i, nid in enumerate(comp):
            p = emb[i] * local
            pos[nid] = (
                float(p[0] + cx),
                float(p[1] + cy) if dim > 1 else cy,
                float(p[2] + cz) if dim > 2 else cz,
            )
    return pos


def shape_signature(pos: Dict[Hashable, Tuple[float, ...]]) -> Dict[str, Any]:
    """Report the layout's aspect ratio — the project's numeric fingerprint."""
    if not pos:
        return {"nodes": 0, "aspect": []}
    A = np.array(list(pos.values()), dtype=np.float64)
    s = A.std(axis=0)
    top = float(s.max()) or 1.0
    return {"nodes": int(A.shape[0]), "aspect": [round(float(x / top), 3) for x in s]}

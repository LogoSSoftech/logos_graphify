"""Deterministic 3D layouts whose *shape* is derived from the graph itself.

A force-directed layout applies uniform repulsion plus a weak central gravity, so
it minimizes energy isotropically and every project converges to the same ball —
the drawing carries no information beyond local adjacency.

``spectral_signature_positions`` instead embeds the graph on the eigenvectors of
its normalized Laplacian (Koren, *Drawing Graphs by Eigenvectors*) and — crucially —
does **not** rescale the axes, so the codebase's intrinsic anisotropy survives into
the picture. Three refinements keep that from degenerating into a featureless blob:

*Many modes, not three.* The first few eigenvectors are the smoothest and carry the
least character. We solve for ``modes`` of them, scale each by ``1/sqrt(lambda)``,
and project to 3D with PCA, so higher-frequency structure reaches the picture.

*Relation-aware coupling.* Edge weight depends on the relation type — ``inherits``
binds far tighter than ``imports`` — so the *mix* of relations a codebase uses,
which is very much its own, bends the geometry.

*Decompression.* Low-frequency eigenvectors are near-constant across the bulk of a
graph, so a raw spectral embedding piles most nodes at the centre and lets a handful
of pendant nodes set the scale: 60% of one real 6k-node graph landed inside the
innermost 10% of the radius, rendering as a dense dot plus outliers. A partial
per-axis quantile transform spreads that core back out while each axis is rescaled
to its *original* spread, so the density becomes readable without flattening the
aspect ratio that makes the silhouette a fingerprint.

No SciPy: the Laplacian is applied as an O(E) matrix-free product, and the
eigenvectors come from dense ``numpy.linalg.eigh`` on small components or subspace
iteration on large ones.
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
#: World size is set from this percentile of the embedding, never the maximum.
_SCALE_PERCENTILE = 95.0

#: How tightly each relation binds its endpoints. A subclass sits practically on
#: top of its parent; a module that merely imports another is only loosely tied.
#: The weights are what let two codebases with the same node count but different
#: relation mixes come out shaped differently.
RELATION_WEIGHTS: Dict[str, float] = {
    "inherits": 3.0,
    "method": 2.0,
    "contains": 2.0,
    "defines": 2.0,
    "calls": 1.0,
    "re_exports": 1.0,
    "references": 0.8,
    "uses": 0.8,
    "imports_from": 0.6,
    "imports": 0.6,
    "indirect_call": 0.5,
    "rationale_for": 0.4,
}
DEFAULT_RELATION_WEIGHT = 1.0


def relation_weight(relation: Any) -> float:
    if not relation:
        return DEFAULT_RELATION_WEIGHT
    return RELATION_WEIGHTS.get(str(relation).strip().lower(), DEFAULT_RELATION_WEIGHT)


def _normalized_adjacency_matvec(edges, weights: np.ndarray, dinv: np.ndarray):
    """Return ``f(X) -> M @ X`` for ``M = D^-1/2 A D^-1/2``, without building M."""
    src = np.fromiter((u for u, _ in edges), dtype=np.int64, count=len(edges))
    dst = np.fromiter((v for _, v in edges), dtype=np.int64, count=len(edges))
    w = weights * dinv[src] * dinv[dst]

    def matvec(X: np.ndarray) -> np.ndarray:
        Y = np.zeros_like(X)
        np.add.at(Y, src, w[:, None] * X[dst])
        np.add.at(Y, dst, w[:, None] * X[src])
        return Y

    return matvec


def _embed_component(
    n: int,
    edges: Sequence[Tuple[int, int]],
    weights: np.ndarray,
    deg: np.ndarray,
    dim: int,
    modes: int,
) -> np.ndarray:
    """Embed one connected component; returns an ``(n, dim)`` array."""
    if n <= dim + 1:
        return _tiny_component_coords(n, dim)

    k = max(dim, min(modes, n - 2))
    dinv = 1.0 / np.sqrt(np.maximum(deg, _EPS))

    if n <= DENSE_MAX_NODES:
        A = np.zeros((n, n), dtype=np.float64)
        for (u, v), w in zip(edges, weights):
            A[u, v] += w
            A[v, u] += w
        M = (A * dinv).T * dinv
        lam_m, vecs = np.linalg.eigh(M)
        order = np.argsort(lam_m)[::-1][1:k + 1]
        lam = np.clip(1.0 - lam_m[order], _EPS, None)
        V = vecs[:, order]
    else:
        lam, V = _subspace_iteration(n, edges, weights, dinv, k)

    scaled = V / np.sqrt(np.maximum(lam, _EPS))
    return _pca(scaled, dim)


def _pca(X: np.ndarray, dim: int) -> np.ndarray:
    """Project a many-mode embedding onto its ``dim`` highest-variance directions.

    Keeping only eigenvectors 2..4 throws away the higher-frequency modes that carry
    a graph's character; solving for more and projecting keeps that structure while
    still handing back 3 coordinates.
    """
    if X.shape[1] <= dim:
        out = np.zeros((X.shape[0], dim), dtype=np.float64)
        out[:, :X.shape[1]] = X
        return out
    Xc = X - X.mean(axis=0)
    # right singular vectors are the principal axes; U*S is the projection
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    return U[:, :dim] * S[:dim]


def _subspace_iteration(
    n: int,
    edges: Sequence[Tuple[int, int]],
    weights: np.ndarray,
    dinv: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Matrix-free eigenvectors for the smallest non-trivial eigenvalues of L.

    Iterates on ``(M + I) / 2`` — a monotone map of the spectrum onto ``[0, 1]``
    that keeps the ordering while making every eigenvalue non-negative, so a
    strongly bipartite component's ``lambda ~ -1`` mode cannot outrank the ones we
    want. The known top eigenvector of ``M`` (``sqrt(deg)``) is deflated each step;
    it is the constant mode of L and carries no shape.
    """
    matvec = _normalized_adjacency_matvec(edges, weights, dinv)
    trivial = 1.0 / np.maximum(dinv, _EPS)
    trivial = trivial / max(float(np.linalg.norm(trivial)), _EPS)

    rng = np.random.default_rng(0)               # deterministic: same code, same shape
    X = rng.standard_normal((n, k))
    X -= np.outer(trivial, trivial @ X)
    X, _ = np.linalg.qr(X)

    for _ in range(_SUBSPACE_ITERS):
        X = 0.5 * (matvec(X) + X)
        X -= np.outer(trivial, trivial @ X)
        X, _ = np.linalg.qr(X)

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


def decompress(X: np.ndarray, amount: float) -> np.ndarray:
    """Spread a centre-piled point cloud without flattening its aspect ratio.

    Blends each axis toward its own quantile (rank) transform, then rescales that
    transform back to the axis's *original* standard deviation. The density becomes
    readable — filaments and clusters separate instead of overlapping in one dot —
    while the per-axis spread, which is what makes the silhouette a fingerprint,
    is preserved. ``amount=0`` is a no-op; ``amount=1`` is a full quantile transform.
    """
    if amount <= 0 or X.size == 0:
        return X
    out = np.array(X, dtype=np.float64, copy=True)
    for j in range(out.shape[1]):
        col = out[:, j]
        sd = float(col.std())
        if sd <= _EPS:
            continue
        ranks = np.argsort(np.argsort(col)).astype(np.float64)
        ranks = ranks / max(len(ranks) - 1, 1) - 0.5
        rsd = float(ranks.std()) or 1.0
        out[:, j] = (1.0 - amount) * col + amount * (ranks / rsd * sd)
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
    links: Sequence[Sequence[Any]],
    *,
    dim: int = 3,
    world: float = 600.0,
    modes: int = 12,
    decompress_amount: float = 0.55,
    weighted: bool = True,
) -> Dict[Hashable, Tuple[float, ...]]:
    """Fixed 3D coordinates whose global shape is intrinsic to this graph.

    ``links`` items are ``(source, target)`` or ``(source, target, relation)``; the
    relation, when present and ``weighted`` is set, picks the coupling strength from
    :data:`RELATION_WEIGHTS`.

    The giant component is embedded at the origin. Remaining components are laid on
    a golden-angle **spiral arm** ordered by size — largest nearest — rather than a
    spherical shell: a shell paints the same generic halo onto every project, while
    an arm whose length and grain follow the component-size distribution is itself
    a property of the codebase (one real app has 226 islands, another 37).
    """
    adj: Dict[Hashable, List[Hashable]] = {n: [] for n in nodes}
    clean: List[Tuple[Hashable, Hashable, float]] = []
    for link in links:
        u, v = link[0], link[1]
        rel = link[2] if len(link) > 2 else None
        if u in adj and v in adj and u != v:
            adj[u].append(v)
            adj[v].append(u)
            clean.append((u, v, relation_weight(rel) if weighted else 1.0))

    comps = _components(nodes, adj)
    if not comps:
        return {}

    embeddings: List[np.ndarray] = []
    for comp in comps:
        index = {nid: i for i, nid in enumerate(comp)}
        c_edges: List[Tuple[int, int]] = []
        c_w: List[float] = []
        for u, v, w in clean:
            if u in index and v in index:
                c_edges.append((index[u], index[v]))
                c_w.append(w)
        weights = np.asarray(c_w, dtype=np.float64)
        deg = np.zeros(len(comp), dtype=np.float64)
        for (u, v), w in zip(c_edges, weights):
            deg[u] += w
            deg[v] += w
        emb = _embed_component(len(comp), c_edges, weights, deg, dim, modes)
        embeddings.append(decompress(emb, decompress_amount))

    # Scale on a robust percentile, not the maximum. A single far-flung pendant node
    # otherwise sets the world size and squeezes the entire bulk into a dot: on a real
    # 6k-node graph that left 69% of nodes inside the innermost 10% of the radius.
    # Outliers past the percentile are not clipped — they are compressed along their
    # own direction with tanh, so they stay visible and keep their bearing without
    # dictating the scale. Measured on that same graph: 69% -> 3.6% central mass.
    main = embeddings[0]
    main_extent = float(np.percentile(np.abs(main), _SCALE_PERCENTILE)) or 1.0
    unit = main / main_extent
    radii = np.linalg.norm(unit, axis=1, keepdims=True)
    unit = np.where(radii > 1.0, unit / np.maximum(radii, _EPS) * (1.0 + np.tanh(radii - 1.0)), unit)
    main = unit * (world * 0.5)

    pos: Dict[Hashable, Tuple[float, ...]] = {}
    for i, nid in enumerate(comps[0]):
        pos[nid] = tuple(float(x) for x in main[i])

    body = np.abs(main).max(axis=0)
    body_max = float(body.max())
    if body_max <= _EPS:
        # Degenerate main body (a lone node, or a graph with no edges at all): no
        # shape to follow, and dividing through would park every satellite at the
        # origin. An isotropic spread is the honest picture of a structureless graph.
        body = np.ones(dim, dtype=np.float64)
    else:
        body = np.maximum(body / body_max, 0.12)

    # Spiral arm: largest islands hug the body, the long tail of tiny ones trails
    # outward, so the component-size distribution is legible as a shape.
    ga = math.pi * (3.0 - math.sqrt(5.0))
    n_sat = max(len(comps) - 1, 1)
    # The arm hugs the body: reaching far out would shrink the main mass — the part
    # that actually holds the structure — into a speck at the centre of the frame.
    inner = world * 0.55
    reach = world * 0.20
    for k, (comp, emb) in enumerate(zip(comps[1:], embeddings[1:]), start=1):
        t = (k - 1) / n_sat
        radius = inner + reach * math.sqrt(t)
        angle = ga * k
        lift = math.sin(angle * 0.5) * 0.35
        cx = math.cos(angle) * radius * float(body[0])
        cy = lift * radius * float(body[1] if dim > 1 else 1.0)
        cz = math.sin(angle) * radius * float(body[2] if dim > 2 else 1.0)
        extent = float(np.abs(emb).max()) or 1.0
        local = (world * 0.035 * (0.5 + 0.5 * (len(comp) / max(len(comps[0]), 1)) ** 0.3)) / extent
        for i, nid in enumerate(comp):
            p = emb[i] * local
            pos[nid] = (
                float(p[0] + cx),
                float(p[1] + cy) if dim > 1 else cy,
                float(p[2] + cz) if dim > 2 else cz,
            )
    return pos


def component_ids(
    nodes: List[Hashable], links: Sequence[Sequence[Any]]
) -> Dict[Hashable, int]:
    """Map each node to its component rank (0 = giant component, 1+ = islands).

    Lets the viewer offer an "islands" toggle: on a graph with hundreds of tiny
    components the arm can be hidden to inspect the main body on its own.
    """
    adj: Dict[Hashable, List[Hashable]] = {n: [] for n in nodes}
    for link in links:
        u, v = link[0], link[1]
        if u in adj and v in adj and u != v:
            adj[u].append(v)
            adj[v].append(u)
    out: Dict[Hashable, int] = {}
    for rank, comp in enumerate(_components(nodes, adj)):
        for nid in comp:
            out[nid] = rank
    return out


def shape_signature(pos: Dict[Hashable, Tuple[float, ...]]) -> Dict[str, Any]:
    """Report the layout's aspect ratio — the project's numeric fingerprint."""
    if not pos:
        return {"nodes": 0, "aspect": []}
    A = np.array(list(pos.values()), dtype=np.float64)
    s = A.std(axis=0)
    top = float(s.max()) or 1.0
    return {"nodes": int(A.shape[0]), "aspect": [round(float(x / top), 3) for x in s]}

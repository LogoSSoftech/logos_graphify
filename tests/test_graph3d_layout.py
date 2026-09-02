import json
import subprocess
import sys

from graphify.graph3d_layout import (
    component_ids,
    shape_signature,
    spectral_signature_positions,
)


def _chain(n):
    """A path graph: maximally elongated — one long thin thread."""
    nodes = [f"n{i}" for i in range(n)]
    links = [(f"n{i}", f"n{i+1}") for i in range(n - 1)]
    return nodes, links


def _clique(n):
    """A complete graph: maximally isotropic — nothing distinguishes any axis."""
    nodes = [f"c{i}" for i in range(n)]
    links = [(f"c{i}", f"c{j}") for i in range(n) for j in range(i + 1, n)]
    return nodes, links


def test_shape_reflects_structure_not_a_template():
    """The whole point of the signature layout: a thread must not look like a ball.

    A path graph's Fiedler vector dominates, so its embedding is far more elongated
    than a clique's, which has no preferred direction at all. If both came out with
    the same aspect ratio the layout would be a mold, not a fingerprint.
    """
    chain_aspect = shape_signature(spectral_signature_positions(*_chain(40)))["aspect"]
    clique_aspect = shape_signature(spectral_signature_positions(*_clique(20)))["aspect"]

    # aspect is sorted-by-axis, normalized to the widest axis == 1.0
    chain_flatness = min(chain_aspect)
    clique_flatness = min(clique_aspect)
    assert chain_flatness < clique_flatness, (
        f"chain {chain_aspect} should be more elongated than clique {clique_aspect}"
    )
    # the clique is genuinely near-isotropic
    assert clique_flatness > 0.4


def test_layout_is_deterministic():
    """Same code must always yield the same shape, or it isn't a fingerprint."""
    nodes, links = _chain(30)
    a = spectral_signature_positions(nodes, links)
    b = spectral_signature_positions(nodes, links)
    assert a == b


def test_disconnected_components_all_placed():
    """Islands are common in real graphs (domina_isp had 226); none may be dropped."""
    nodes = [f"a{i}" for i in range(10)] + [f"b{i}" for i in range(6)] + ["lonely"]
    links = [(f"a{i}", f"a{i+1}") for i in range(9) if True]
    links += [(f"b{i}", f"b{i+1}") for i in range(5)]
    pos = spectral_signature_positions(nodes, links)
    assert set(pos) == set(nodes)
    assert all(len(v) == 3 for v in pos.values())
    assert all(all(isinstance(c, float) for c in v) for v in pos.values())


def test_self_loops_and_dangling_links_are_ignored():
    nodes, links = _chain(8)
    links = list(links) + [("n0", "n0"), ("n0", "ghost")]
    pos = spectral_signature_positions(nodes, links)
    assert set(pos) == set(nodes)


def test_empty_graph():
    assert spectral_signature_positions([], []) == {}
    assert shape_signature({})["nodes"] == 0


def test_cli_signature_pins_coordinates_and_freezes(tmp_path):
    out = tmp_path / "graphify-out"
    out.mkdir()
    nodes, links = _chain(25)
    graph = {
        "directed": False, "multigraph": False, "graph": {},
        "nodes": [{"id": n, "label": n, "source_file": "src/x.py",
                   "file_type": "code", "community": 0} for n in nodes],
        "links": [{"source": s, "target": t, "relation": "calls",
                   "confidence": "EXTRACTED"} for s, t in links],
        "hyperedges": [],
    }
    (out / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
    dest = tmp_path / "sig.html"

    result = subprocess.run(
        [sys.executable, "-m", "graphify", "graph3d", "--graph", str(out / "graph.json"),
         "--output", str(dest), "--layout", "signature", "--label", "demo"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    html = dest.read_text(encoding="utf-8")
    # a precomputed layout must freeze the engine, else the sim drifts the shape away
    assert "cooldownTicks(0)" in html
    assert "cooldownTime" not in html
    assert '"fx":' in html and '"fz":' in html
    assert "shape" in html  # the fingerprint is reported in the header


def test_cli_rejects_unknown_layout(tmp_path):
    out = tmp_path / "graphify-out"
    out.mkdir()
    (out / "graph.json").write_text(json.dumps(
        {"directed": False, "multigraph": False, "graph": {},
         "nodes": [{"id": "a", "label": "a"}], "links": [], "hyperedges": []}
    ), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "graphify", "graph3d",
         "--graph", str(out / "graph.json"), "--layout", "banana"],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "unknown --layout" in result.stderr


def test_force_layout_stays_unpinned(tmp_path):
    """Default layout must keep the live simulation — no regression for `force`."""
    out = tmp_path / "graphify-out"
    out.mkdir()
    (out / "graph.json").write_text(json.dumps(
        {"directed": False, "multigraph": False, "graph": {},
         "nodes": [{"id": "a", "label": "a"}, {"id": "b", "label": "b"}],
         "links": [{"source": "a", "target": "b", "relation": "calls"}],
         "hyperedges": []}
    ), encoding="utf-8")
    dest = tmp_path / "f.html"
    result = subprocess.run(
        [sys.executable, "-m", "graphify", "graph3d",
         "--graph", str(out / "graph.json"), "--output", str(dest)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    html = dest.read_text(encoding="utf-8")
    assert "cooldownTime(15000)" in html
    assert '"fx":' not in html


def test_edgeless_graph_does_not_collapse_to_a_point():
    """A graph with no edges has no shape — but its nodes must still be distinct.

    Every node is its own component, so the "main body" has zero extent; dividing
    the satellite shell through by that extent parked all of them at the origin
    (all nodes stacked on one invisible point). Real stub apps hit this:
    domina_megasoft and domina_hr_ve are 8 nodes with 0 edges.
    """
    nodes = [f"s{i}" for i in range(8)]
    pos = spectral_signature_positions(nodes, [])
    assert set(pos) == set(nodes)
    assert len(set(pos.values())) == len(nodes), "nodes collapsed onto each other"
    assert any(any(abs(c) > 1.0 for c in p) for p in pos.values())


def test_flat_body_still_spreads_its_islands():
    """A near-flat main body must not crush its satellites onto a line."""
    nodes, links = _chain(30)
    nodes = nodes + [f"iso{i}" for i in range(6)]
    pos = spectral_signature_positions(nodes, links)
    iso = [pos[f"iso{i}"] for i in range(6)]
    assert len(set(iso)) == 6


def test_scale_is_robust_to_a_single_outlier():
    """One far-flung pendant node must not squeeze the bulk into a dot.

    Scaling on the maximum let a single extreme coordinate set the world size, so
    the body collapsed to the centre: 69% of a real 6k-node graph landed inside the
    innermost 10% of the radius. Scaling on a percentile (outliers compressed, not
    clipped) keeps the mass spread out.
    """
    import numpy as np

    # a dense blob plus one long tail — the tail node is the outlier
    nodes = [f"b{i}" for i in range(60)] + ["tail1", "tail2", "tail3"]
    links = [(f"b{i}", f"b{j}") for i in range(30) for j in range(i + 1, 30)]
    links += [(f"b{i}", f"b{i+1}") for i in range(30, 59)]
    links += [("b59", "tail1"), ("tail1", "tail2"), ("tail2", "tail3")]

    pos = spectral_signature_positions(nodes, links)
    P = np.array(list(pos.values()))
    P = P - P.mean(axis=0)
    r = np.linalg.norm(P, axis=1)
    r = r / max(r.max(), 1e-9)
    assert (r < 0.10).mean() < 0.35, "the bulk is still crushed at the centre"


def test_relation_type_changes_the_shape():
    """The mix of relations must bend the geometry, or the weights do nothing."""
    nodes = [f"n{i}" for i in range(24)]
    edges = [(f"n{i}", f"n{i+1}") for i in range(23)]
    as_inherits = spectral_signature_positions(nodes, [(u, v, "inherits") for u, v in edges])
    as_imports = spectral_signature_positions(nodes, [(u, v, "imports") for u, v in edges])
    assert as_inherits != as_imports


def test_unweighted_mode_ignores_relations():
    nodes = [f"n{i}" for i in range(20)]
    edges = [(f"n{i}", f"n{i+1}") for i in range(19)]
    a = spectral_signature_positions(nodes, [(u, v, "inherits") for u, v in edges], weighted=False)
    b = spectral_signature_positions(nodes, [(u, v, "imports") for u, v in edges], weighted=False)
    assert a == b


def test_component_ids_ranks_giant_first():
    nodes = [f"a{i}" for i in range(9)] + [f"b{i}" for i in range(3)] + ["solo"]
    links = [(f"a{i}", f"a{i+1}") for i in range(8)] + [(f"b{i}", f"b{i+1}") for i in range(2)]
    ids = component_ids(nodes, links)
    assert all(ids[f"a{i}"] == 0 for i in range(9))
    assert len({ids[f"b{i}"] for i in range(3)}) == 1
    assert ids["b0"] > 0 and ids["solo"] > 0

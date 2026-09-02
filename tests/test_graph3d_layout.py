import json
import subprocess
import sys

from graphify.graph3d_layout import (
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

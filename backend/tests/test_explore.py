import itertools

import numpy as np
import pandas as pd
import pytest

from causality import data, explore, sim
from causality.schema import Status, to_jsonable


@pytest.fixture(scope="module")
def chain_section():
    df, _ = sim.chain()
    return explore.run(data.prepare(df))


def _drawn(sec):
    return sec.charts[0].data["edges"]


def _pair(sec, a, b):
    return {frozenset((e["from"], e["to"])): e for e in _drawn(sec)}.get(frozenset((a, b)))


def _collider_then_child(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    y = rng.uniform(-2, 2, n)
    z = x + y + rng.uniform(-1, 1, n)
    w = 1.0 * z + rng.uniform(-1, 1, n)
    return pd.DataFrame({"X": x, "Y": y, "Z": z, "W": w})


def test_causallearn_mark_convention():
    """graph[i, j] is the mark at node i on the i-j edge: X -> Z <- Y, Z -> W.
    FCI gives X o-> Z, Y o-> Z and (rule R1) Z --> W."""
    df = _collider_then_child()
    X = ((df - df.mean()) / df.std()).to_numpy()
    G = explore._fci_graph(X)
    X_, Y_, Z_, W_ = 0, 1, 2, 3
    assert explore.pag_mark(G, X_, Z_) == (explore.CIRCLE, explore.ARROW)   # X o-> Z
    assert explore.pag_mark(G, Y_, Z_) == (explore.CIRCLE, explore.ARROW)   # Y o-> Z
    assert explore.pag_mark(G, Z_, W_) == (explore.TAIL, explore.ARROW)     # Z --> W
    assert explore.pag_mark(G, X_, Y_) == (0, 0)
    assert explore.edge_string(*explore.pag_mark(G, Z_, W_)) == "-->"
    assert explore.edge_string(*explore.pag_mark(G, W_, Z_)) == "<--"


def test_edge_type_marks():
    T, A, C = explore.TAIL, explore.ARROW, explore.CIRCLE
    assert explore.edge_type(T, A) == "directed"
    assert explore.edge_type(A, T) == "directed"
    assert explore.edge_type(A, A) == "bidirected"
    assert explore.edge_type(C, A) == "partial"
    assert explore.edge_type(A, C) == "partial"
    assert explore.edge_type(C, C) == "undirected"
    assert explore.edge_type(T, C) == "undirected"
    assert explore.edge_type(0, 0) is None


def test_chain_adjacencies(chain_section):
    s = chain_section
    assert s.error is None and s.status is None and s.key == "explore"
    for a, b in [("A", "B"), ("B", "C"), ("D", "C")]:
        assert _pair(s, a, b) is not None, (a, b)
    assert not any("E" in (e["from"], e["to"]) for e in _drawn(s))
    assert len(_drawn(s)) == 3


def test_chain_collider_oriented_into_C(chain_section):
    for parent in ("B", "D"):
        e = _pair(chain_section, parent, "C")
        assert e["type"] in ("directed", "partial")
        assert e["from"] == parent and e["to"] == "C"
    # nothing points out of C
    assert not any(e["from"] == "C" and e["type"] in ("directed", "partial") for e in _drawn(chain_section))


def test_chain_A_B_not_a_confident_arrow(chain_section):
    e = _pair(chain_section, "A", "B")
    assert e["type"] != "directed"   # FCI cannot orient A-B in this chain


def test_lingam_finds_A_to_B(chain_section):
    le = chain_section.metrics["lingam_edges"]
    assert any(x["from"] == "A" and x["to"] == "B" for x in le)
    g = [a for a in chain_section.assumptions if a.key == "explore_lingam_nongaussian"][0]
    assert g.status == Status.PASS   # uniform errors


def test_assumptions_and_json(chain_section):
    assumed = [a for a in chain_section.assumptions if a.status == Status.ASSUMED]
    assert len(assumed) == 3 and all(not a.testable for a in assumed)
    assert "without arrowheads" in chain_section.summary
    j = to_jsonable(chain_section)
    assert j["charts"][0]["kind"] == "dag"
    for p in chain_section.metrics["pairs"]:
        assert {"a", "b", "fci", "pc", "lingam", "stability", "corr", "partial_corr", "checks", "reading"} <= set(p)


@pytest.mark.parametrize("make", [lambda: sim.chain()[0], _collider_then_child,
                                  lambda: sim.observational()[0]])
def test_uncertain_marks_never_directed(make):
    df = make()
    sec = explore.run(data.prepare(df))
    assert sec.error is None
    for p in sec.metrics["pairs"]:
        if p["drawn"] and p["drawn"]["type"] == "directed":
            # only from a full-data tail-arrow, in the same orientation
            assert p["fci"] is not None and "o" not in p["fci"].split()[1]
            assert p["fci"].split()[1] in ("-->", "<--")
            frm = p["a"] if p["fci"].split()[1] == "-->" else p["b"]
            assert p["drawn"]["from"] == frm
            assert p["stability"]["confidence"] >= explore.ORIENT_STABLE
        if p["fci"] and "o" in p["fci"].split()[1]:
            assert not p["drawn"] or p["drawn"]["type"] != "directed"


def test_justified_arrow_is_drawn():
    sec = explore.run(data.prepare(_collider_then_child()))
    e = _pair(sec, "Z", "W")
    assert e["type"] == "directed" and e["from"] == "Z" and e["to"] == "W"


def test_two_columns_graceful():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=100), "y": rng.normal(size=100)})
    sec = explore.run(data.prepare(df))
    assert sec.error and "at least 3" in sec.error


def test_exact_duplicate_column_dropped():
    df, _ = sim.chain(n=800)
    df["A2"] = 2 * df["A"] + 1
    sec = explore.run(data.prepare(df))
    assert sec.error is None
    assert "A2" not in sec.metrics["variables"]
    assert any(d["column"] == "A2" for d in sec.metrics["dropped_duplicates"])

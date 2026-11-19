import numpy as np
import pytest

from app.vectorstore import IndexMismatchError, VectorStore


def rec(i, doc):
    return {"id": f"c{i}", "text": f"t{i}", "metadata": {"doc_id": doc}}


def unit(*v):
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


def test_add_search_filter_delete(tmp_path):
    s = VectorStore(str(tmp_path), "m")
    s.add([rec(0, "A"), rec(1, "A"), rec(2, "B")],
          np.stack([unit(1, 0, 0), unit(0.9, 0.1, 0), unit(0, 1, 0)]))
    top = s.search(unit(1, 0, 0), 2)
    assert [i for i, _ in top] == [0, 1]
    assert [s.records[i]["id"] for i, _ in s.search(unit(1, 0, 0), 5, doc_ids=["B"])] == ["c2"]
    assert s.delete_doc("A") == 2 and len(s) == 1
    assert s.search(unit(1, 0, 0), 5)[0][0] == 0


def test_persistence_roundtrip(tmp_path):
    s = VectorStore(str(tmp_path), "m")
    s.add([rec(0, "A")], np.stack([unit(1, 1, 0)]))
    s.save()
    s2 = VectorStore(str(tmp_path), "m")
    assert len(s2) == 1 and s2.records[0]["id"] == "c0"
    assert s2.search(unit(1, 1, 0), 1)[0][1] == pytest.approx(1.0, abs=1e-5)


def test_embedding_model_mismatch_is_rejected(tmp_path):
    s = VectorStore(str(tmp_path), "model-a")
    s.add([rec(0, "A")], np.stack([unit(1, 0)]))
    s.save()
    with pytest.raises(IndexMismatchError):
        VectorStore(str(tmp_path), "model-b")


def test_dim_mismatch_and_idempotent_add(tmp_path):
    s = VectorStore(str(tmp_path), "m")
    s.add([rec(0, "A")], np.stack([unit(1, 0)]))
    with pytest.raises(ValueError):
        s.add([rec(1, "A")], np.stack([unit(1, 0, 0)]))
    s.add([rec(0, "A")], np.stack([unit(0, 1)]))  # same id replaces
    assert len(s) == 1


def test_empty_store_search(tmp_path):
    assert VectorStore(str(tmp_path), "m").search(unit(1, 0), 3) == []

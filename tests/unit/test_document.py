"""
Unit tests for NormalizedDocument and SourceType.
Run with: pytest tests/unit/test_document.py -v
"""
import pytest
from cloudsearch_shared.document import NormalizedDocument, SourceType, _stable_hash


class TestSourceType:
    def test_all_values_are_strings(self):
        for st in SourceType:
            assert isinstance(st.value, str)

    def test_from_string(self):
        assert SourceType("INDEXED") == SourceType.INDEXED
        assert SourceType("WEB") == SourceType.WEB

    def test_unknown_fallback(self):
        with pytest.raises(ValueError):
            SourceType("NONEXISTENT")


class TestNormalizedDocument:
    def _make_doc(self, **kwargs) -> NormalizedDocument:
        defaults = dict(
            title="Test Doc",
            url="https://example.com/test",
            content="This is the full content of the document chunk.",
            score=0.8,
            source_type=SourceType.INDEXED,
        )
        defaults.update(kwargs)
        return NormalizedDocument.create(**defaults)

    def test_create_generates_stable_id(self):
        doc1 = self._make_doc()
        doc2 = self._make_doc()
        assert doc1.id == doc2.id  # Same URL + chunk_idx → same ID

    def test_different_chunk_idx_different_id(self):
        doc1 = self._make_doc(chunk_idx=0)
        doc2 = self._make_doc(chunk_idx=1)
        assert doc1.id != doc2.id

    def test_score_clamped_above_one(self):
        doc = self._make_doc(score=1.5)
        assert doc.score == 1.0

    def test_score_clamped_below_zero(self):
        doc = self._make_doc(score=-0.3)
        assert doc.score == 0.0

    def test_score_within_range_preserved(self):
        doc = self._make_doc(score=0.75)
        assert doc.score == pytest.approx(0.75)

    def test_snippet_auto_generated_from_content(self):
        long_content = "x" * 400
        doc = self._make_doc(content=long_content)
        assert len(doc.snippet) <= 300

    def test_doc_id_derived_from_url(self):
        doc = self._make_doc()
        expected = _stable_hash("https://example.com/test")
        assert doc.doc_id == expected

    def test_to_dict_round_trip(self):
        doc = self._make_doc(
            metadata={"author": "Alice"},
            chunk_idx=3,
        )
        data = doc.to_dict()
        restored = NormalizedDocument.from_dict(data)
        assert restored.id == doc.id
        assert restored.title == doc.title
        assert restored.url == doc.url
        assert restored.score == pytest.approx(doc.score)
        assert restored.source_type == doc.source_type
        assert restored.metadata == doc.metadata
        assert restored.chunk_idx == doc.chunk_idx

    def test_from_dict_with_missing_optional_fields(self):
        minimal = {
            "id": "abc123",
            "title": "Minimal",
            "url": "https://example.com",
            "snippet": "",
            "content": "",
            "score": 0.5,
            "source_type": "WEB",
        }
        doc = NormalizedDocument.from_dict(minimal)
        assert doc.chunk_idx == 0
        assert doc.metadata == {}

    def test_source_type_preserved_through_serialization(self):
        for st in SourceType:
            doc = self._make_doc(source_type=st)
            data = doc.to_dict()
            restored = NormalizedDocument.from_dict(data)
            assert restored.source_type == st

    def test_repr_is_not_empty(self):
        # SearchProvider has __repr__ — document has no __repr__ override
        # but dataclass gives one; just assert it doesn't crash
        doc = self._make_doc()
        assert "NormalizedDocument" in repr(doc)


class TestStableHash:
    def test_deterministic(self):
        assert _stable_hash("hello") == _stable_hash("hello")

    def test_different_inputs_different_hashes(self):
        assert _stable_hash("a") != _stable_hash("b")

    def test_output_is_16_hex_chars(self):
        h = _stable_hash("test")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

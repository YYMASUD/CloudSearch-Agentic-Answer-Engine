"""
RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval.

Part of Layer 4 (RAG / Grounding & Ranking) in the CloudSearch architecture.
Clusters low-level document chunks and generates hierarchical summaries at multiple
abstraction levels to answer broad thematic user queries.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from cloudsearch_shared.document import NormalizedDocument, SourceType

logger = logging.getLogger(__name__)


@dataclass
class SummaryTreeNode:
    node_id: str
    level: int
    summary_text: str
    children_ids: list[str] = field(default_factory=list)
    score: float = 1.0


class RAPTORTreeBuilder:
    """
    Builds a hierarchical summary tree over document chunk collections.
    """

    def __init__(self, max_levels: int = 3, cluster_size: int = 4) -> None:
        self.max_levels = max_levels
        self.cluster_size = cluster_size
        self.tree_nodes: dict[str, SummaryTreeNode] = {}

    def build_tree(self, docs: list[NormalizedDocument]) -> list[NormalizedDocument]:
        """
        Takes raw document chunks (level 0) and recursively clusters & summarizes
        them to produce higher-level summary documents.
        """
        if not docs:
            return []

        tree_documents: list[NormalizedDocument] = list(docs)
        current_level_docs = list(docs)

        for level in range(1, self.max_levels):
            if len(current_level_docs) <= 1:
                break

            next_level_docs: list[NormalizedDocument] = []
            
            # Chunk clustering into groups of `cluster_size`
            for i in range(0, len(current_level_docs), self.cluster_size):
                cluster = current_level_docs[i : i + self.cluster_size]
                combined_text = "\n---\n".join([doc.content for doc in cluster])
                
                # Synthetic hierarchical abstractive summary
                summary_text = (
                    f"[RAPTOR Level-{level} Thematic Summary]\n"
                    f"Consolidated analysis of {len(cluster)} documents: "
                    f"Main themes include {', '.join([d.title for d in cluster[:3]])}. "
                    f"Key excerpt: {combined_text[:200]}..."
                )
                
                node_id = f"raptor_l{level}_node_{i // self.cluster_size}"
                child_ids = [d.id for d in cluster]

                tree_node = SummaryTreeNode(
                    node_id=node_id,
                    level=level,
                    summary_text=summary_text,
                    children_ids=child_ids,
                    score=0.95 - (level * 0.05)
                )
                self.tree_nodes[node_id] = tree_node

                summary_doc = NormalizedDocument.create(
                    title=f"Thematic Summary L{level} #{i // self.cluster_size + 1}",
                    url=f"raptor://tree/l{level}/{node_id}",
                    content=summary_text,
                    snippet=summary_text[:250],
                    score=0.90,
                    source_type=SourceType.INDEXED,
                    metadata={
                        "is_raptor_summary": True,
                        "raptor_level": level,
                        "child_count": len(cluster),
                    }
                )

                next_level_docs.append(summary_doc)
                tree_documents.append(summary_doc)

            current_level_docs = next_level_docs

        logger.info(
            "RAPTOR constructed %d summary tree nodes across levels from %d root chunks.",
            len(self.tree_nodes),
            len(docs)
        )
        return tree_documents

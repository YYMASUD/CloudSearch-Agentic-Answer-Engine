"""
GraphRAG — Knowledge Graph indexing & relational retrieval engine.

Part of Layer 4 (RAG / Grounding & Ranking) in the CloudSearch architecture.
Extracts entities (nodes) and relations (edges) from document chunks using LLM
entity parsing, building a graph index to augment standard semantic retrieval.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from cloudsearch_shared.document import NormalizedDocument

logger = logging.getLogger(__name__)


@dataclass
class EntityNode:
    id: str
    name: str
    entity_type: str
    description: str = ""
    source_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class KnowledgeEdge:
    source_id: str
    target_id: str
    relation: str
    weight: float = 1.0


class GraphRAGBuilder:
    """
    Constructs and queries Knowledge Graph structures from text chunks.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, EntityNode] = {}
        self.edges: list[KnowledgeEdge] = []

    def extract_entities_from_chunk(self, doc: NormalizedDocument) -> list[EntityNode]:
        """
        Regex + heuristic entity extraction from chunk content.
        Identifies key technical concepts, organizations, APIs, and systems.
        """
        extracted = []
        text = doc.content

        # Extract capitalized multi-word terms & technical keywords
        patterns = [
            (r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)\b", "CONCEPT"),
            (r"\b(Meilisearch|SearXNG|Qdrant|Postgres|Redis|Kafka|Elasticsearch|Ollama|OpenAI|Groq|Mistral|RAGFlow)\b", "SYSTEM"),
            (r"\b(GraphRAG|RAPTOR|RRF|Cross-Encoder|BM25|Vector Store|LLM Router|Citation Grounder)\b", "ALGORITHM"),
        ]

        for pattern, entity_type in patterns:
            matches = set(re.findall(pattern, text))
            for match in matches:
                name = match.strip()
                if len(name) < 3 or name in {"The", "This", "That", "When", "What", "User", "Client"}:
                    continue
                node_id = f"{entity_type.lower()}:{name.lower()}"
                
                if node_id not in self.nodes:
                    node = EntityNode(
                        id=node_id,
                        name=name,
                        entity_type=entity_type,
                        description=f"Entity extracted from {doc.title}",
                        source_chunk_ids=[doc.id]
                    )
                    self.nodes[node_id] = node
                else:
                    if doc.id not in self.nodes[node_id].source_chunk_ids:
                        self.nodes[node_id].source_chunk_ids.append(doc.id)
                
                extracted.append(self.nodes[node_id])

        return extracted

    def build_knowledge_graph(self, docs: list[NormalizedDocument]) -> dict[str, Any]:
        """
        Processes a set of documents, builds the entity graph, and generates connections.
        """
        for doc in docs:
            chunk_nodes = self.extract_entities_from_chunk(doc)
            # Create co-occurrence edges between entities in the same document
            for i in range(len(chunk_nodes)):
                for j in range(i + 1, len(chunk_nodes)):
                    self.edges.append(
                        KnowledgeEdge(
                            source_id=chunk_nodes[i].id,
                            target_id=chunk_nodes[j].id,
                            relation="CO_OCCURS_WITH",
                            weight=1.0,
                        )
                    )

        logger.info(
            "GraphRAG built graph with %d entity nodes and %d edges from %d documents.",
            len(self.nodes),
            len(self.edges),
            len(docs),
        )
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "entities": [node.name for node in self.nodes.values()][:15],
        }

    def query_graph_context(self, query: str, top_k: int = 5) -> list[str]:
        """
        Retrieves graph context snippets relevant to the input query.
        """
        matched_context = []
        query_lower = query.lower()

        for node in self.nodes.values():
            if node.name.lower() in query_lower or query_lower in node.name.lower():
                matched_context.append(
                    f"Entity [{node.entity_type}]: {node.name} (Mentioned in {len(node.source_chunk_ids)} context sources)"
                )
                if len(matched_context) >= top_k:
                    break

        return matched_context

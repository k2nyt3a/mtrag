"""RAG conversation-collection pipeline over MTRAG human-generated conversations.

This package is a thin integration layer. It does NOT reimplement Self-RAG,
HippoRAG or RAPTOR: the connectors under ``RAG/<system>/rag.py`` delegate to the
existing implementations in the sibling research repo ``jaist/mtrag_three_rag``.
Only the Vector adapter is fully self-contained (dense retrieval over the official
MTRAG corpus using precomputed embeddings + the MTRAG-baseline generator).

Nothing here touches question-generation research code (Xie / Proposed / static /
dynamic / failure). We ONLY collect raw RAG conversation data.
"""

import json
import os
from pathlib import Path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import QueryBundle, MetadataMode, TextNode
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.vector_stores.milvus import MilvusVectorStore
from typing import Any, Dict, List
from dataclasses import dataclass
import re

from pymilvus import MilvusException

from server import getLogger
from server.db import (
    MILVUS_COLLECTION_NAME,
    MILVUS_EMBEDDING_FIELD,
    MILVUS_PASSWORD,
    MILVUS_PK_FIELD,
    MILVUS_TEXT_FIELD,
    MILVUS_URI,
    MILVUS_USERNAME,
    MILVUS_VECTOR_DIM,
)
from server.llm.vllm import LlmVllm

from .reranker import BGERerankPostprocessor

logger = getLogger(__name__)


@dataclass
class ChunkingResultData:
    score: float
    text: str
    metadata: Dict[str, Any]
    node_id: str
    file_name: str
    page_number: int


class ChunkingProcess:

    def __init__(
        self, model_id="BAAI/bge-m3", model_dir=None, reranker_model_path=None
    ):

        script_dir = Path(__file__).parent.resolve()
        default_persist_dir = script_dir / "storage" / "your-agent"
        self.vector_persist_dir = os.getenv("VECTOR_PERSIST_DIR", default_persist_dir)

        os.makedirs(self.vector_persist_dir, exist_ok=True)

        # When implementing Graph Rag
        # self.vllm = LlmVllm()

        self.embed_model = HuggingFaceEmbedding(
            model_name=model_id,
            cache_folder=model_dir,
            device="cuda",  # GPU 사용 권장
        )

        self.base_splitter = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=100,
            paragraph_separator="\n\n",
            secondary_chunking_regex=r"[。.!?\n]",  # 한국어/영어 문장 경계
        )

        self.semantic_splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,  # 95% 이상 차이 시 분할
            embed_model=self.embed_model,
        )

        self.reranker_processor = BGERerankPostprocessor(
            reranker_model_path=reranker_model_path, top_k=10
        )
        Settings.embed_model = self.embed_model

        self.vector_store = self.get_vector_store()
        self.vector_store._collection.load()

        # self.index = VectorStoreIndex(
        #     nodes=[],
        #     vector_store=self.vector_store,
        #     storage_context=None,
        #     embed_model=self.embed_model,
        #     store_nodes_override=True,
        #     upsert=True,
        #     upsert_kwargs={"ignore_duplicate": True},
        # )

        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store, embed_model=self.embed_model
        )

        logger.info(
            f"Existing index loading complete (number of nodes: {self.vector_store._collection.num_entities})"
        )
        # if os.path.exists(os.path.join(self.vector_persist_dir, "docstore.json")):
        # storage_context = StorageContext.from_defaults(
        #     vector_store=vector_store, persist_dir=self.vector_persist_dir
        # )

        # self.index = load_index_from_storage(
        #     storage_context,
        #     embed_model=self.embed_model,
        #     store_nodes_override=True,
        #     upsert=True,
        #     upsert_kwargs={"ignore_duplicate": True},
        # )
        # else:
        #     storage_context = StorageContext.from_defaults(vector_store=vector_store)
        #     # documents = self.load_documents(doc_dir)
        #     self.index = VectorStoreIndex(
        #         nodes=[],
        #         # storage_context=storage_context,
        #         embed_model=self.embed_model,
        #         store_nodes_override=True,
        #         upsert=True,
        #         upsert_kwargs={"ignore_duplicate": True},
        #     )
        # self.index.storage_context.persist(persist_dir=self.vector_persist_dir)

    def get_vector_store(self, collection=MILVUS_COLLECTION_NAME) -> MilvusVectorStore:
        return MilvusVectorStore(
            uri=MILVUS_URI,
            token=f"{MILVUS_USERNAME}:{MILVUS_PASSWORD}",
            collection_name=collection,
            dim=MILVUS_VECTOR_DIM,
            pk_field=MILVUS_PK_FIELD,
            embedding_field=MILVUS_EMBEDDING_FIELD,
            text_field=MILVUS_TEXT_FIELD,
            enable_upsert=True,
            overwrite=False,
        )

    def filter_meaningless_nodes(
        self, nodes, min_length: int = 30, remove_patterns: list = None
    ):
        if remove_patterns is None:
            remove_patterns = [
                r"^[\.\s\*\-\•]*$",  # ".", "...", "•"
                r"^[0-9\.\-\s\*\•]+$",  # "1.", "1.1"
                r"^Figure\s*\d+[\.\s]*$",  # "Figure 1"
                r"^Table\s*\d+[\.\s]*$",  # "Table 1"
                r"^[\(\[]?\d+[\)\]]?[\.\s]*$",  # "(1)", "[1]"
            ]

        filtered = []
        for node in nodes:
            text = node.text.strip()

            if len(text) < min_length:
                continue

            if any(
                re.match(pattern, text, re.IGNORECASE) for pattern in remove_patterns
            ):
                continue

            cleaned_text = re.sub(r"\s+", " ", text)
            if len(cleaned_text) < min_length:
                continue

            clean_node = TextNode(
                text=cleaned_text,
                node_id=node.node_id,
                metadata=node.metadata.copy(),
                embedding=node.embedding,  # 기존 임베딩 유지
            )
            filtered.append(clean_node)

        removed = len(nodes) - len(filtered)
        logger.info(
            f"Node filtering: {len(nodes)} → {len(filtered)} (removed: {removed})"
        )
        return filtered

    def safe_query_existing_ids(self, semantic_nodes, batch_size: int = 500):
        col = self.vector_store._collection
        col.load()

        existing_ids = set()
        for i in range(0, len(semantic_nodes), batch_size):
            batch = semantic_nodes[i : i + batch_size]
            ids = [n.node_id for n in batch]
            if not ids:
                continue

            ids_str = json.dumps(ids)

            try:
                results = col.query(
                    expr=f"id in {ids_str}",
                    output_fields=["id"],
                    limit=len(ids),  # 최대 반환 제한
                )
                existing_ids.update(r["id"] for r in results)
                logger.info(
                    f"Batch {i//batch_size + 1}: {len(results)} existing IDs found"
                )
            except MilvusException as e:
                if "limit exceeded" in str(e):
                    logger.info(
                        f"Batch size exceeded: Reduce {len(ids)} → {batch_size//2}"
                    )
                    return self.safe_query_existing_ids(semantic_nodes, batch_size // 2)
                else:
                    raise e

        return existing_ids

    def process_chucking(self, doc_dir):
        documents = SimpleDirectoryReader(
            doc_dir, file_metadata=lambda x: {"file_name": os.path.basename(x)}
        ).load_data()

        # print(f"documents: {documents}")

        # doc_text = ""
        # for doc in documents:
        #     doc_text = doc_text + doc.text_resource.text

        base_nodes = self.base_splitter.get_nodes_from_documents(documents)
        logger.info(f"SentenceSplitter length → {len(base_nodes)}")
        semantic_nodes = self.semantic_splitter.get_nodes_from_documents(base_nodes)
        logger.info(f"SemanticSplitter length → {len(semantic_nodes)}")
        semantic_nodes = self.filter_meaningless_nodes(semantic_nodes)

        # logger.info(f"semantic_nodes[0] → {semantic_nodes[0]}")
        # logger.info(f"semantic_nodes[0].text → {semantic_nodes[0].text}")

        for i, node in enumerate(semantic_nodes):
            if not hasattr(node, "embedding") or node.embedding is None:
                node.embedding = self.embed_model.get_text_embedding(node.text)
            node.metadata.setdefault("file_name", "unknown_file")
            node.node_id = f"{node.metadata["file_name"]}_chunk_{i}"
            # When implementing Graph Rag
            # logger.info(f"---------------")
            # logger.info(f"node.text → {node.text}")
            # extracted_extity = self.vllm.extract_entities(node.text)
            # logger.info(f"extracted_extity: {extracted_extity}")

        if semantic_nodes:
            existing_ids = self.safe_query_existing_ids(semantic_nodes)
            new_nodes = [n for n in semantic_nodes if n.node_id not in existing_ids]
            if new_nodes:
                self.vector_store.add(new_nodes)
                self.vector_store._collection.flush()
                logger.info(f"New insertions: {len(new_nodes)}")
            # existing_ids = set(self.index.docstore.docs.keys())
            # new_nodes = [n for n in semantic_nodes if n.node_id not in existing_ids]

            # if new_nodes:
            #     self.index.insert_nodes(new_nodes)
            #     # extracted_extity = self.vllm.extract_entities(new_nodes)
            #     # logger.info(f"extracted_extity: {extracted_extity}")
            #     self.index.storage_context.persist(persist_dir=self.vector_persist_dir)
            #     logger.info(f"Processing: New {len(new_nodes)}")

        logger.info(
            f"Existing index loading complete (number of nodes: {self.vector_store._collection.num_entities})"
        )
        # index = VectorStoreIndex(semantic_nodes)
        logger.info(f"Complete generate index")

    def build_query_filters(self, filters: dict) -> MetadataFilters:
        llama_filters = []
        for key, value in filters.items():
            llama_filters.append(ExactMatchFilter(key=key, value=value))
        return MetadataFilters(filters=llama_filters)

    def query_retirival(
        self,
        query_str: str,
        top_k: int = 50,
        filters: dict = None,
        min_similarity: float = 0.5,
    ) -> List[ChunkingResultData]:

        retriever = self.index.as_retriever(
            similarity_top_k=top_k,
            filters=self.build_query_filters(filters) if filters else None,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=min_similarity)
            ],
        )

        print(
            f"Retriever settings: top_k={top_k}, filter={filters}, cutoff={min_similarity}"
        )

        # nodes = retriever.retrieve(query_str)
        query_bundle = QueryBundle(query_str)
        nodes_with_scores = retriever.retrieve(query_bundle)
        reranked_nodes = self.reranker_processor.postprocess_nodes(
            nodes_with_scores, query_bundle
        )

        results = []
        # When implementing Graph Rag
        # entity_list = []
        # relation_list = []
        for node_with_score in reranked_nodes:
            node = node_with_score.node

            result = ChunkingResultData(
                round(node_with_score.score, 4),
                node.get_content(metadata_mode=MetadataMode.ALL),
                node.metadata,
                node.node_id,
                node.metadata.get("file_name", "unknown"),
                node.metadata.get("page_number", -1),
            )
            results.append(result)

            # When implementing Graph Rag
            # extracted = vllm.extract_entities(result["text"])

            # for ent in extracted.get("entities", []):
            #     ent_doc = Document(
            #         text=ent["description"],
            #         metadata={
            #             "name": ent["name"],
            #             "type": ent["type"],
            #             "source_node_id": node.node_id,
            #         },
            #     )
            #     entity_list.append(ent_doc)

            # for rel in extracted.get("relations", []):
            #     rel_doc = Document(
            #         text=rel["description"],
            #         metadata={
            #             "source": rel["source"],
            #             "target": rel["target"],
            #             "type": rel["type"],
            #             "source_node_id": node.node_id,
            #         },
            #     )
            #     relation_list.append(rel_doc)

        print(f"Search results: {len(nodes_with_scores)} nodes")

        return results


if __name__ == "__main__":

    llm = LlmVllm()
    script_dir = Path(__file__).parent.resolve()
    embed_model_path = script_dir.parent.parent / "models" / "bge-m3"
    # bge-reranker-v2.5-gemma2-lightweight , bge-reranker-v2-m3
    reranker_model_path = script_dir.parent.parent / "models" / "bge-reranker-v2-m3"
    doc_dir = script_dir / "data"

    logger.info(f"doc_dir: {doc_dir}")

    embedding_model_name = "BAAI/bge-m3"

    processor = ChunkingProcess(
        model_id=embedding_model_name,
        model_dir=embed_model_path,
        reranker_model_path=reranker_model_path,
    )

    # processor.process_chucking(doc_dir=doc_dir)

    query_str = "주주환원 촉진세제 이유는?"
    retirival_data = processor.query_retirival(query_str=query_str)
    logger.info(f"reranked: {len(retirival_data)}")

    context_str = ""
    for r in retirival_data:  # 상위 2개 출력
        logger.info(f"Score: {r.score} | File: {r.file_name} | Page: {r.page_number}")
        # print(f"Text: {r['text'][:200]}...\n")
        logger.info(f"Text: {r.text}\n")
        context_str = context_str + r.text

    response = llm.query_rag(context_str=context_str, query_str=query_str)

    logger.info(response[0].outputs[0].text)

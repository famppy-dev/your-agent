import json
import os
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import (
    BaseNode,
    Document,
    ImageNode,
    MetadataMode,
    QueryBundle,
    TextNode,
)
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader
from llama_index.vector_stores.milvus import MilvusVectorStore
from pymilvus import MilvusException
from transformers import AutoModel, AutoProcessor

from server import (
    MILVUS_COLLECTION_NAME,
    MILVUS_EMBEDDING_FIELD,
    MILVUS_PASSWORD,
    MILVUS_PK_FIELD,
    MILVUS_TEXT_FIELD,
    MILVUS_URI,
    MILVUS_USERNAME,
    MILVUS_VECTOR_DIM,
    getLogger,
)
from server.llm.param_util import image_to_base64_data_uri
from server.llm.prompts.img_caption import IMG_CAPTION_PROMPT
from server.llm.vllm import getLlm
from server.models.enums import AppErrorCode
from server.models.response import ErrorDetail

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
        self,
        model_id="BAAI/bge-m3",
        model_img_id=None,
        reranker_model_path=None,
    ):
        self.device = "cuda"
        self.embed_model = self._init_embed_model(model_id)
        logger.info(f"model_img_id: {model_img_id}")
        if model_img_id is not None:
            self.embed_img_model, self.embed_img_model_processor = (
                self._init_embed_img_model(model_img_id)
            )

        self.pdf_reader = PyMuPDFReader()

        self.base_splitter = SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=200,
            paragraph_separator=r"\n\s*\n",
            secondary_chunking_regex=r"(?<=[。.!?\n])\s+",
        )

        self.semantic_splitter = SemanticSplitterNodeParser(
            buffer_size=3,
            breakpoint_percentile_threshold=85,
            embed_model=self.embed_model,
            include_metadata=True,
        )

        if reranker_model_path is None:
            logger.info("Not found reranker model")
        else:
            self.reranker_processor = self._init_reranker_processor(reranker_model_path)
        Settings.embed_model = self.embed_model

        self.vector_store = self._get_vector_store()
        self.vector_store._collection.load()

        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store, embed_model=self.embed_model
        )

        logger.info(
            f"Existing index loading complete (number of nodes: {self.vector_store._collection.num_entities})"
        )

    def _init_embed_model(self, model_id: str):
        return HuggingFaceEmbedding(
            model_name=model_id,
            device=self.device,
        )

    def _init_embed_img_model(self, model_id: str):
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True, device_map="auto"
        )
        model.eval()
        logger.info(f"Load complete SigLIP2 : {model_id}")
        return model, processor

    def _init_reranker_processor(self, reranker_model_path: str):
        return BGERerankPostprocessor(reranker_model_path=reranker_model_path, top_k=10)

    def _get_vector_store(self, collection=MILVUS_COLLECTION_NAME) -> MilvusVectorStore:
        return MilvusVectorStore(
            uri=MILVUS_URI,
            token=f"{MILVUS_USERNAME}:{MILVUS_PASSWORD}",
            collection_name=collection,
            dim=MILVUS_VECTOR_DIM,
            pk_field=MILVUS_PK_FIELD,
            embedding_field=MILVUS_EMBEDDING_FIELD,
            text_field=MILVUS_TEXT_FIELD,
            additional_vector_fields={"img_vector": "img_vector"},
            enable_upsert=True,
            overwrite=False,
        )

    def _filter_meaningless_nodes(
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
                embedding=node.embedding,
            )
            filtered.append(clean_node)

        removed = len(nodes) - len(filtered)
        logger.info(
            f"Node filtering: {len(nodes)} → {len(filtered)} (removed: {removed})"
        )
        return filtered

    def _safe_query_existing_ids(self, semantic_nodes, batch_size: int = 500):
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
                    limit=len(ids),
                )
                existing_ids.update(r["id"] for r in results)
                logger.info(
                    f"Batch {i // batch_size + 1}: {len(results)} existing IDs found"
                )
            except MilvusException as e:
                if "limit exceeded" in str(e):
                    logger.info(
                        f"Batch size exceeded: Reduce {len(ids)} → {batch_size // 2}"
                    )
                    return self._safe_query_existing_ids(
                        semantic_nodes, batch_size // 2
                    )
                else:
                    raise e

        return existing_ids

    def _build_query_filters(self, filters: dict) -> MetadataFilters:
        llama_filters = []
        for key, value in filters.items():
            llama_filters.append(ExactMatchFilter(key=key, value=value))
        return MetadataFilters(filters=llama_filters)

    async def _process_img_node(self, nodes: List[BaseNode]):
        llm = await getLlm()
        for i, node in enumerate(nodes):
            if (
                isinstance(node, ImageNode)
                and self.embed_img_model_processor is not None
            ):
                inputs = self.embed_img_model_processor(
                    images=[node.image_path], return_tensors="pt", padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                input_embedding = (
                    self.embed_img_model.get_image_features(**inputs).cpu().tolist()
                )
                node.metadata = {**node.metadata, "img_vector": input_embedding[0]}

                if llm is not None:
                    # When using vision llm separately
                    # node.text = await llm.describe_image(node.image_path)
                    img = image_to_base64_data_uri(node.image_path)
                    messages = [
                        {
                            "role": "system",
                            "content": [
                                {"type": "text", "text": IMG_CAPTION_PROMPT},
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "url": img,
                                },
                            ],
                        },
                    ]
                    response = await llm.query(
                        messages, temperature=0.2, top_p=0.95, top_k=50
                    )
                    node.text = response.outputs[0].text.strip()
            else:
                node.metadata = {**node.metadata, "img_vector": [0.0] * 768}
        return nodes

    async def _split_by_node(self, documents=List[Document]) -> List[BaseNode]:
        try:
            base_nodes = self.base_splitter.get_nodes_from_documents(documents)
            logger.info(f"SentenceSplitter length → {len(base_nodes)}")

            base_nodes = await self._process_img_node(base_nodes)
            logger.info(f"SentenceSplitter after img base_nodes → {base_nodes}")

            semantic_nodes = self.semantic_splitter.get_nodes_from_documents(base_nodes)
            logger.info(f"SemanticSplitter length → {len(semantic_nodes)}")

            semantic_nodes = self._filter_meaningless_nodes(semantic_nodes)
            logger.info(f"meaningless_nodes length → {len(semantic_nodes)}")

            for i, node in enumerate(semantic_nodes):
                if not hasattr(node, "embedding") or node.embedding is None:
                    node.embedding = self.embed_model.get_text_embedding(node.text)
                node.metadata.setdefault("file_name", "unknown_file")
                node.node_id = f"{node.metadata['file_name']}_chunk_{i}"

            return semantic_nodes
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"_split_by_node error: {tb}")
            raise ErrorDetail(
                error_code=AppErrorCode.INTERNAL, message=repr(e), details=None
            )

    def get_text_embedding(self, text: str) -> List[float]:
        return self.embed_model.get_text_embedding(text)

    async def process_chucking(
        self, doc_dir: Path | str | None = None, input_files: list | None = None
    ):
        try:
            documents = SimpleDirectoryReader(
                input_dir=doc_dir,
                input_files=input_files,
                file_metadata=lambda x: {"file_name": os.path.basename(x)},
                file_extractor={".pdf": self.pdf_reader},
            ).load_data()

            # Remove Zero Width Non-Joiner when using PyMuPDFReader
            for doc in documents:
                if doc.text_resource is not None:
                    doc.text_resource.text = doc.text_resource.text.replace(
                        "\u200b", ""
                    )

            logger.info(f"documents: {documents}")

            nodes = await self._split_by_node(documents)

            if nodes:
                existing_ids = self._safe_query_existing_ids(nodes)
                new_nodes = [n for n in nodes if n.node_id not in existing_ids]
                if new_nodes:
                    self.vector_store.add(new_nodes)
                    self.vector_store._collection.flush()
                    logger.info(f"New insertions: {len(new_nodes)}")

            logger.info(
                f"Existing index loading complete (number of nodes: {self.vector_store._collection.num_entities})"
            )
            logger.info("Complete generate index")
        except Exception as e:
            logger.error(f"process_chucking error: {e}")
            raise ErrorDetail(
                error_code=AppErrorCode.INTERNAL, message=repr(e), details=None
            )

    async def query_retrieval(
        self,
        query_str: str,
        top_k: int = 50,
        filters: dict = None,
        min_similarity: float = 0.5,
    ) -> List[ChunkingResultData]:
        try:
            retriever = self.index.as_retriever(
                similarity_top_k=top_k,
                filters=self._build_query_filters(filters) if filters else None,
                node_postprocessors=[
                    SimilarityPostprocessor(similarity_cutoff=min_similarity)
                ],
            )

            logger.info(
                f"Retriever settings: top_k={top_k}, filter={filters}, cutoff={min_similarity}"
            )

            query_bundle = QueryBundle(query_str)
            nodes_with_scores = retriever.retrieve(query_bundle)
            if self.reranker_processor is not None:
                reranked_nodes = self.reranker_processor.postprocess_nodes(
                    nodes_with_scores, query_bundle
                )

            results = []
            for node_with_score in (
                reranked_nodes if reranked_nodes is not None else nodes_with_scores
            ):
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

            # logger.info(f"Search results: {len(nodes_with_scores)} nodes")

            return results
        except Exception as e:
            logger.error(f"process_chucking error: {e}")
            raise ErrorDetail(
                error_code=AppErrorCode.INTERNAL, message=repr(e), details=None
            )

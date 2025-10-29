from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import QueryBundle

from .reranker import BGERerankPostprocessor


class ChunkingProcess:

    def __init__(
        self, model_id="BAAI/bge-m3", model_dir=None, reranker_model_path=None
    ):
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
            reranker_model_path=reranker_model_path
        )
        Settings.embed_model = self.embed_model

    def process_chucking(self, doc_dir):
        documents = SimpleDirectoryReader(doc_dir).load_data()

        base_nodes = self.base_splitter.get_nodes_from_documents(documents)
        print(f"SentenceSplitter length → {len(base_nodes)}")
        semantic_nodes = self.semantic_splitter.get_nodes_from_documents(base_nodes)
        print(f"SemanticSplitter length → {len(semantic_nodes)}")

        index = VectorStoreIndex(semantic_nodes)
        print(f"Complete generate index")

        retriever = index.as_retriever(similarity_top_k=50)  # 넉넉히 가져와서 rerank

        return index, retriever

    def query_retirival(self, retriever, query_str: str):
        nodes = retriever.retrieve(query_str)
        query_bundle = QueryBundle(query_str=query_str)
        reranked_nodes = self.reranker_processor.postprocess_nodes(nodes, query_bundle)

        print(f"\n[query]: {query_str}")
        print(f"[result] Top-{len(reranked_nodes)}:")
        for i, node in enumerate(reranked_nodes, 1):
            print(f"\n--- #{i} (Score: {node.score:.4f}) ---")
            print(node.get_content()[:500] + "...")
        return reranked_nodes


if __name__ == "__main__":

    script_dir = Path(__file__).parent.resolve()
    embed_model_path = script_dir.parent.parent / "models" / "bge-m3"
    # bge-reranker-v2.5-gemma2-lightweight , bge-reranker-v2-m3
    reranker_model_path = script_dir.parent.parent / "models" / "bge-reranker-v2-m3"
    doc_dir = script_dir / "data"

    embedding_model_name = "BAAI/bge-m3"

    processor = ChunkingProcess(
        model_id=embedding_model_name,
        model_dir=embed_model_path,
        reranker_model_path=reranker_model_path,
    )

    index, retriever = processor.process_chucking(doc_dir=doc_dir)

    reranked = processor.query_retirival(retriever=retriever, query_str="전공")
    # print(reranked)

from pathlib import Path

from server import (
    getLogger,
)
from server.llm.vllm import getLlm

from .chunking_base import ChunkingProcess

logger = getLogger(__name__)


class BasicChunkingProcess(ChunkingProcess):

    def __init__(
        self, model_id="BAAI/bge-m3", model_dir=None, reranker_model_path=None
    ):

        super.__init__(
            self,
            model_id=model_id,
            model_dir=model_dir,
            reranker_model_path=reranker_model_path,
        )


if __name__ == "__main__":

    llm = getLlm()
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

    query_str = "주주환원 촉진세제란 무엇인가?"
    retirival_data = processor.query_retirival(query_str=query_str)
    logger.info(f"reranked: {len(retirival_data)}")

    context_str = ""
    for r in retirival_data:
        logger.info(f"Score: {r.score} | File: {r.file_name} | Page: {r.page_number}")
        logger.info(f"Text: {r.text}\n")
        context_str = context_str + r.text

    response = llm.query_rag(context_str=context_str, query_str=query_str)

    logger.info(response[0].outputs[0].text)

import asyncio
from pathlib import Path

from server import (
    EMBED_IMG_MODEL,
    EMBED_MODEL,
    EMBED_RERANK_MODEL,
    getLogger,
)

from .chunking_base import ChunkingProcess

logger = getLogger(__name__)


class BasicChunkingProcess(ChunkingProcess):

    def __init__(
        self, model_id="BAAI/bge-m3", model_img_id=None, reranker_model_path=None
    ):

        super().__init__(
            model_id=model_id,
            model_img_id=model_img_id,
            reranker_model_path=reranker_model_path,
        )


chunking_process: BasicChunkingProcess | None = None
async_lock = asyncio.Lock()
initialized = False


async def get_chunking_process() -> BasicChunkingProcess:
    global chunking_process, initialized
    if initialized:
        return chunking_process

    async with async_lock:
        if initialized:
            return chunking_process

    chunking_process = BasicChunkingProcess(
        model_id=EMBED_MODEL,
        model_img_id=EMBED_IMG_MODEL,
        reranker_model_path=EMBED_RERANK_MODEL,
    )

    initialized = True
    return chunking_process


async def call_main():
    # llm = await getLlm()
    script_dir = Path(__file__).parent.resolve()
    doc_dir = script_dir / "data"

    logger.info(f"doc_dir: {doc_dir}")

    processor = await get_chunking_process()

    await processor.process_chucking(doc_dir)

    # query_str = "주주환원 촉진세제란 무엇인가?"
    # retirival_data = await processor.query_retirival(query_str=query_str)
    # logger.info(f"reranked: {len(retirival_data)}")

    # context_str = ""
    # for r in retirival_data:
    #     logger.info(f"Score: {r.score} | File: {r.file_name} | Page: {r.page_number}")
    #     logger.info(f"Text: {r.text}\n")
    #     context_str = context_str + r.text

    # response = await llm.query_rag(context_str=context_str, query_str=query_str)

    # logger.info(f"response: {response}")

    # logger.info(response.outputs[0].text)


if __name__ == "__main__":
    asyncio.run(call_main())

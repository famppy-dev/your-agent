import asyncio
from pathlib import Path

from server import EMBED_IMG_MODEL, EMBED_MODEL, EMBED_RERANK_MODEL, getLogger

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


if __name__ == "__main__":
    asyncio.run(call_main())

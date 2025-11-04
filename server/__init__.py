import logging
from dotenv import load_dotenv
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

load_dotenv()

MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_ADMIN_TOKEN = os.getenv("MILVUS_ADMIN_TOKEN")
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME")
MILVUS_USERNAME = os.getenv("MILVUS_USERNAME")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD")
MILVUS_ROLE_NAME = os.getenv("MILVUS_ROLE_NAME")
MILVUS_VECTOR_DIM = os.getenv("MILVUS_VECTOR_DIM")
MILVUS_PK_FIELD = os.getenv("MILVUS_PK_FIELD")
MILVUS_EMBEDDING_FIELD = os.getenv("MILVUS_EMBEDDING_FIELD")
MILVUS_TEXT_FIELD = os.getenv("MILVUS_TEXT_FIELD")

LLM_MODEL = os.getenv("LLM_MODEL")
LLM_GPU_UTIL = os.getenv("LLM_GPU_UTIL")
LLM_DTYPE = os.getenv("LLM_DTYPE")
LLM_MAX_MODEL_LEN = int(os.getenv("LLM_MAX_MODEL_LEN"))


def getLogger(name: str) -> logging.Logger:
    return logging.getLogger(name=name)

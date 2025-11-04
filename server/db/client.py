from pymilvus import DataType, MilvusClient
from server import getLogger
from server.db import (
    MILVUS_EMBEDDING_FIELD,
    MILVUS_PASSWORD,
    MILVUS_TEXT_FIELD,
    MILVUS_URI,
    MILVUS_USERNAME,
)

logger = getLogger(__name__)


def connect_milvus():
    try:
        token = f"{MILVUS_USERNAME}:{MILVUS_PASSWORD}"
        client = MilvusClient(uri=MILVUS_URI, token=token)
        logger.info("Successfully connected to Milvus.")
        return client
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        raise


def close_milvus(client: MilvusClient):
    try:
        client.close()
        logger.info("Milvus disconnection successful.")
    except Exception as e:
        logger.error(f"Milvus disconnect failed: {e}")


def get_embedding_schema():
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
    )
    schema.add_field(
        field_name="id", datatype=DataType.VARCHAR, max_length=512, is_primary=True
    )
    schema.add_field(
        field_name=MILVUS_EMBEDDING_FIELD, datatype=DataType.FLOAT_VECTOR, dim=1024
    )
    schema.add_field(
        field_name=MILVUS_TEXT_FIELD, datatype=DataType.VARCHAR, max_length=65535
    )
    schema.add_field(field_name="file_name", datatype=DataType.VARCHAR, max_length=256)
    return schema

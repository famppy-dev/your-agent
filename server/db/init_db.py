from pymilvus import MilvusClient

from server import (
    MILVUS_ADMIN_TOKEN,
    MILVUS_COLLECTION_NAME,
    MILVUS_PASSWORD,
    MILVUS_ROLE_NAME,
    MILVUS_URI,
    MILVUS_USERNAME,
    getLogger,
)
from server.db.client import (
    get_embedding_schema,
)

logger = getLogger(__name__)


def gen_schema(client: MilvusClient):
    """Create schema"""
    schema = get_embedding_schema()

    client.create_collection(
        collection_name=MILVUS_COLLECTION_NAME,
        schema=schema,
    )

    """Create Index"""
    index_params = MilvusClient.prepare_index_params()

    index_params.add_index(
        field_name="vector",
        metric_type="COSINE",
        index_type="HNSW",
        index_name="vector_index",
        params={"M": 16, "efConstruction": 200},
    )

    client.create_index(
        collection_name=MILVUS_COLLECTION_NAME,
        index_params=index_params,
        sync=False,  # Whether to wait for index creation to complete before returning. Defaults to True.
    )

    client.load_collection(MILVUS_COLLECTION_NAME)


def gen_initial_data(client: MilvusClient):
    """Generate initial data"""

    """User creation"""
    try:
        client.create_user(user_name=MILVUS_USERNAME, password=MILVUS_PASSWORD)
        logger.info(f"User '{MILVUS_USERNAME}' creation complete.")
    except Exception as e:
        logger.warning(f"Failed to create user: {e}")

    """Create a role"""
    try:
        client.create_role(role_name=MILVUS_ROLE_NAME)
        logger.info(f"Role '{MILVUS_ROLE_NAME}' creation complete.")
    except Exception as e:
        logger.warning(f"Failed to create role: {e}")

    try:
        client.grant_privilege_v2(
            role_name=MILVUS_ROLE_NAME,
            privilege="CollectionAdmin",
            collection_name="*",
        )
        client.grant_privilege_v2(
            role_name=MILVUS_ROLE_NAME,
            privilege="DatabaseAdmin",
            collection_name="*",
        )
        logger.info(f"Granting permission to role '{MILVUS_ROLE_NAME}' completed.")
    except Exception as e:
        logger.warning(f"Authorization failed: {e}")

    """Role assignment"""
    try:
        client.grant_role(user_name=MILVUS_USERNAME, role_name=MILVUS_ROLE_NAME)
        logger.info(
            f"Successfully assigned role '{MILVUS_ROLE_NAME}' to user '{MILVUS_USERNAME}'."
        )
    except Exception as e:
        logger.warning(f"Role assignment failed: {e}")

    user_info = client.describe_user(user_name=MILVUS_USERNAME)
    role_info = client.describe_role(role_name=MILVUS_ROLE_NAME)
    logger.info(f"User '{MILVUS_USERNAME}' role: {user_info.get('roles', [])}")
    logger.info(
        f"Role '{MILVUS_ROLE_NAME}' permissions: {role_info.get('privileges', [])}"
    )


if __name__ == "__main__":
    client = MilvusClient(uri=MILVUS_URI, token=MILVUS_ADMIN_TOKEN)

    gen_initial_data(client)

    gen_schema(client)

    client.close()

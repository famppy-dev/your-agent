from pymilvus import DataType, MilvusClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MILVUS_URI = "http://localhost:19530"
MILVUS_TOKEN = "root:"  # root:Milvus

DATABASE_NAME = "your_agent"
COLLECTION_NAME = "rag_docs"

USERNAME = "your_agent"
PASSWORD = "youragentpwd"

ROLE_NAME = "your_agent_role"


def connect_milvus():
    """Milvus Server Connection"""
    try:
        client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
        logger.info("Successfully connected to Milvus.")
        return client
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        raise


def gen_scheme(client: MilvusClient):
    """Use default database"""
    client.use_database(
        db_name=DATABASE_NAME
    )
    """Create schema"""
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
    )
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1024)

    client.create_collection(
        collection_name=COLLECTION_NAME,
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
        collection_name=COLLECTION_NAME,
        index_params=index_params,
        sync=False,  # Whether to wait for index creation to complete before returning. Defaults to True.
    )

    client.load_collection(COLLECTION_NAME)


def gen_initial_data(client: MilvusClient):
    """Generate initial data"""

    """User creation"""
    try:
        client.create_user(user_name=USERNAME, password=PASSWORD)
        logger.info(f"User '{USERNAME}' creation complete.")
    except Exception as e:
        logger.warning(f"Failed to create user: {e}")

    """Create a database"""
    try:
        client.create_database(db_name=DATABASE_NAME)
        logger.info(f"Database '{DATABASE_NAME}' creation complete.")
    except Exception as e:
        logger.warning(f"Failed to create database: {e}")

    """Create a role"""
    try:
        client.create_role(role_name=ROLE_NAME)
        logger.info(f"Role '{ROLE_NAME}' creation complete.")
    except Exception as e:
        logger.warning(f"Failed to create role: {e}")

    try:
        client.grant_privilege_v2(
            role_name=ROLE_NAME,
            privilege="CollectionAdmin",
            collection_name="*",
            db_name=DATABASE_NAME,
        )
        client.grant_privilege_v2(
            role_name=ROLE_NAME,
            privilege="DatabaseAdmin",
            collection_name="*",
            db_name=DATABASE_NAME,
        )
        logger.info(f"Granting permission to role '{ROLE_NAME}' completed.")
    except Exception as e:
        logger.warning(f"Authorization failed: {e}")

    """Role assignment"""
    try:
        client.grant_role(user_name=USERNAME, role_name=ROLE_NAME)
        logger.info(f"Successfully assigned role '{ROLE_NAME}' to user '{USERNAME}'.")
    except Exception as e:
        logger.warning(f"Role assignment failed: {e}")

    user_info = client.describe_user(user_name=USERNAME)
    role_info = client.describe_role(role_name=ROLE_NAME)
    logger.info(f"User '{USERNAME}' role: {user_info.get('roles', [])}")
    logger.info(f"Role '{ROLE_NAME}' permissions: {role_info.get('privileges', [])}")


if __name__ == "__main__":
    client = connect_milvus()

    gen_initial_data(client)

    gen_scheme(client)

    client.close()

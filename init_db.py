from pymilvus import MilvusClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MILVUS_URI = "http://localhost:19530"
MILVUS_TOKEN = "root:"  # root:Milvus

DATABASE_NAME = "lazy_agent"

USERNAME = "lazy_agent"
PASSWORD = "lazyagentpwd"

ROLE_NAME = "lazy_agent_role"


def connect_milvus():
    """Milvus Server Connection"""
    try:
        client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
        logger.info("Successfully connected to Milvus.")
        return client
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        raise

def gen_initial_data():
    """Generate initial data"""

    """User creation"""
    try:
        client.create_user(user_name=USERNAME, password=PASSWORD)
        logger.info(f"User '{USERNAME}' creation complete.")
    except Exception as e:
        logger.warning(f"Failed to create user: {e}")

    """Create a database"""
    try:
        client.create_database(
            db_name=DATABASE_NAME
        )
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

    gen_initial_data()

    client.close()

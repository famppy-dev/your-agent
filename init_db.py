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
    """Milvus 서버 연결"""
    try:
        client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
        logger.info("Milvus에 성공적으로 연결되었습니다.")
        return client
    except Exception as e:
        logger.error(f"연결 실패: {e}")
        raise

def gen_initial_data():
    """초기 데이터 생성"""

    """사용자 생성"""
    try:
        client.create_user(user_name=USERNAME, password=PASSWORD)
        logger.info(f"사용자 '{USERNAME}' 생성 완료.")
    except Exception as e:
        logger.warning(f"사용자 생성 실패: {e}")

    """데이터베이스 생성"""
    try:
        client.create_database(
            db_name=DATABASE_NAME
        )
        logger.info(f"데이터베이스 '{DATABASE_NAME}' 생성 완료.")
    except Exception as e:
        logger.warning(f"데이터베이스 생성 실패: {e}")

    """역할 생성"""
    try:
        client.create_role(role_name=ROLE_NAME)
        logger.info(f"역할 '{ROLE_NAME}' 생성 완료.")
    except Exception as e:
        logger.warning(f"역할 생성 실패: {e}")

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
        logger.info(f"역할 '{ROLE_NAME}'에 권한 부여 완료.")
    except Exception as e:
        logger.warning(f"권한 부여 실패: {e}")

    """역할 할당"""
    try:
        client.grant_role(user_name=USERNAME, role_name=ROLE_NAME)
        logger.info(f"사용자 '{USERNAME}'에게 역할 '{ROLE_NAME}' 할당 완료.")
    except Exception as e:
        logger.warning(f"역할 할당 실패: {e}")

    user_info = client.describe_user(user_name=USERNAME)
    role_info = client.describe_role(role_name=ROLE_NAME)
    logger.info(f"사용자 '{USERNAME}' 역할: {user_info.get('roles', [])}")
    logger.info(f"역할 '{ROLE_NAME}' 권한: {role_info.get('privileges', [])}")

if __name__ == "__main__":
    client = connect_milvus()

    gen_initial_data()

    client.close()

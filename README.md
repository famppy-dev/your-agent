# Lazy agent

### Install 
Docker composer 를 활용하여 Container 설치 

```sh
docker compose up -d
```

```
NAMES               STATUS                        PORTS
milvus-standalone   Up About a minute (healthy)   0.0.0.0:9091->9091/tcp, [::]:9091->9091/tcp, 0.0.0.0:19530->19530/tcp, [::]:19530->19530/tcp
milvus-minio        Up About a minute (healthy)   0.0.0.0:9000-9001->9000-9001/tcp, [::]:9000-9001->9000-9001/tcp
milvus-etcd         Up About a minute (healthy)   2379-2380/tcp
```

### Milvus


#### 관리자 화면 접속 
```
http://localhost:8080/
```
기본 접속 계정 `root`, 기본 비밀번호 `Milvus`

#### 사용자 추가 
`Role` 을 먼저 생성하고 `User` 생성


### 서버 
가상 환경 설정 

```sh
uv venv .venv
source .venv/bin/activate 
```

### DB 초기화

```sh
python init_db.py
```

<!-- pip3 install pymilvus==2.6.4 -->
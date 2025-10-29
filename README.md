# Your Agent

### Install 
Installing a container using Docker composer

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


#### Access the admin screen
```
http://localhost:8080/
```
Default login account `root`, default password `Milvus`

#### Add User
Create `Role` first, then create `User`


### Server
Setting up a virtual environment

```sh
uv venv .venv
source .venv/bin/activate 
```

### DB initialization

```sh
python init_db.py
```
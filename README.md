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
uv sync 
```

### DB initialization

```sh
python -m server.db.init_db
```

### Start server
```sh
./start.sh
```

### Start with docker
Please rename the `.env` file to `.env-prod` and modify the settings to suit your operating environment. Please check the model's `path` and `milvus` addresses.
```sh
docker compose up -d
```

#### If you need a build...
Please check the docker base image when building.
```sh
docker compose build
```

### Basic call (OpenAI Compatible)
```python
import json
import requests

with requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "local",
    "messages": [
        {
            "role": "user",
            "content": msg,
        }
    ]
}, headers={"Content-Type": "application/json"}, stream=stream) as response:
    response.raise_for_status()
    result = response.json()
    full_text = result["choices"][0]["message"]["content"]
    print(f"Result:\n{full_text}\n\nUsage:\n\n{result['usage']}")
```

### Basic call for embedding text
```python
import requests

with requests.post("http://localhost:8000/v1/embedding", json={"model": "local", "input": msg}, headers={"Content-Type": "application/json"}) as response:
    response.raise_for_status()
    result = response.json()
    print(result)
```

### Basic call for embedding files
```python
import requests

with open("sample.pdf", "rb") as f:
    files = {"file": f}

    with requests.post("http://localhost:8000/v1/rag/embedding", files=files) as response:
        response.raise_for_status()

        result = response.json()

        if result["result"] == 200:
            print(f"Result: \n{result['data']}")
        else:
            print(f"Result: \n{result['error']}")

```

### Basic call for Retrieval
```python
import json

import requests

url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "local",
    "messages": [
        {
            "role": "user",
            "content": msg,
        }
    ],
    "is_rag": True,
}

with requests.post(url, json=data, headers=headers, stream=stream) as response:
    response.raise_for_status()
    result = response.json()
    full_text = result["choices"][0]["message"]["content"]
    print(f"Result:\n{full_text}\n\nUsage:\n\n{result['usage']}")

```
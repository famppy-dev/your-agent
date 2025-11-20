FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV TZ=Asia/Seoul

COPY pyproject.toml /home/app/
COPY server /home/app/server

WORKDIR /home/app

RUN apt-get update && \
  apt install -y software-properties-common tzdata && \
  add-apt-repository ppa:deadsnakes/ppa && \
  apt install -y python3.13 python3-pip python3.13-dev && \
  pip install --upgrade pip && \
  pip install uv && \
  uv sync && \
  echo "$TZ" > /etc/timezone && \
  ln -sf /usr/share/zoneinfo/$TZ /etc/localtime

RUN rm -rf /var/lib/apt/lists/*

ENV PATH="/home/app/.venv/bin:$PATH"

CMD ["uvicorn", "server.server:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
#!/bin/bash

# イメージ名を設定
IMAGE_NAME="llm-driven-opt-img"
CONTAINER_NAME="llm-driven-opt-container"


# Dockerイメージを作成するかどうか確認
read -p "Do you want to build the Docker image? (y/n): " build_image

if [ "$build_image" = "y" ]; then
    # Dockerイメージをビルド
    docker build -t $IMAGE_NAME .
fi

# コンテナを実行
REPO_DIR=$(cd $(dirname $0)/..; pwd)
docker run --name $CONTAINER_NAME -v "$REPO_DIR":/app -d -it $IMAGE_NAME /bin/bash
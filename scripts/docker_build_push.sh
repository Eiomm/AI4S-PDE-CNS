#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-ai4s-cns-agent}"
IMAGE_TAG="${IMAGE_TAG:-$(date +%Y%m%d_%H%M%S)}"
REGISTRY_IMAGE="${1:-}"
BUILD_ARGS=()

if [[ -n "${DOCKER_BUILD_PROXY:-}" ]]; then
  BUILD_ARGS+=(
    --network host
    --build-arg "HTTP_PROXY=${DOCKER_BUILD_PROXY}"
    --build-arg "HTTPS_PROXY=${DOCKER_BUILD_PROXY}"
    --build-arg "ALL_PROXY=${DOCKER_BUILD_PROXY}"
    --build-arg "http_proxy=${DOCKER_BUILD_PROXY}"
    --build-arg "https_proxy=${DOCKER_BUILD_PROXY}"
    --build-arg "all_proxy=${DOCKER_BUILD_PROXY}"
  )
  echo "[docker] 构建代理已启用：${DOCKER_BUILD_PROXY}"
fi

if [[ -n "${DOCKER_BUILD_NO_PROXY:-}" ]]; then
  BUILD_ARGS+=(
    --build-arg "NO_PROXY=${DOCKER_BUILD_NO_PROXY}"
    --build-arg "no_proxy=${DOCKER_BUILD_NO_PROXY}"
  )
fi

if [[ -n "${DOCKER_CONDA_CHANNEL_ALIAS:-}" ]]; then
  BUILD_ARGS+=(--build-arg "CONDA_CHANNEL_ALIAS=${DOCKER_CONDA_CHANNEL_ALIAS}")
  echo "[docker] conda 镜像源：${DOCKER_CONDA_CHANNEL_ALIAS}"
fi

if [[ -n "${DOCKER_PIP_INDEX_URL:-}" ]]; then
  BUILD_ARGS+=(--build-arg "PIP_INDEX_URL=${DOCKER_PIP_INDEX_URL}")
  echo "[docker] pip 镜像源：${DOCKER_PIP_INDEX_URL}"
fi

if [[ -n "${DOCKER_PIP_TRUSTED_HOST:-}" ]]; then
  BUILD_ARGS+=(--build-arg "PIP_TRUSTED_HOST=${DOCKER_PIP_TRUSTED_HOST}")
fi

cd "${ROOT_DIR}"

echo "[docker] 构建本地镜像：${IMAGE_NAME}:${IMAGE_TAG}"
docker build "${BUILD_ARGS[@]}" -t "${IMAGE_NAME}:${IMAGE_TAG}" .

if [[ -z "${REGISTRY_IMAGE}" ]]; then
  echo "[docker] 未传入 registry 镜像地址，仅完成本地构建。"
  echo "[docker] 推送示例："
  echo "  bash scripts/docker_build_push.sh registry.cn-hangzhou.aliyuncs.com/<namespace>/<repo>:${IMAGE_TAG}"
  exit 0
fi

echo "[docker] 打标签：${REGISTRY_IMAGE}"
docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${REGISTRY_IMAGE}"

echo "[docker] 推送镜像：${REGISTRY_IMAGE}"
docker push "${REGISTRY_IMAGE}"

#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
IMAGE="registry.cn-shenzhen.aliyuncs.com/ai4s-junao/ai4s-junao:${TAG}"

echo "[acr] 目标镜像：${IMAGE}"
echo "[acr] 请先确认已经登录："
echo "      docker login registry.cn-shenzhen.aliyuncs.com"

bash "$(dirname "${BASH_SOURCE[0]}")/docker_build_push.sh" "${IMAGE}"

echo "[acr] 已推送镜像："
echo "${IMAGE}"

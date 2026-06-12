#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "用法：bash docker_build.sh <version>"
  echo "示例：DOCKER_REGISTRY=registry.cn-shenzhen.aliyuncs.com/ai4s-junao bash docker_build.sh v1"
  exit 2
fi

if [[ -z "${DOCKER_REGISTRY:-}" ]]; then
  echo "错误：请先设置 DOCKER_REGISTRY，例如 registry.cn-shenzhen.aliyuncs.com/ai4s-junao" >&2
  exit 2
fi

VERSION="$1"
IMAGE="${DOCKER_REGISTRY%/}/ai4s-junao:${VERSION}"

echo "[docker] 目标镜像：${IMAGE}"
bash "$(dirname "${BASH_SOURCE[0]}")/scripts/docker_build_push.sh" "${IMAGE}"
echo "[docker] 提交镜像地址：${IMAGE}"

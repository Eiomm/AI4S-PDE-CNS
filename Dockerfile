# Base image
# 使用 miniforge 是因为本项目需要 RDKit/OpenBabel/fpocket/AiZynthFinder 等 conda-forge 工具。
FROM quay.io/condaforge/miniforge3:24.11.3-0

SHELL ["/bin/bash", "-lc"]

WORKDIR /workspace

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG ALL_PROXY
ARG NO_PROXY
ARG http_proxy
ARG https_proxy
ARG all_proxy
ARG no_proxy
ARG CONDA_CHANNEL_ALIAS=https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

ENV PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_PROGRESS_BAR=off \
    PIP_RETRIES=10 \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST} \
    CONDA_REMOTE_CONNECT_TIMEOUT_SECS=60 \
    CONDA_REMOTE_READ_TIMEOUT_SECS=300 \
    SAISDATA_DIR=/saisdata \
    SAISRESULT_DIR=/saisresult \
    AGENT_RUNNER=agent \
    AGENT_MODE=competition \
    AGENT_DOCKING_LIMIT=10 \
    AGENT_ROUNDS=8 \
    AGENT_PER_ROUND=32 \
    AGENT_TOP_K=10 \
    AI4S_ROUTE_LIMIT_PER_ROUND=10 \
    AI4S_VINA_FEEDBACK_PER_ROUND=1 \
    AI4S_AGENT_MEMORY_LIMIT=20 \
    CHEM_EVOLVE_LLM_ENABLED=1 \
    AI4S_ROUTE_ENGINE=aizynthfinder \
    AIZYNTHFINDER_CONFIG=/workspace/data/aizynthfinder/config.yml \
    AI4S_AGENT_MODEL=openai/claude-opus-4-8 \
    AI4S_AGENT_PROVIDER=openai \
    AI4S_AGENT_BASE_URL=https://api.gpt.ge/v1 \
    AI4S_AGENT_BYPASS_PROXY=1 \
    AI4S_AGENT_MEMORY_FILE=/workspace/data/agent_experience.jsonl \
    LITELLM_LOCAL_MODEL_COST_MAP=True

# 安装依赖。默认使用清华 conda/pip 镜像源；如源不可用，可在 docker build 时覆盖
# CONDA_CHANNEL_ALIAS / PIP_INDEX_URL / PIP_TRUSTED_HOST。
COPY environment.docker.yml /tmp/environment.docker.yml
COPY requirements-docker-core.txt /tmp/requirements-docker-core.txt
COPY requirements-docker-retro.txt /tmp/requirements-docker-retro.txt
RUN printf '%s\n' \
        'channels:' \
        '  - conda-forge' \
        "channel_alias: ${CONDA_CHANNEL_ALIAS}" \
        'show_channel_urls: true' \
        > /root/.condarc \
    && python -m pip config set global.index-url "${PIP_INDEX_URL}" \
    && python -m pip config set global.trusted-host "${PIP_TRUSTED_HOST}" \
    && mamba env create -f /tmp/environment.docker.yml \
    && conda run --no-capture-output -n ai4s-chem-evolve \
        python -m pip install --no-cache-dir --prefer-binary -v -r /tmp/requirements-docker-core.txt \
    && conda run --no-capture-output -n ai4s-chem-evolve \
        python -m pip install --no-cache-dir --prefer-binary -v -r /tmp/requirements-docker-retro.txt \
    && mamba clean -afy

# 把当前项目构建到镜像工作目录。
COPY . /workspace

# 赛题审查路径和启动脚本权限。
RUN mkdir -p /app \
    && cp -a /workspace/app/training_code /app/training_code \
    && cp /workspace/run.sh /app/run.sh \
    && test -f /app/training_code/README.md \
    && test -f /workspace/configs/docker_llm.env \
    && ! grep -q '<your-docker-llm-api-key>' /workspace/configs/docker_llm.env \
    && chmod +x /app/run.sh \
    && chmod +x /workspace/docker_build.sh \
    && chmod +x /workspace/run.sh \
    && chmod +x /workspace/scripts/run_competition_final.sh \
    && chmod +x /workspace/scripts/run_competition_4h.sh

# 镜像启动后执行官方要求的 /app/run.sh。
CMD ["bash", "/app/run.sh"]

# Docker 与阿里云 ACR 提交说明

## 1. 本地构建

`examples/tianchi_docker.zip` 里的样例格式是：

```text
Dockerfile
requirements.txt
run.sh
main.py
docker_build.sh
```

我们的提交格式对应为：

```text
Dockerfile             构建完整 agent 镜像
run.sh                 容器启动脚本，调用 Code/main.py
docker_build.sh        天池样例风格的构建推送入口，读取 DOCKER_REGISTRY 和版本号
Code/main.py           复赛入口，读取 /saisdata/target1-3.pdb，兼容 /saisdata/37
environment.docker.yml conda 依赖
requirements-docker-*.txt pip 依赖
scripts/docker_build_push.sh / scripts/push_aliyun_ai4s_junao.sh  构建推送脚本
```

天池/阿里云最终提交的是镜像地址，不是把 `tianchi_docker.zip` 上传为代码包。官方运行镜像时会挂载 `/saisdata` 和 `/saisresult`，并要求执行脚本放在 `/app/run.sh`。

```bash
cd /data/wangjunao/AI4S
bash scripts/docker_build_push.sh
```

这会构建本地镜像：

```text
ai4s-cns-agent:<timestamp>
```

Dockerfile 使用 `quay.io/condaforge/miniforge3:24.11.3-0` 作为基础镜像，避免从 Docker Hub 拉取 `condaforge/miniforge3` 时被网络重置。

Docker 构建环境使用 `environment.docker.yml` 安装 conda 依赖，再用 `requirements-docker-core.txt` 和 `requirements-docker-retro.txt` 安装 pip 依赖。Dockerfile 默认把 conda-forge 指到清华镜像 `https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud`，并把 pip 指到 `https://pypi.tuna.tsinghua.edu.cn/simple`。这样全量工具仍会装进去，但 pip 阶段会显示具体正在处理哪个包；`aizynthfinder` 被单独放在最后，方便判断是否卡在逆合成工具安装。

如果卡在 pip 依赖下载，通常是 pip/conda 下载依赖太慢。可以只给构建阶段加代理，不要把 Docker daemon 永久代理打开：

```bash
cd /data/wangjunao/AI4S
export DOCKER_BUILD_PROXY=socks5://127.0.0.1:7890
export DOCKER_BUILD_NO_PROXY=localhost,127.0.0.1,registry.cn-shenzhen.aliyuncs.com,.aliyuncs.com
sudo -E bash scripts/push_aliyun_ai4s_junao.sh v1
```

这里脚本会对 `docker build` 使用 `--network host`，所以 `127.0.0.1:7890` 指的是宿主机上的代理服务。推送到阿里云 ACR 仍然走直连。

如果清华源临时不可用，可以覆盖构建镜像源：

```bash
export DOCKER_CONDA_CHANNEL_ALIAS=https://mirrors.ustc.edu.cn/anaconda/cloud
export DOCKER_PIP_INDEX_URL=https://pypi.mirrors.ustc.edu.cn/simple
export DOCKER_PIP_TRUSTED_HOST=pypi.mirrors.ustc.edu.cn
bash scripts/docker_build_push.sh
```

容器默认行为：

- 执行 `/app/run.sh`
- 读取 `/saisdata/target1.pdb`
- 读取 `/saisdata/target2.pdb`
- 读取 `/saisdata/target3.pdb`
- 如果直接路径不存在，兼容读取 `/saisdata/37/target1-3.pdb`
- 写出 `/saisresult/result.zip`
- zip 内包含 `result1.csv`、`result2.csv`、`result3.csv`
- 复制审查用预处理/训练说明到 `/app/training_code`

容器默认走严格 agent 主路径：

```text
AGENT_MODE=competition
CHEM_EVOLVE_LLM_ENABLED=1
AGENT_ROUNDS=8
AGENT_PER_ROUND=32
AGENT_TOP_K=10
AGENT_DOCKING_LIMIT=10
AI4S_ROUTE_ENGINE=aizynthfinder
AIZYNTHFINDER_CONFIG=/workspace/data/aizynthfinder/config.yml
AI4S_ROUTE_LIMIT_PER_ROUND=10
AI4S_VINA_FEEDBACK_PER_ROUND=1
AI4S_AGENT_MEMORY_LIMIT=20
AI4S_AGENT_MEMORY_FILE=/workspace/data/agent_experience.jsonl
```

`competition` 模式会要求 Vina/OpenBabel/AiZynthFinder 可用，并默认调用 API 让 agent 推理。为适配不方便额外传环境变量的平台提交方式，镜像会打包 `configs/docker_llm.env`，并在 `/app/run.sh` 启动时自动加载。构建前必须把其中的 `DOCKER_LLM_API_KEY` 占位符替换为 Docker 专用 key；Dockerfile 会检查占位符，未替换时直接构建失败。原始 DUD-E actives/decoys 不会打包进镜像，镜像只保留聚合的 `benchmark_prior.json` 和精简后的 `agent_experience.jsonl` 经验种子。经验种子可作为训练/搜索沉淀随镜像上传，但 agent 会拒绝长期记忆 SMILES 的原样复用，避免变成固定答案库。

赛题要求的 `/app/training_code/README.md` 会在 Docker 构建时从仓库的 `app/training_code` 复制过去。这里记录的是可复现的 benchmark descriptor prior 预处理流程，不包含固定候选分子库。

## 2. 本地容器测试

准备模拟输入：

```bash
mkdir -p /tmp/ai4s_saisdata /tmp/ai4s_saisresult
cp examples/target.pdb /tmp/ai4s_saisdata/target1.pdb
cp examples/target.pdb /tmp/ai4s_saisdata/target2.pdb
cp examples/target.pdb /tmp/ai4s_saisdata/target3.pdb
```

运行容器：

```bash
docker run --rm \
  -v /tmp/ai4s_saisdata:/saisdata:ro \
  -v /tmp/ai4s_saisresult:/saisresult \
  ai4s-cns-agent:<timestamp>
```

镜像内置的 `configs/docker_llm.env` 会被 `/app/run.sh` 自动加载。核心变量是：

```bash
AI4S_AGENT_API_KEY_ENVS=DOCKER_LLM_API_KEY
DOCKER_LLM_API_KEY="<your-docker-only-key>"
```

如果 Docker API 和本地测试 API 类型相同，就不需要改 `AI4S_AGENT_PROVIDER`、`AI4S_AGENT_MODEL`、`AI4S_AGENT_BASE_URL`；它们会沿用 Dockerfile 默认值。只有当 Docker API 的模型名或网关地址不同，才在 `configs/docker_llm.env` 里额外覆盖这些变量。

检查结果：

```bash
python scripts/inspect_result_zip.py \
  /tmp/ai4s_saisresult/result.zip \
  result1.csv result2.csv result3.csv
```

## 3. 推送到阿里云 ACR

你的 ACR 控制台地址：

```text
https://cr.console.aliyun.com/repository/cn-shenzhen/ai4s-junao/ai4s-junao/details
```

对应 Docker 镜像地址：

```text
registry.cn-shenzhen.aliyuncs.com/ai4s-junao/ai4s-junao:<tag>
```

推荐直接使用项目脚本：

```bash
docker login registry.cn-shenzhen.aliyuncs.com
bash scripts/push_aliyun_ai4s_junao.sh v1
```

如果想按 `examples/tianchi_docker/docker_build.sh` 的样例方式运行，也可以使用根目录脚本：

```bash
docker login registry.cn-shenzhen.aliyuncs.com
export DOCKER_REGISTRY=registry.cn-shenzhen.aliyuncs.com/ai4s-junao
bash docker_build.sh v1
```

推送成功后，比赛平台或后续部署应使用这个镜像地址：

```text
registry.cn-shenzhen.aliyuncs.com/ai4s-junao/ai4s-junao:v1
```

### 个人版常见格式

```bash
docker login registry.cn-hangzhou.aliyuncs.com
bash scripts/docker_build_push.sh \
  registry.cn-hangzhou.aliyuncs.com/<namespace>/<repo>:<tag>
```

### 企业版常见格式

```bash
docker login <instance>-registry.cn-hangzhou.cr.aliyuncs.com
bash scripts/docker_build_push.sh \
  <instance>-registry.cn-hangzhou.cr.aliyuncs.com/<namespace>/<repo>:<tag>
```

实际地域、namespace、repo 以阿里云控制台“镜像仓库 > 操作指南”为准。

## 4. 常见覆盖参数

第一次上传只想检查镜像是否能跑通，可以用轻量参数在本地先试：

```bash
docker run --rm \
  -e AGENT_ROUNDS=2 \
  -e AGENT_PER_ROUND=16 \
  -e AGENT_TOP_K=5 \
  -e AGENT_DOCKING_LIMIT=5 \
  -e AI4S_ROUTE_LIMIT_PER_ROUND=4 \
  -e AI4S_VINA_FEEDBACK_PER_ROUND=0 \
  -v /tmp/ai4s_saisdata:/saisdata:ro \
  -v /tmp/ai4s_saisresult:/saisresult \
  ai4s-cns-agent:<timestamp>
```

正式跑分建议用 Dockerfile 默认稳态 profile，或显式写出来便于审计：

如果要在容器内启用更多搜索：

```bash
docker run --rm \
  -e AGENT_MODE=competition \
  -e AGENT_ROUNDS=8 \
  -e AGENT_PER_ROUND=32 \
  -e AGENT_TOP_K=10 \
  -e AGENT_DOCKING_LIMIT=10 \
  -e AI4S_ROUTE_LIMIT_PER_ROUND=10 \
  -e AI4S_VINA_FEEDBACK_PER_ROUND=1 \
  -v /tmp/ai4s_saisdata:/saisdata:ro \
  -v /tmp/ai4s_saisresult:/saisresult \
  ai4s-cns-agent:<timestamp>
```

`AGENT_ROUNDS`、`AGENT_PER_ROUND`、`AGENT_TOP_K` 必须为正整数。`docking` 和 `competition` 模式下 `AGENT_DOCKING_LIMIT` 也必须为正整数；启用 LLM 时必须能找到 LiteLLM 和 API key；选择 AiZynthFinder、Vina 或 SBDD 命令时对应工具必须可用。参数或工具配置错误会在清理输出目录前直接失败。

本地只检查提交格式时可以用：

```bash
-e AGENT_MODE=proxy
-e CHEM_EVOLVE_LLM_ENABLED=0
-e AI4S_ROUTE_ENGINE=aizynthfinder
-e AGENT_DOCKING_LIMIT=0
```

## 5. 注意事项

- 不要把 `.env`、API key、阿里云密码写入镜像或 git。
- 首次推送前，需要在阿里云 ACR 控制台设置 Registry 登录密码。
- 推送失败时，优先检查登录服务器地址、namespace、仓库名和 tag 是否一致。

# Official LLM Logging Proxy

Install dependencies:

```bash
/root/miniconda3/envs/ai4s-pde-cns/bin/pip install -r task_log_sample/openai-log/requirements.txt
/root/miniconda3/envs/ai4s-pde-cns/bin/pip install openai
```

Start the proxy. This Task1 repo defaults to the third-party OpenAI-compatible
API target used by the old `apifox_gpt_ge_gpt55_official_proxy` profile:

```bash
bash scripts/start_openai_proxy.sh
```

Point the Agent API base URL to:

```text
http://127.0.0.1:8080/v1
```

Set one of:

```bash
export APIFOX_GPT_GE_API_KEY=...
# or
export VAPI_API_KEY=...
```

If `.env` exists in the Task1 root, `scripts/start_openai_proxy.sh` loads it
automatically and does not print secrets.

The proxy forwards OpenAI-compatible requests to `https://api.gpt.ge` by default
and writes logs under `logs/`.
The official note says Anthropic `/v1/messages` requires adapting response
parsing in `proxy.py`.

Submission bundles must include:

```text
code/
submission.json
task1_pred.hdf5
task1_time.csv
task1_logs.log
```

`submission.json`:

```json
{
  "submission_id": "your_team_name",
  "problem_id": "PDE_Burgers",
  "code_path": "code"
}
```

The official sample under the old repository also includes optional
`methodology` and `submission` fields and uses plain-text training logs. Our
internal logs may be JSONL for structure, but submission tooling must accept
plain text logs as valid.

The final `code/` snapshot should be generated through the Agent path and be
traceable through official proxy logs before final submission.

Current status: the lightweight runtime in `src/` was created during repository
reconstruction. It is acceptable as a base, but final submission-grade code
should be regenerated or modified through the GPT-5.5 Agent path configured in
`configs/agent_gpt55.yaml`, with requests passing through this proxy.

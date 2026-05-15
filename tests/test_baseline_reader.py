from agent.baseline_reader import index_baseline_repos, summarize_baseline_context
from agent.pde_observer import observe_research_context
from agent.pde_planner import ExperimentPlanner
from agent.logging import LLMCallLogger
from agent.pde_journal import ExperimentJournal


class RecordingClient:
    provider = "recording"
    model = "recording"

    def __init__(self):
        self.messages = []

    def complete(self, messages):
        self.messages = messages
        return {
            "content": (
                '{"intent":"stop","hypothesis":"baseline context inspected","action_type":"stop",'
                '"params":{"reason":"done"},"expected_effect":"none","risk":"none"}'
            )
        }


def test_baseline_reader_indexes_known_repos_and_symbols(tmp_path):
    root = tmp_path / "third_party" / "baseline"
    fno_file = root / "PDEBench" / "pdebench" / "models" / "fno" / "fno.py"
    fno_file.parent.mkdir(parents=True)
    fno_file.write_text("class FNO1d:\n    pass\n\ndef train():\n    pass\n", encoding="utf-8")
    deeponet_file = root / "deeponet" / "src" / "deeponet_pde.py"
    deeponet_file.parent.mkdir(parents=True)
    deeponet_file.write_text("class DeepONet:\n    pass\n", encoding="utf-8")
    ignored = root / "PDEBench" / "data" / "large.npy"
    ignored.parent.mkdir(parents=True)
    ignored.write_bytes(b"no")

    index = index_baseline_repos(root)

    assert "PDEBench" in index
    assert "deeponet" in index
    assert any(item["path"].endswith("pdebench/models/fno/fno.py") for item in index["PDEBench"]["files"])
    assert index["PDEBench"]["files"][0]["symbols"] == ["FNO1d", "train"]
    assert not any("large.npy" in item["path"] for repo in index.values() for item in repo["files"])


def test_summarize_baseline_context_mentions_agent_use(tmp_path):
    root = tmp_path / "third_party" / "baseline"
    fno_file = root / "PDEBench" / "pdebench" / "models" / "fno" / "fno.py"
    fno_file.parent.mkdir(parents=True)
    fno_file.write_text("class SpectralConv1d:\n    pass\nclass FNO1d:\n    pass\n", encoding="utf-8")

    context = summarize_baseline_context(root)

    assert context["PDEBench"]["agent_use"]
    assert "FNO" in context["PDEBench"]["summary"]
    assert context["PDEBench"]["source_files"][0].endswith("pdebench/models/fno/fno.py")


def test_observer_includes_baseline_context(tmp_path):
    fno_file = tmp_path / "third_party" / "baseline" / "PDEBench" / "pdebench" / "models" / "fno" / "fno.py"
    fno_file.parent.mkdir(parents=True)
    fno_file.write_text("class FNO1d:\n    pass\n", encoding="utf-8")

    state = observe_research_context(tmp_path)

    assert "baseline_context" in state
    assert "PDEBench" in state["baseline_context"]


def test_planner_prompt_requires_source_files_for_code_evolution(tmp_path):
    client = RecordingClient()
    planner = ExperimentPlanner(
        client=client,
        logger=LLMCallLogger(tmp_path / "planner.log"),
        journal=ExperimentJournal(tmp_path / "journal.json"),
    )

    planner.plan_next({"task": "task1", "baseline_context": {"PDEBench": {"source_files": ["fno.py"]}}})

    prompt = "\n".join(message["content"] for message in client.messages)
    assert "source_files" in prompt
    assert "source_method" in prompt
    assert "third_party/baseline" in prompt

"""Lưu xác nhận qua API thật; không thay ảnh dashboard yêu cầu trong lab."""
import config
import json
from datetime import datetime, timezone
from pathlib import Path
from langsmith import Client
from prompts import PROMPT_V1_NAME, PROMPT_V2_NAME, SYSTEM_V1, SYSTEM_V2


def main():
    client = Client(api_key=config.LANGSMITH_API_KEY)
    project = client.read_project(project_name=config.LANGSMITH_PROJECT)
    proof = {"verified_at": datetime.now(timezone.utc).isoformat(),
             "project_url": project.url, "project_name": project.name,
             "steps": {}, "prompts": {}}
    for name in ("rag-query", "ab-rag-query"):
        runs = list(client.list_runs(project_name=project.name, is_root=True,
                                     filter=f'eq(name, "{name}")', limit=100))
        successful = [run for run in runs if run.end_time and not run.error]
        item = {"successful_traces": len(successful),
                "errors": sum(bool(run.error) for run in runs),
                "run_ids": [str(run.id) for run in successful]}
        proof["steps"][name] = item
        print(f"{name}: {len(successful)} successful traces")
        if name == "rag-query" and successful:
            detailed = client.read_run(successful[0].id, load_child_runs=True)
            children, stack = [], list(detailed.child_runs or [])
            while stack:
                child = stack.pop()
                children.append(child)
                stack.extend(child.child_runs or [])
            retrieval = [child for child in children if child.run_type == "retriever"]
            item.update(sample_has_question="question" in detailed.inputs,
                        sample_has_answer=bool(detailed.outputs),
                        sample_has_retrieval=any(bool(child.outputs) for child in retrieval))
        if name == "ab-rag-query":
            item["version_counts"] = {
                version: sum((run.outputs or {}).get("version") == version for run in successful)
                for version in ("v1", "v2")}
            item["contexts_present"] = all(
                len((run.outputs or {}).get("contexts", [])) == 3 for run in successful)
            print("Routing:", item["version_counts"])
    for name, system in ((PROMPT_V1_NAME, SYSTEM_V1), (PROMPT_V2_NAME, SYSTEM_V2)):
        prompt = client.pull_prompt(name)
        proof["prompts"][name] = {
            "pulled_from_hub": True,
            "matches_code": prompt.messages[0].prompt.template == system,
            "input_variables": sorted(prompt.input_variables),
            "commit_hash": (prompt.metadata or {}).get("lc_hub_commit_hash")}
    path = Path(__file__).resolve().parents[1] / "evidence" / "01_02_langsmith_api_verification.json"
    path.write_text(json.dumps(proof, indent=2), encoding="utf-8")
    print("Project:", project.url)
    print("Saved API verification; dashboard screenshots still required.")
    rag, ab = proof["steps"]["rag-query"], proof["steps"]["ab-rag-query"]
    passed = (rag["successful_traces"] >= 50 and ab["successful_traces"] >= 50
              and rag.get("sample_has_question") and rag.get("sample_has_answer")
              and rag.get("sample_has_retrieval") and ab["contexts_present"]
              and all(ab["version_counts"].values())
              and all(item["matches_code"] for item in proof["prompts"].values()))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Kiểm tra bài nộp tại máy; không thay việc xác nhận dashboard LangSmith."""
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent
METRICS = ("faithfulness", "answer_relevancy", "context_recall", "context_precision")
EVIDENCE_FILES = (
    "01_langsmith_traces.png", "02_prompt_hub.png", "02_ab_routing_log.txt",
    "03_ragas_scores.png", "03_ragas_report.json",
    "04_pii_demo_log.txt", "04_json_demo_log.txt",
)


def main():
    """Trả exit code 1 nếu thiếu evidence, sai report hoặc .env bị track."""
    failures = []

    def check(condition, message):
        print(f"{'OK' if condition else 'THIEU/LOI'}: {message}")
        if not condition:
            failures.append(message)

    for name in EVIDENCE_FILES:
        path = ROOT / "evidence" / name
        present = path.is_file() and path.stat().st_size > 0
        check(present, f"evidence/{name}")
        if present and path.suffix == ".png":
            with path.open("rb") as stream:
                check(stream.read(8) == b"\x89PNG\r\n\x1a\n", f"PNG signature: {name}")

    report_path = ROOT / "evidence" / "03_ragas_report.json"
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            check(report.get("samples_per_version") == 50, "50 QA cho moi prompt")
            scores = [report[key] for key in ("prompt_v1_scores", "prompt_v2_scores")]
            for version, values in zip(("v1", "v2"), scores):
                check(all(isinstance(values.get(metric), (int, float))
                          and not isinstance(values.get(metric), bool)
                          and math.isfinite(values[metric])
                          for metric in METRICS), f"{version}: du 4 metric huu han")
            best = max(values["faithfulness"] for values in scores)
            check(math.isfinite(best) and best >= 0.8, "Faithfulness >= 0.8")
            data_report = ROOT / "data" / "ragas_report.json"
            check(data_report.is_file() and json.loads(data_report.read_text(encoding="utf-8")) == report,
                  "Report evidence khop data/ragas_report.json")
        except (ValueError, KeyError, TypeError, AttributeError, OSError) as error:
            check(False, f"Report RAGAS khong hop le ({type(error).__name__})")

    ab_log = ROOT / "evidence" / "02_ab_routing_log.txt"
    if ab_log.is_file():
        log = ab_log.read_text(encoding="utf-8")
        check("[prompt-v1]" in log and "[prompt-v2]" in log, "A/B log co ca v1 va v2")
        check(sum("[prompt-v" in line and " Q:" in line for line in log.splitlines()) >= 50,
              "A/B log co it nhat 50 cau hoi")

    try:
        result = subprocess.run(["git", "ls-files", "--", ".env"], cwd=ROOT,
                                capture_output=True, text=True, check=True)
        check(not result.stdout.strip(), ".env khong bi Git track")
        ignored = subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=ROOT)
        check(ignored.returncode == 0, ".env duoc Git ignore")
    except (OSError, subprocess.CalledProcessError):
        check(False, "Khong kiem tra duoc Git")

    print("Can xac nhan trong LangSmith UI: >=100 traces, context/answer va 2 prompt tren Hub.")
    print(f"Ket qua: {len(failures)} muc chua dat.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Ghi stdout thật ra evidence, đồng thời hiển thị trong terminal."""
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
import sys

EVIDENCE_DIR = Path(__file__).resolve().parents[2] / "evidence"


class Tee:
    """Gửi cùng nội dung đến terminal và file log UTF-8."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextmanager
def capture_log(filename):
    """Lưu log của lần chạy hiện tại; không tạo kết quả mô phỏng."""
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    with (EVIDENCE_DIR / filename).open("w", encoding="utf-8") as output:
        with redirect_stdout(Tee(sys.stdout, output)):
            yield

"""Bước 4: custom PII/JSON validators dùng Guardrails FIX, không cần LLM."""
import config
import ast
import io
import json
import re
import tokenize
from guardrails import Guard, OnFailAction
from guardrails.validators import Validator, register_validator, PassResult, FailResult
from utils.evidence import capture_log


@register_validator(name="custom/pii-detector", data_type="string")
class PIIDetector(Validator):
    """Che email, phone (US/VN), SSN và thẻ 16 chữ số bằng regex."""

    # Số dài được che trước để phone không lấy nhầm một phần của credit card.
    PII_PATTERNS = {
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "CREDIT_CARD": r"(?<!\w)(?:\d{4}[- ]?){3}\d{4}(?!\w)",
        "SSN": r"(?<!\w)\d{3}-\d{2}-\d{4}(?!\w)",
        "PHONE": (
            r"(?<![\w+])(?:"
            r"(?:\+?1[-. ]?)?(?:\(\d{3}\)|\d{3})[-. ]?\d{3}[-. ]?\d{4}"
            r"|(?:\+84|0)[-. ]?\d{3}[-. ]?\d{3}[-. ]?\d{3}"
            r")(?!\w)"
        ),
    }

    def validate(self, value: str, metadata: dict):
        """FailResult.fix_value là output an toàn được Guard FIX áp dụng."""
        redacted_text = value
        found_types = []
        for pii_type, pattern in self.PII_PATTERNS.items():
            def redact(match):
                found_types.append(pii_type)
                return f"[{pii_type}_REDACTED]"
            redacted_text = re.sub(pattern, redact, redacted_text)
        if found_types:
            return FailResult(error_message=f"Phát hiện PII: {', '.join(found_types)}",
                              fix_value=redacted_text)
        return PassResult()


@register_validator(name="custom/json-formatter", data_type="string")
class JSONFormatter(Validator):
    """Sửa fences, nháy đơn và trailing commas mà giữ nội dung chuỗi."""

    @staticmethod
    def _repair(text: str) -> str:
        """Chỉ thay quote/token ngoài string, không làm hỏng dấu nháy trong dữ liệu."""
        text = re.sub(r"\A\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*\Z", "", text).strip()
        try:
            tokens = []
            for token in tokenize.generate_tokens(io.StringIO(text).readline):
                if token.type == tokenize.STRING and token.string.startswith("'"):
                    token = token._replace(string=json.dumps(ast.literal_eval(token.string),
                                                           ensure_ascii=False))
                tokens.append(token)
            text = tokenize.untokenize(tokens)
        except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
            # Nếu token hóa thất bại, để parser quyết định và dùng JSON dự phòng.
            pass
        # Match toàn bộ JSON string trước để không xóa dấu phẩy bên trong string.
        pattern = r'"(?:\\.|[^"\\])*"|,\s*(?=[}\]])'
        return re.sub(pattern, lambda m: "" if m.group().startswith(",") else m.group(), text)

    def validate(self, value: str, metadata: dict):
        """Giữ JSON đúng; trả bản sửa hoặc JSON error nếu không thể parse."""
        try:
            json.loads(value)
            return PassResult()
        except json.JSONDecodeError:
            pass
        try:
            parsed = json.loads(self._repair(value))
            return FailResult(error_message="JSON lỗi, đã tự sửa",
                              fix_value=json.dumps(parsed, indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            fallback = json.dumps({"error": "Không thể phân tích JSON", "raw": value[:200]},
                                  ensure_ascii=False)
            return FailResult(error_message="Không thể sửa JSON", fix_value=fallback)


def demo_pii_guard():
    """Demo PII giả; assert nội dung thực tế để lỗi FIX không bị bỏ sót."""
    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))
    cases = [
        ("Email", "Contact John at john.doe@example.com for details.",
         "Contact John at [EMAIL_REDACTED] for details."),
        ("Phone", "Call our support line at (555) 867-5309.",
         "Call our support line at [PHONE_REDACTED]."),
        ("SSN", "Patient SSN is 123-45-6789 on file.",
         "Patient SSN is [SSN_REDACTED] on file."),
        ("Credit Card", "Payment made with card 4532 1234 5678 9010.",
         "Payment made with card [CREDIT_CARD_REDACTED]."),
        ("Multi-PII", "Email: alice@example.com, Phone: 555-123-4567",
         "Email: [EMAIL_REDACTED], Phone: [PHONE_REDACTED]"),
        ("Clean", "No sensitive information in this text.",
         "No sensitive information in this text."),
        ("Vietnam phone", "So gia lap: +84 912 345 678 / 0912345678",
         "So gia lap: [PHONE_REDACTED] / [PHONE_REDACTED]"),
    ]
    print("Demo: PII Detection & Redaction — dữ liệu giả")
    for label, text, expected in cases:
        result = guard.validate(text)
        print(f"\n[{label}]\n  Input:  {text}\n  Output: {result.validated_output}")
        if result.validated_output != expected:
            raise AssertionError(f"PII output sai trong case {label}")
    print(f"\n✅ {len(cases)}/{len(cases)} PII cases đạt.")


def demo_json_guard():
    """Demo cả JSON hợp lệ, lỗi sửa được và fallback có parse hợp lệ."""
    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))
    cases = [
        ("Valid JSON", '{"name": "Alice", "age": 30}', {"name": "Alice", "age": 30}),
        ("Markdown fences", '```json\n{"name": "Bob"}\n```', {"name": "Bob"}),
        ("Single quotes", "{'name': 'Charlie', 'score': 95}", {"name": "Charlie", "score": 95}),
        ("Trailing comma", '{"key": "value",}', {"key": "value"}),
        ("Truly invalid", "This is not JSON at all: ??? {]", None),
        ("Combined", "```json\n{'items': [1, 2,],}\n```", {"items": [1, 2]}),
        ("Preserve string", '{"text": "don\'t delete ,}",}', {"text": "don't delete ,}"}),
    ]
    print("Demo: JSON Formatting & Repair")
    for label, text, expected in cases:
        result = guard.validate(text)
        parsed = json.loads(result.validated_output)
        if expected is None:
            if parsed.get("error") != "Không thể phân tích JSON" or parsed.get("raw") != text:
                raise AssertionError("JSON fallback sai")
        elif parsed != expected:
            raise AssertionError(f"JSON output sai trong case {label}")
        print(f"\n[{label}]\n  Input:  {text}\n  Output: {result.validated_output}")
    print(f"\n✅ {len(cases)}/{len(cases)} JSON cases đạt.")


def main():
    """Lưu hai log độc lập từ lần chạy Guard thật."""
    with capture_log("04_pii_demo_log.txt"):
        demo_pii_guard()
    with capture_log("04_json_demo_log.txt"):
        demo_json_guard()
    print("✅ Bước 4 hoàn thành, đã lưu hai log trong evidence/.")


if __name__ == "__main__":
    main()

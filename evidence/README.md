# Evidence — Nguyen Ngoc Bao, 2A202602951

Hai file `04_pii_demo_log.txt` và `04_json_demo_log.txt` được tạo bằng chạy Guardrails tại máy, có kiểm tra output thực tế. Các thông tin PII trong demo đều là dữ liệu giả.

Đã kiểm tra ngày 08/10/2026: Guardrails đạt 7/7 ca PII và 7/7 ca JSON;
`python -m unittest discover -s tests -v` đạt 9/9 bài kiểm thử offline;
`python -m pip check` không phát hiện dependency lỗi.

Đã cấu hình Gemini và LangSmith, có log A/B thật. API xác nhận 50 traces `rag-query` và
50 traces `ab-rag-query` thành công, không có lỗi; A/B phân phối V1=19, V2=31.
Trace RAG mẫu có question, answer và retrieval trong run con; cả 50 traces A/B
đều có 3 contexts. Hai prompt trên Hub đã được pull, đối chiếu khớp với code.
Chi tiết và commit hashes: `01_02_langsmith_api_verification.json`. Xác nhận API
này là evidence bổ sung, không thay ảnh dashboard. RAGAS đang tiếp tục với
evaluator `gemma-4-26b-a4b-it` qua Google API, còn answers vẫn từ
`gemini-3.5-flash-lite`. Lần chấm trước dừng vì Gemini báo hết quota 500
requests/ngày; không dùng các điểm chưa hoàn chỉnh của lần đó làm report.

LangSmith project: [day22-nguyen-ngoc-bao-2a202602951](https://smith.langchain.com/o/b51ae144-8481-4bf3-87f5-0850afd02c81/projects/p/271813f2-cd09-4b5a-8f77-ed4cf1fe6c7f).

Chạy lại phần xác nhận (không gọi Gemini):

```powershell
.\.venv\Scripts\python.exe src/verify_langsmith.py
```

Các evidence LangSmith/Prompt Hub/RAGAS chỉ được bổ sung sau khi cấu hình API và chạy thật; không tạo ảnh dashboard hoặc điểm giả để lấp file thiếu.

## So sánh prompt

V1: tutor trả lời trực tiếp, ngắn gọn. V2: analyst tổ chức câu trả lời theo định nghĩa/kết luận và cơ chế/phân biệt. Cả hai bắt buộc bám context và thừa nhận khi thiếu thông tin.

Chưa có điểm RAGAS thực tế, nên chưa kết luận phiên bản nào tốt hơn hoặc đã đạt faithfulness ≥0.8. Sau khi chạy, đối chiếu bốn chỉ số trong `03_ragas_report.json` cùng câu trả lời và context trong `data/rag_outputs_v1.json`/`rag_outputs_v2.json` để giải thích chênh lệch.

Trong 50 cặp câu trả lời đã chạy, độ dài trung bình theo số từ tách bằng khoảng
trắng là 44,4 ở V1 và 83,8 ở V2. Cả 50 câu hỏi nhận cùng contexts ở hai phiên bản.
Vì vậy, khác biệt điểm trả lời cần được phân tích theo nội dung và prompt;
khác biệt context metrics giữa hai lần chấm còn có thể do evaluator.

Ví dụ QA 19 (chain-of-thought): V1 giải thích định nghĩa và các bước trung gian;
V2 thêm mục cơ chế, tác động và biến thể zero-shot từ context. QA 33 (LangGraph):
V2 thêm phần observability ngoài định nghĩa, directed graph và cycles mà V1
đã nêu. Các chi tiết thêm này giải thích vì sao V2 dài hơn; chúng chỉ có ích
nếu vừa đúng context vừa phục vụ câu hỏi, nên cần đọc cùng faithfulness và
answer relevancy, không dùng độ dài để kết luận chất lượng.

Hướng dẫn chạy và chụp đủ bảy tệp: [HUONG_DAN_CHAY.md](../HUONG_DAN_CHAY.md).

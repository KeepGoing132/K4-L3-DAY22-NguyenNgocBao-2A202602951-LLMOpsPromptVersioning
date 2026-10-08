# Bài lab của Nguyen Ngoc Bao — 2A202602951

Tên repo nộp bài: `K4-L3-DAY22-NguyenNgocBao-2A202602951-LLMOpsPromptVersioning`.

Repo public: [KeepGoing132/Day22](https://github.com/KeepGoing132/K4-L3-DAY22-NguyenNgocBao-2A202602951-LLMOpsPromptVersioning).
Project LangSmith: [day22-nguyen-ngoc-bao-2a202602951](https://smith.langchain.com/o/b51ae144-8481-4bf3-87f5-0850afd02c81/projects/p/271813f2-cd09-4b5a-8f77-ed4cf1fe6c7f).

## Chạy trên Windows PowerShell

Chạy từ thư mục gốc dự án, không cần activate venv:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# Máy hiện tại đã có .env; máy mới sao chép .env.example thành .env rồi điền key.
.\.venv\Scripts\python.exe src/config.py
.\.venv\Scripts\python.exe src/run_all.py
```

Nếu chưa cấu hình API, chạy riêng phần Guardrails:

```powershell
.\.venv\Scripts\python.exe src/run_all.py --step 4
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Kiểm tra các tệp bắt buộc trước khi nộp (không cần gọi API):

```powershell
.\.venv\Scripts\python.exe src/check_submission.py
```

Lệnh này kiểm tra đủ 7 tệp, chữ ký PNG, hai nhãn và 50 câu hỏi trong log A/B,
report có 50 mẫu mỗi phiên bản và đủ 4 metric hữu hạn, faithfulness ≥0.8,
bản sao report khớp nhau, cùng trạng thái Git của `.env`. Vẫn cần tự xác nhận
≥100 traces và hai prompt trên dashboard. Thiếu mục nào sẽ trả exit code 1.
Bước 3 cũng trả exit code 1 nếu đã chấm xong nhưng faithfulness chưa đạt; report
thực tế vẫn được lưu để xem và cải thiện.

`config.py` nạp `.env` trước khi import LangChain. Chọn `PROVIDER` và điền key tương ứng.
Với `openai`, cần `LANGCHAIN_API_KEY` và `OPENAI_API_KEY`.
Với `anthropic`/`openrouter`, vẫn cần `OPENAI_API_KEY` cho embeddings.
Giữ `LANGCHAIN_TRACING_V2=true` khi chạy bài thật.

Nếu dùng Gemini, chỉnh `.env` như sau:

```env
PROVIDER=gemini
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
RAGAS_GEMINI_MODEL=gemma-4-26b-a4b-it
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_TRACING_V2=true
```

Gemini dùng cùng `GOOGLE_API_KEY` cho chat và embeddings, không cần OpenAI key.
Key LangSmith vẫn riêng để ghi traces và push/pull Prompt Hub.
`GEMINI_MODEL` tạo câu trả lời RAG; `RAGAS_GEMINI_MODEL` chỉ làm evaluator.
Lần chạy này dùng Gemma 4 26B A4B để chấm vì Gemini Flash-Lite đã hết quota 500
requests/ngày trước khi chấm xong V1. Hai phiên bản được chấm lại bằng cùng
evaluator; không trộn điểm từ lần chấm dở. Gemma 4 được cung cấp miễn phí qua
[Google API](https://ai.google.dev/gemini-api/docs/pricing#gemma-4).
Đã cập nhật các model Gemini cũ trong cấu hình theo
[danh sách model Google](https://ai.google.dev/gemini-api/docs/models) và
[lịch ngừng model](https://ai.google.dev/gemini-api/docs/deprecations).
Model thực tế dùng được còn phụ thuộc quyền truy cập và quota của key.

Khi Gemini báo lỗi 429 theo phút, chương trình chờ theo thời gian API yêu cầu
rồi thử lại. Embeddings được chia thành batch 40 và lưu trong
`data/embedding_cache.json` (Git ignore); các bước sau tái sử dụng vector thật
của cùng model và cùng dữ liệu. Cache phân biệt vector document với query.
Nếu quota ngày đã hết hoặc quota model bằng 0, chương trình báo lỗi để tránh
chờ vô ích. Không tự bật billing hay nâng cấp gói.

Với Gemini, metric RAGAS `answer_relevancy` dùng `strictness=1` (một câu hỏi
tổng hợp cho mỗi câu trả lời), vì model đang dùng không hỗ trợ nhiều candidates
trong một request. Điều này có thể làm điểm relevancy biến động hơn so với mặc
định 3 câu hỏi; cả V1 và V2 dùng cùng cấu hình. Ba metric còn lại vẫn giữ cách
chấm của RAGAS. Report ghi rõ `answer_relevancy_strictness`.

Nếu phần tạo đủ 100 câu trả lời đã xong nhưng chấm lỗi, có thể chấm lại từ file
đã lưu mà không gọi lại RAG (chỉ dùng khi prompt, model và retrieval chưa đổi):

```powershell
.\.venv\Scripts\python.exe src/03_ragas_evaluation.py --reuse-outputs
```

RAGAS lưu checkpoint thật sau mỗi nhóm 5 QA vào `data/ragas_checkpoint_*.json`.
Chạy lại tiếp tục từ nhóm đã lưu nếu answers, contexts, judge, embedding model
và cấu hình metric khớp fingerprint; thay đầu vào thì chấm lại. Report cuối
chỉ xuất khi đủ 50 QA của cả hai phiên bản và không có NaN. Điểm vẫn là trung
bình của đủ 50 mẫu, không phải trung bình riêng từng nhóm.
Evaluator Google dùng chung rate limiter khoảng 24 requests/phút, tối đa 8
metric đang chờ API để tránh một request chậm giữ cả hàng đợi. Nếu quota của
project thấp hơn, API có thể vẫn báo 429 và SDK thử lại.

## Cách giải thích code

1. **RAG:** `split_text` chia knowledge base thành chunks 500 ký tự, overlap 50. Embeddings biến chunks thành vector, FAISS index và retriever lấy 3 chunks gần câu hỏi nhất. LCEL ghép context vào prompt rồi gọi LLM và parse câu trả lời. `@traceable` tạo run gốc; run con chứa retrieval, prompt và LLM.
2. **Prompt versioning:** `src/prompts.py` là nơi định nghĩa hai prompt. V1 trả lời trực tiếp 2–4 câu; V2 phân tích chứng cứ và tổ chức câu trả lời 3–5 câu. Cả hai chỉ dùng context. Bước 2 push và pull qua LangSmith SDK. `MD5(request_id) % 2` giữ routing ổn định qua lần chạy/process; phân phối không bắt buộc đúng 25/25.
3. **RAGAS:** mỗi câu hỏi được chạy với cả hai prompt. Dataset giữ riêng question, generated answer, list contexts và reference. Reference chỉ được dùng chấm, không đưa vào prompt trả lời. Điểm tổng hợp là trung bình điểm sample; nếu có NaN hoặc thiếu điểm, chương trình báo lỗi.
4. **Guardrails:** validator regex phát hiện PII, trả `FailResult(fix_value=...)`; `OnFailAction.FIX` ở constructor áp dụng bản che. JSON validator gỡ fences, chuyển token nháy đơn, bỏ dấu phẩy thừa ngoài string rồi parse lại; lỗi không sửa được trả JSON chứa `error` và `raw`.

## Bốn chỉ số RAGAS

| Chỉ số | Ý nghĩa |
|---|---|
| Faithfulness | Các khẳng định của câu trả lời có được context hỗ trợ không |
| Answer relevancy | Câu trả lời có liên quan tới câu hỏi không |
| Context recall | Context có đủ thông tin để hỗ trợ đáp án tham chiếu không |
| Context precision | Các chunks liên quan có được ưu tiên trong kết quả retrieval không |

Chỉ kết luận V1/V2 tốt hơn sau khi có điểm thực tế. Câu trả lời dài hơn có thể cung cấp nhiều thông tin liên quan hơn, nhưng cũng có thể thêm khẳng định thiếu chứng cứ. Vì retrieval dùng cùng cấu hình, không nên quy mọi khác biệt context metrics cho phong cách prompt.

## Evidence và nộp bài

- Bước 2 tự ghi `evidence/02_ab_routing_log.txt`.
- Bước 3 lưu `data/ragas_report.json`, bản sao `evidence/03_ragas_report.json`, log terminal và điểm sample trong `data/`.
- Bước 4 tự ghi `evidence/04_pii_demo_log.txt` và `evidence/04_json_demo_log.txt`.
- Sau khi chạy thật, chụp LangSmith project có ≥50 `rag-query` thành `01_langsmith_traces.png`, chụp hai prompt trên Hub thành `02_prompt_hub.png`, chụp bảng RAGAS terminal thành `03_ragas_scores.png` (đặt trong `evidence/`).
- Xác nhận tổng cộng ≥100 traces từ bước 1 và 2 trên dashboard. Thông báo terminal không chứng minh dashboard đã nhận đủ traces.

Nếu chương trình đã chạy xong mà chưa chụp bảng điểm, mở lại log thật để chụp:

```powershell
Get-Content evidence/03_ragas_evaluation_log.txt -Encoding UTF8 -Tail 15
```

- Nộp URL repo public và URL project theo `SUBMISSION.md`. Không push vào repo nguồn của lớp. Không commit `.env`.

Kiểm thử offline dùng embeddings và LLM thay thế để kiểm tra đường đi dữ liệu, không tạo traces hoặc điểm RAGAS làm evidence.

## Tài liệu API đã đối chiếu

- [LangSmith: quản lý prompt bằng SDK](https://docs.langchain.com/langsmith/manage-prompts-programmatically)
- [RAGAS evaluate](https://docs.ragas.io/en/stable/references/evaluate/)
- [Guardrails custom validator source](https://github.com/guardrails-ai/guardrails/blob/main/guardrails/validator_base.py)

`requirements.txt` giữ `langchain-community==0.3.31` (<0.4 theo lab), dùng LangChain Core 1.x vì Guardrails 0.11 yêu cầu Core ≥1.0. Môi trường được cài riêng trong `.venv`.

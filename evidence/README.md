# Evidence — Nguyen Ngoc Bao, 2A202602951

Ngày chạy: 08/10/2026. Các log và report trong thư mục này được tạo từ lần
chạy thật; dữ liệu PII trong demo là giả.

## Kết quả và trạng thái nộp

API xác nhận 50 traces `rag-query` và 50 traces `ab-rag-query` thành công,
không có lỗi. A/B: V1=19, V2=31; cả 50 traces A/B có 3 contexts. Trace RAG mẫu
có question, answer và retrieval trong run con. Hai prompt đã được pull về
đối chiếu khớp code; commit hashes và run IDs nằm trong
[01_02_langsmith_api_verification.json](01_02_langsmith_api_verification.json).

LangSmith project: [day22-nguyen-ngoc-bao-2a202602951](https://smith.langchain.com/o/b51ae144-8481-4bf3-87f5-0850afd02c81/projects/p/271813f2-cd09-4b5a-8f77-ed4cf1fe6c7f).

Guardrails đạt 7/7 ca PII và 7/7 ca JSON. `python -m unittest discover -s tests -v`
đạt 9/9 bài kiểm thử offline; `pip check` không phát hiện dependency lỗi.
Kết quả mock của kiểm thử chỉ ghi vào thư mục tạm, không dùng làm evidence.

Đã lưu nguyên bản ba ảnh chụp thật do người học cung cấp:

- [01_langsmith_traces.png](01_langsmith_traces.png): tên project, bộ lọc
  `name:"rag-query"`, danh sách traces và `Stats · 50 traces`.
- [02_prompt_hub.png](02_prompt_hub.png): tên đầy đủ
  `nguyen-ngoc-bao-2a202602951-rag-prompt-v1` và
  `nguyen-ngoc-bao-2a202602951-rag-prompt-v2`; commit hashes
  `8d838802` (V1) và `bf354ca6` (V2) khớp xác nhận API.
- [03_ragas_scores.png](03_ragas_scores.png): bảng terminal với đủ bốn
  metric V1/V2 khớp report.

Ảnh bổ sung [01_langsmith_all_traces.png](01_langsmith_all_traces.png) giữ
dashboard chưa lọc với 344 traces tổng. Xác nhận API ở trên phân biệt rõ
50 traces RAG và 50 traces A/B trong tổng số này.

Đã đủ bảy tệp evidence và đối chiếu trực quan các ảnh. Kiểm tra tự động
xác nhận tệp, chữ ký PNG, log, report và bảo mật `.env`; không tự đánh giá
nội dung ảnh. JSON xác nhận API là evidence bổ sung, không thay ảnh dashboard.
Chưa nộp lên LMS.

## RAGAS: 50 QA ở hai phiên bản

Answers được tạo bởi `gemini-3.5-flash-lite`; evaluator cuối là
`gemma-4-26b-a4b-it`, `thinking_level=minimal`, qua Google API. Cả hai phiên bản
được chấm đủ 50 QA bằng cùng evaluator. Các lần chấm Gemini trước đó chạm quota
500 requests/ngày; [log lỗi quota](03_gemini31_quota_error_log.txt) được giữ
riêng. Report cuối chỉ lấy điểm của lần chấm Gemma hoàn chỉnh.

| Metric | V1 | V2 | Kết quả |
|---|---:|---:|---|
| Faithfulness | 1.0000 | 0.9801 | V1 cao hơn |
| Answer relevancy | 0.8480 | 0.8453 | V1 nhỉnh hơn |
| Context recall | 1.0000 | 1.0000 | Bằng nhau |
| Context precision | 0.9600 | 0.9600 | Bằng nhau |

Cả hai đạt mục tiêu faithfulness ≥0.8. Đã đối chiếu report evidence với bản trong
`data/` và tính lại trung bình từ đủ 50 điểm hữu hạn cho mỗi metric.
Report: [03_ragas_report.json](03_ragas_report.json).
Log bảng so sánh: [03_ragas_evaluation_log.txt](03_ragas_evaluation_log.txt).

## Phân tích V1 và V2

V1 trả lời trực tiếp, ngắn gọn; V2 tổ chức câu trả lời theo định nghĩa/kết luận,
cơ chế và các phần phân biệt. Cả hai chỉ dùng context. Trong 50 cặp answers,
độ dài trung bình theo số từ tách bằng khoảng trắng là 44,4 ở V1 và 83,8 ở V2.
Cả 50 câu hỏi nhận cùng contexts ở hai phiên bản. Context recall/precision dùng
cùng question, reference và contexts; cache lưu phản hồi judge thật cho các
đầu vào trùng nhau, nên hai chỉ số context bằng nhau.

V1 là lựa chọn mặc định hợp lý cho bộ câu hỏi định nghĩa trong lab: câu trả lời
ngắn hơn, faithfulness cao hơn và relevancy nhỉnh hơn. Chênh lệch relevancy
khoảng 0,0027 còn nhỏ; chưa có đánh giá lặp lại hoặc kiểm định để kết luận khác
biệt này có ý nghĩa thống kê. V2 hữu ích khi cần trình bày cơ chế và nhiều phần
rõ ràng, nhưng trong bộ này nội dung dài hơn chưa giúp tăng điểm trung bình.

Ví dụ QA19 (chain-of-thought): V1 nêu định nghĩa và các bước trung gian; V2 thêm
cơ chế, tác động và biến thể zero-shot từ context. QA33 (LangGraph): V2 thêm
observability ngoài directed graph và cycles mà V1 đã nêu. Đây là các ví dụ
cho sự khác biệt phong cách, không phải bằng chứng câu trả lời dài luôn tốt hơn.

QA20 có faithfulness V2 bằng 2/3: judge chấp nhận hai khẳng định về thông tin
trong dataset nhưng đánh dấu câu kết “context does not establish any further
specifications” là suy luận về thông tin vắng mặt. Chi tiết verdict thật:
[03_qa20_faithfulness_detail.json](03_qa20_faithfulness_detail.json).
Với lần cải tiến tiếp theo, V2 nên chỉ thêm câu nói thiếu thông tin khi thực sự
không trả lời được câu hỏi. Đây cũng là ví dụ cần đọc verdict và context cùng
điểm số thay vì coi một metric là kết luận tuyệt đối.

Các điểm phụ thuộc vào judge và 50 QA của lab. `answer_relevancy` dùng
`strictness=1` do giới hạn candidate của Google API; một câu hỏi tổng hợp có
thể làm điểm biến động hơn mặc định 3. Reference chỉ được dùng trong đánh giá,
không đưa vào prompt RAG để tạo answer. Chưa kiểm tra một tập câu hỏi độc lập.

Hướng dẫn chạy và chụp ảnh: [HUONG_DAN_CHAY.md](../HUONG_DAN_CHAY.md).

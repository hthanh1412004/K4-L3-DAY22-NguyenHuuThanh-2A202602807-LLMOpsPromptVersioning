# Bằng chứng bài lab — Nguyễn Hữu Thành, 2A202602807

## Trạng thái thực tế

- Hoàn tất bước 1: LangSmith xác nhận 50 traces `rag-query`, có retriever với 3 documents.
- Hoàn tất bước 2: 2 prompt push/pull trên Hub; 50 traces A/B, V1=19 và V2=31.
- Hoàn tất bước 4: 6 case PII và 5 case JSON qua Guard thật; log đã lưu.
- Hoàn tất bước 3: chấm đủ 50 câu V1 và 50 câu V2 bằng Gemma.
  Báo cáo có đủ 4 metrics; faithfulness V1=1.0000, V2=0.9885, cả hai đạt ≥ 0.8.
  Đã đối chiếu trung bình trong report với đủ 50 dòng checkpoint mỗi phiên bản;
  bản report trong `data/` khớp bản trong `evidence/`.
- 12 bài kiểm tra offline đã qua, gồm kiểm tra dừng trước batch khi hết quota ngày
  và tiếp tục chấm từ checkpoint sau timeout;
  `pip check` và kiểm tra cú pháp trước đó đã qua.
- Hiện có đủ 7/7 bằng chứng bắt buộc. `03_ragas_scores.png` hiển thị bảng điểm
  V1/V2 và dòng đạt mục tiêu, đã đối chiếu với report.
  Hai ảnh traces/Prompt Hub được sao chép từ ảnh gốc người dùng sang tên bắt buộc,
  không chỉnh nội dung ảnh.
- `.env` không được Git theo dõi; quét code/log/JSON không thấy API key thật.

Hai checkpoint câu trả lời và các điểm đã chấm được giữ trong `.cache/`.
Nếu dùng key provider khác để chấm, giữ `PROVIDER=gemini`, thêm `EVAL_PROVIDER`
(ví dụ `openai`) và key tương ứng vào `.env`; không cần gửi key trong chat.
Bước 3 kiểm tra evaluator trước khi chấm hàng loạt và lưu lý do lỗi vào log.
Key Gemini mới trong cùng project không tăng quota của project.
Điểm chấm được lưu sau từng câu vào `.cache/ragas_samples_v1.json` và
`.cache/ragas_samples_v2.json`. Nếu timeout hoặc mất mạng, chạy lại cùng lệnh
để bỏ qua các câu đã lưu với cùng dữ liệu và cấu hình evaluator.
RAGAS chấm metrics tuần tự với Gemma, timeout 600 giây; lỗi timeout được thử lại một lần.

## Cách tái hiện

Chạy từng đợt để đổi key giữa các lượt:

```powershell
.\.venv\Scripts\python.exe src/run_all.py --step 3 --eval-limit 5
```

Mỗi lượt chấm tối đa 5 câu mới cho mỗi phiên bản rồi lưu checkpoint và dừng
với trạng thái `ĐÃ LƯU TIẾN ĐỘ`. Đổi `GOOGLE_API_KEY` trong `.env` rồi chạy
lại cùng lệnh; tiến trình mới đọc key mới. Giữ nguyên provider/model evaluator
để dùng lại điểm; API key không tham gia fingerprint checkpoint. Nếu hết quota
theo ngày, key khác phải thuộc project còn quota; key khác cùng project không
tăng hạn mức. Không gửi key vào chat. Khóa `.cache/ragas.lock` ngăn chạy hai
lượt cùng lúc; khóa được giải phóng khi tiến trình dừng.
Chỉ khi đủ 50 câu cho cả V1/V2 mới tạo báo cáo để chụp ảnh kết quả.

Từ thư mục gốc, dùng PowerShell:

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python.exe src\run_all.py
```

Runner lưu log thực tế của từng bước. Hai log Guardrails được tách từ cùng phiên
demo: `04_pii_demo_log.txt` chứa 6 trường hợp PII, `04_json_demo_log.txt` chứa
5 trường hợp JSON. Nội dung kết quả được giữ nguyên khi tách. Chạy lại bước 4
sẽ ghi toàn bộ phiên demo vào cả hai file theo cách của runner.
Bước 3 lưu `data/ragas_report.json` và bản sao `evidence/03_ragas_report.json`.
Điểm chỉ được ghi khi đánh giá thật hoàn tất; không dùng điểm mẫu trong tài liệu.

## Prompt và cách giải thích code

- V1: câu trả lời trực tiếp, ngắn gọn 2–4 câu, chỉ dựa trên context.
- V2: phân tích có kết luận và các ý hỗ trợ, 3–5 câu, mỗi nhận định phải có căn cứ.
- Hai bước A/B và RAGAS cùng import `SYSTEM_V1`/`SYSTEM_V2` từ một file để tránh sai lệch.
- `request_id` được băm bằng MD5; phần dư khi chia 2 chọn prompt. Kết quả không đổi
  giữa các lần chạy và không phụ thuộc seed ngẫu nhiên của Python.
- LCEL lấy 3 documents gần nhất, ghép nội dung cho prompt, gọi LLM, rồi lấy chuỗi
  qua `StrOutputParser`. Run retriever con ghi các documents vào cây trace.
- RAGAS nhận từng context riêng dưới dạng `list[str]`, câu trả lời và đáp án chuẩn.
  Faithfulness đo mức bám context; relevancy đo đúng trọng tâm; recall đo độ đầy đủ
  của context so với reference; precision đo mức liên quan của các đoạn truy xuất.
- Guardrails: dữ liệu hợp lệ trả `PassResult()`. Dữ liệu cần sửa trả
  `FailResult(fix_value=...)`; `OnFailAction.FIX` thay output bằng giá trị này.
  JSON không sửa được được thay bằng JSON có trường `error` và đoạn `raw`.
- Gemini embeddings được cache trong `.cache/` và điều tiết dưới quota 100
  requests/phút. Cache phân biệt embedding của tài liệu và của câu hỏi.
- Evaluator có thể chọn provider riêng bằng `EVAL_PROVIDER` hoặc model Gemini riêng
  bằng `GEMINI_EVAL_MODEL`. Cấu hình hiện tại dùng
  Gemma `gemma-4-26b-a4b-it` (reasoning `minimal`, giới hạn thời gian mỗi API call 90 giây);
  câu trả lời RAG đã tạo dùng Gemini 3.5 Flash Lite. Gemma không bị ép JSON mode;
  adapter kiểm tra JSON trả về và gọi lại nếu JSON lỗi. Tốc độ đầu vào được điều tiết
  theo ngân sách 12.000 token/phút ước lượng, không phải cam kết quota của project.
  Với Gemma, các metrics và các candidates được gọi tuần tự để giảm burst token.
  Lỗi máy chủ 500/502/503/504 được thử lại tối đa 3 lần, chờ 5/10/20 giây;
  không tự áp dụng retry này cho lỗi quota 429 hoặc lỗi quyền 403.
  JSON lỗi được gọi lại, không sửa điểm. Khi RAGAS cần 3 candidates, wrapper
  gửi các request riêng để tránh lỗi Gemini không hỗ trợ nhiều candidates và
  tránh thay đổi tham số `n` chung giữa các metric chạy đồng thời.

## Ảnh bằng chứng

Project đã xác minh:
[day22-lab](https://smith.langchain.com/o/dd87feed-c199-4862-8525-63720b6d2eb8/projects/p/dcd97460-4584-49d0-9311-01bd88606b2f).

Ảnh phải chụp từ kết quả thật, không dựng lại dashboard:

1. `01_langsmith_traces.png`: project LangSmith có ít nhất 50 `rag-query`;
   mở một trace kiểm tra question, retriever contexts và answer.
2. `02_prompt_hub.png`: hai prompt
   `nguyen-huu-thanh-2a202602807-rag-prompt-v1` và `...-v2` trên Hub.
3. `03_ragas_scores.png`: bảng so sánh 4 metrics của phiên chạy thực tế.

## Phân tích V1/V2

Điểm trung bình trên 50 câu mỗi phiên bản, cùng evaluator `gemma-4-26b-a4b-it`:

| Metric | V1 | V2 |
| --- | ---: | ---: |
| faithfulness | 1.0000 | 0.9885 |
| answer_relevancy | 0.8394 | 0.8082 |
| context_recall | 1.0000 | 1.0000 |
| context_precision | 0.9600 | 0.9600 |

V1 cao hơn V2 0.0115 về faithfulness và khoảng 0.0312 về answer relevancy.
V1 yêu cầu trả lời trực tiếp trong 2–4 câu; V2 yêu cầu kết luận và các ý hỗ trợ
trong 3–5 câu. Một cách giải thích là câu trả lời ngắn của V1 giữ trọng tâm và
đưa ra ít nhận định hơn. Đây là suy luận từ thiết kế prompt và điểm đo, chưa phải
kết luận nhân quả hay kiểm định khác biệt có ý nghĩa thống kê.

Đã đối chiếu dữ liệu: cả 50 câu đều có cùng retrieved contexts giữa V1/V2.
Context recall và precision bằng nhau trong lần chấm này phù hợp với việc
retrieval được giữ nguyên. Cải thiện retrieval cần điều chỉnh chunking, embeddings
hoặc k. Cả hai phiên bản đạt ngưỡng faithfulness của lab. Kết quả áp dụng cho bộ
QA và evaluator ghi trong báo cáo; thứ hạng có thể khác với bộ dữ liệu/model khác.

## Tài liệu API đối chiếu

- [RAGAS evaluate 0.4](https://docs.ragas.io/en/v0.4.0/references/evaluate/)
- [LangSmith pull_prompt](https://reference.langchain.com/python/langsmith/client/Client/pull_prompt)

Không ghi key vào log hoặc commit `.env`. Nộp URL repo và URL project qua LMS
sau khi kiểm tra đủ 7 file bắt buộc theo `SUBMISSION.md`.

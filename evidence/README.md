# Kết quả lab — Nguyễn Hữu Thành, 2A202602807

## Kết quả chạy

- RAG: 50 traces `rag-query`, mỗi lần truy xuất 3 đoạn tài liệu.
- A/B: 50 câu hỏi, chia thành V1=19 và V2=31 bằng MD5 của `request_id`.
  Cả hai prompt đã được push lên Hub và pull về khi chạy.
- RAGAS: chấm đủ 50 câu cho mỗi phiên bản, có đủ 4 chỉ số trong
  `03_ragas_report.json`.
- Guardrails: 6 case PII và 5 case JSON. Hai log `04` chứa từng phần demo tương ứng.
- Có đủ 7 file evidence bắt buộc và 12 bài kiểm tra offline đã qua.

Project LangSmith:
[day22-lab](https://smith.langchain.com/o/dd87feed-c199-4862-8525-63720b6d2eb8/projects/p/dcd97460-4584-49d0-9311-01bd88606b2f).

Ba ảnh gồm danh sách traces (`01_langsmith_traces.png`), hai prompt trên Hub
(`02_prompt_hub.png`) và bảng điểm cuối (`03_ragas_scores.png`).
Ảnh `lang-smith.png` bổ sung số lượng traces của project.

## So sánh V1 và V2

Model trả lời là `gemini-3.5-flash-lite`; model chấm là `gemma-4-26b-a4b-it`.
Điểm dưới đây là trung bình của 50 câu mỗi phiên bản.

| Chỉ số | V1 | V2 |
| --- | ---: | ---: |
| faithfulness | 1.0000 | 0.9885 |
| answer_relevancy | 0.8394 | 0.8082 |
| context_recall | 1.0000 | 1.0000 |
| context_precision | 0.9600 | 0.9600 |

V1 cao hơn một chút về faithfulness và answer relevancy. Prompt V1 yêu cầu trả
lời ngắn, trực tiếp trong 2–4 câu; V2 yêu cầu kết luận kèm các ý hỗ trợ trong
3–5 câu. Có thể việc trả lời ngắn giúp V1 giữ trọng tâm tốt hơn. Kết quả này
chỉ đo trên 50 câu hỏi của lab, với model chấm nêu trên.

Hai phiên bản dùng cùng context ở cả 50 câu, nên recall và precision bằng nhau
trong lần chạy này là hợp lý. Cả V1 và V2 đều đạt ngưỡng faithfulness ≥ 0.8.

## Các lỗi gặp khi chạy

Gemini bị hết quota trong phần chấm RAGAS, nên evaluator được đổi sang Gemma.
Có thêm lỗi timeout và lỗi máy chủ 500. Code lưu điểm sau từng câu để chạy
tiếp từ câu chưa xong, gọi Gemma tuần tự và thử lại các lỗi máy chủ tạm thời.

Hai bước A/B và RAGAS dùng chung prompt trong `src/prompt_templates.py`.
RAGAS nhận context dạng `list[str]`. Với Guardrails, phần sửa đầu ra dùng
`FailResult(fix_value=...)` và `OnFailAction.FIX`.

## Cách chạy lại

Từ thư mục gốc, chạy trong PowerShell:

```powershell
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe src/run_all.py
```

Để chấm từng đợt tối đa 5 câu mới cho mỗi phiên bản:

```powershell
.\.venv\Scripts\python.exe src/run_all.py --step 3 --eval-limit 5
```

Nếu cần đổi key, sửa `GOOGLE_API_KEY` trong `.env` sau khi lượt chạy dừng,
rồi chạy lại cùng lệnh. Giữ nguyên model để dùng tiếp điểm đã lưu trong
`.cache/`. Key khác cùng project không tăng quota của project.

Chạy lại bước 4 sẽ ghi cả phiên demo vào hai file log `04`. Hai log trong
bản nộp đã được tách thành phần PII và phần JSON.

Không commit `.env`. Các file cần nộp được liệt kê trong `SUBMISSION.md`.

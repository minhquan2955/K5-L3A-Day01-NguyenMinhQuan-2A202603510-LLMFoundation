# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV: 2A202603510**
- **Họ và tên: Nguyễn Minh Quân**
- **SEED: 1065** (in ở ô đầu notebook):

## 1. Bằng chứng

| Tệp                                                                  | Có? |
| -------------------------------------------------------------------- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output  | v   |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | v   |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước          | v   |

## 2. Kết quả chính

_Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn._
                                  F1     F1 SUB 40
1. BoW + LogReg (tự code)        0.868   0.750
2. BoW + MLP (deep learning)     0.944   0.889
3a. Encoder (MiniLM) + kNN       0.906   0.974
3b. Encoder (MiniLM) + LogReg    0.966   0.974
3c. Encoder (MiniLM) + MLP head  0.950   0.974
4a. LLM nhỏ zero-shot            0.000   0.000
4b. LLM lớn zero-shot            0.000   0.000
4c. LLM nhỏ few-shot             0.000   0.000

## 3. Dự đoán và thực tế

_Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch._
predict("0.9", "Bỏ ReLU khiến hai lớp Linear liên tiếp tương đương một lớp Linear; khi dùng sigmoid để dự đoán, mô hình tương đương logistic regression. Dự đoán F1 gần logistic regression và có thể giảm nhẹ so với MLP có ReLU vì mất khả năng học quan hệ phi tuyến.")
Do sai lệch khi tính toán

## 4. Phân tích 50 case khó

_Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai._
Model BoW + LogReg tụt nhiều nhất, ở nhóm case Spam giọng tự nhiên
2 case cụ thể: {"id": 12, "category": "natural_spam", "label": "spam", "text": "hi there, sorry to bother you, my friend gave me your number, I'm new in town and looking for someone to chat with, check my profile at meetlocal.club", "why_tricky": "Mở đầu lịch sự, không từ khoá — dụ vào trang hẹn hò"}, {"id": 13, "category": "natural_spam", "label": "spam", "text": "Hope you're well! Quick reminder that your photos from the event are ready, view & order prints here: snap-prints.co/r/88", "why_tricky": "Giả làm nhắc nhở thân thiện, link bán hàng không yêu cầu"}

## 5. Thử thách cá nhân

_Đề theo SEED của bạn, cách làm, kết quả, giải thích._
Đề #4: Embedding · Chạy knn_predict với k = 1, 5, 25. Báo F1. Vì sao k quá nhỏ hoặc quá lớn đều có hại?
kNN tìm k tin nhắn có embedding gần nhất với tin nhắn cần phân loại, sau đó dự đoán bằng cách bỏ phiếu theo nhãn của chúng.

k = 1: Chỉ dựa vào một hàng xóm. Nếu tin nhắn gần nhất bị gán nhãn sai hoặc là trường hợp đặc biệt, dự đoán dễ sai theo. Mô hình nhạy với nhiễu, có phương sai cao, dễ overfit.
k = 5: Dựa trên nhiều hàng xóm nên một điểm nhiễu ít có khả năng quyết định toàn bộ kết quả. Đây có thể là mức cân bằng tốt, nhưng cần kiểm chứng bằng F1 thực tế.
k = 25: Có thể lấy thêm những tin nhắn ít tương đồng vào cuộc bỏ phiếu, làm mất đặc điểm của nhóm gần nhất. Mô hình có độ chệch cao, dễ underfit. Nếu ham chiếm đa số, nhiều hàng xóm ham có thể lấn át spam, khiến bỏ sót spam và giảm recall.

## 6. Phương án cho sản phẩm thật

_Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì._
chọn BoW + Logistic Regression vì mô hình nhẹ, suy luận nhanh và chi phí thấp. 10 triệu SMS/ngày ≈ 116 tin/giây; cần dẫn F1, p95 và throughput đo thực tế để chứng minh đáp ứng p95 < 50 ms ở tải yêu cầu—hiện chưa có bảng benchmark để khẳng định. Đánh đổi là khả năng nhận diện spam né từ khóa hoặc diễn đạt mới kém hơn; cần ưu tiên hạn chế chặn nhầm SMS quan trọng như OTP.

## 7. AI log và điều đã học

_Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu._
| Hỏi AI gì | Nhận được gì | Tôi sửa / kiểm chứng gì |
|---|---|---|
| Viết code kNN với k = 1, 5, 25 và tính F1. | Code dự đoán và giải thích ảnh hưởng của k. | Chạy thử, phát hiện thiếu hàm `knn_predict` và nhờ bổ sung. |
| Hỗ trợ các phần lập trình. | Code mẫu, giải thích và hướng sửa lỗi. | Tự kiểm tra lại bằng tài liệu chính thống của thư viện. |
| Chọn mô hình đáp ứng yêu cầu độ trễ và lưu lượng. | Đề xuất BoW + LogReg, router và cách tính chi phí LLM. | Phân biệt số liệu giả định với benchmark thực tế. |
| Tạo 5 test case khó. | Các SMS né từ khóa, dễ chặn nhầm và chứa prompt injection. | Rà soát nội dung, nhãn kỳ vọng; cần chạy để xác định lỗi thực tế. |
| Giải thích kết quả stress test. | Phân tích điểm yếu của từng phương pháp. | Giữ các nhận xét chưa có số liệu ở mức dự kiến. |

Hôm nay em hiểu rằng điểm cao trên tập test thường chưa bảo đảm mô hình xử lý tốt những tình huống khó ngoài thực tế. Em cũng nhận ra lựa chọn mô hình cần cân bằng chất lượng, độ trễ, chi phí và hậu quả của lỗi; code do AI hỗ trợ vẫn cần được tự kiểm chứng.

# Báo cáo tiến độ Project SO-101 VLA

Ngày cập nhật: 08/10/2026. Mọi số liệu dưới đây lấy từ report, log và checkpoint trong repo (nguồn ghi trong ngoặc).
Trạng thái dùng: PASS (hoàn thành, có kiểm chứng) · RUNNING/PAUSED (đang làm, tạm dừng) · PENDING (chưa làm) · NOT ACHIEVED (đã làm nhưng chưa đạt) · BLOCKED.

## 1. Mục tiêu

Xây dựng một policy Vision-Language-Action (VLA) cho cánh tay SO-101 (single arm) thực hiện 11 task thao tác:
- đầu vào: câu lệnh ngôn ngữ + 2 camera (`top`, `wrist`) + trạng thái khớp;
- model chính: Evo-1, dùng bản tích hợp chính thức trong LeRobot 0.6.1;
- kiểm chứng trong mô phỏng MuJoCo, kể cả ở vị trí vật chưa xuất hiện khi train;
- tối ưu bằng sensitivity/CKA, CLP và lượng tử hoá (W8A8 → W4A8 → W4A4) để chạy hoàn toàn local trên RTX 3050 6 GB / RAM 12 GB;
- chuẩn bị deploy an toàn lên robot thật.

## 2. Kế hoạch thực hiện

Dataset (11 task, Hugging Face) → kiểm tra / chia split → mô phỏng MuJoCo SO-101 → baseline ACT → Evo-1 Stage 1 →
REAL_ONLY vs REAL_PLUS_SIM → Stage 2 Lite → sensitivity/CKA → CLP → W8A8 / W4A8 / W4A4 → chọn model tối ưu (BEST_LOCAL)
→ đánh giá đủ 11 task (gồm vị trí chưa thấy) → video mô phỏng → dry-run và deploy SO-101 thật.

## 3. Những nội dung đã hoàn thành

**A. Dataset**: PASS (`reports/dataset_audit.md`, `validation.md`, `split_manifest.json`, `aggregate_verification.md`)
- 11 dataset, **551 episode, 253.745 frame**, LeRobot v3.0, 30 fps, `robot_type = so_follower`. Quy ước giống nhau ở cả 11 dataset.
- State / action: 6 chiều (5 khớp tính bằng độ + gripper 0–100). Action là **vị trí khớp tuyệt đối**; state trễ theo action khoảng 4 frame.
- Camera: `top` 640×360, `wrist` 640×480 (AV1). Đã kiểm tra 46 file video, decode 2.204 frame đầu/cuối episode, 0 lỗi.
- Loại **9 episode không có chuyển động** (biên độ action < 1° ở mọi khớp). Đây là episode không có thao tác, không phải dữ liệu hỏng; dữ liệu gốc giữ nguyên.
- Chia theo episode, mỗi task 5 val + 5 test: **train 432 / val 55 / test 55**.
- Validate bằng loader chính thức 112/112 PASS. Gộp 11 dataset bằng API chính thức, kiểm tra 14/14 PASS (frame giống từng pixel với nguồn).
- Stats chuẩn hoá tính **chỉ trên 432 episode train** (200.257 frame).

**B. Môi trường** (`reports/system_check.md`)
- Ubuntu 26.04, Python 3.12, PyTorch 2.11 + CUDA 13.0, LeRobot 0.6.1.
- RTX 3050 Laptop 6 GB (compute 8.6), hỗ trợ BF16. RAM hệ điều hành thấy được: 10,9 GB.
- Isaac Sim 5.1 không đạt cấu hình tối thiểu (32 GB RAM / 16 GB VRAM) nên **chọn MuJoCo**.

**C. Mô phỏng**: PASS cho môi trường (`reports/oracle/summary.md`, `sim_reset_check.json`, `sim_dataset_report.md`)
- Dùng model SO-101 chính thức (MJCF của TheRobotStudio/SO-ARM100). Quy đổi action từ độ sang radian: đã test round-trip.
- Đủ 11 task với đúng câu lệnh của dataset; vật thể dựng bằng hình khối gần đúng (`simulation_asset_report.md`).
- Vị trí vật chia vùng train / val / test, nên vị trí test không bao giờ xuất hiện khi train. Kiểm tra 1.012 lần reset, 0 lỗi.
- Controller oracle (có biết vị trí vật) chứng minh các task giải được:
  - bố cục cố định: 11/11;
  - vị trí ngẫu nhiên: 194/210;
  - **vị trí chưa thấy: 65/70**;
  - yếu nhất là task_04 (10/15) và task_11 (11/15).
- Dataset mô phỏng: **220 episode / 62.998 frame** (20 mỗi task, chỉ dùng vùng train). Dataset trộn REAL_PLUS_SIM: 652 episode train.

**D. Baseline ACT (chỉ task_01)** (`reports/experiment_20261007_172212.json`, `offline_act_task01_10k_h50_*.json`, `evaluation_act_task01_10k_position.md`)
- Checkpoint hoàn chỉnh ở bước 10.000 (batch 8). Peak VRAM 4,2 GB, RAM 2,7 GB.
- Offline trên episode test thật (horizon 50): MAE **8,9°** (val 7,5°), so với 11,1° nếu robot đứng yên. Gripper 6,8; cosine 0,77.
- Trong mô phỏng: **0/20**. Robot nâng tay rồi lơ lửng, không gắp. Nguyên nhân là ảnh mô phỏng khác xa video thật.

**E. Evo-1 Stage 1** (VLM InternVL3-1B đóng băng) (`checkpoints/*/training_summary.json`, `offline_evo1_*.json`, `evo1_sweep.md`)
- **Model:** 775,2M tham số, 122,0M được train. 2 camera ở **448 px**: bản chính thức không cho phép 256 px. Chunk 50 action, BF16 autocast.
- **Lần 1, train đầy đủ:** 3.000 lần cập nhật × batch 8, val loss 0,338. Sai số offline 44,3°, **kém hơn robot đứng yên ở cả 11 task** (train quá ít).
- **Lần 2, train trên đầu ra VLM đã lưu sẵn:**
  - Ở stage 1 VLM đóng băng, nên đầu ra VLM của mỗi frame có thể tính một lần rồi lưu đĩa. Đã kiểm tra: loss chênh ≤ 0,5% so với đường chính thức.
  - Tốc độ train tăng từ 1,8 lên ~74 mẫu/s.
  - 20.000 lần cập nhật × batch 32 = 640.000 mẫu, val loss **0,035**.
- **Lần 3, với gấp đôi số frame:** đã lưu đủ 40.220 frame (39 GB). Đang **tạm dừng ở bước 1.000/20.000**; có checkpoint đầy đủ để resume.
- **Bộ nhớ:** peak VRAM ≤ 4,5 GB, RAM ≤ 4,5 GB (gồm cả bước lưu đầu ra VLM). Suy luận ~0,8 s cho mỗi chunk 50 action.

**F. Kỹ thuật**
- Resume chính xác: chạy liền và chạy dừng-rồi-resume cho loss trùng khớp. Checkpoint load lại được bằng API LeRobot.
- **54/54 unit test PASS** (chạy lại ngày 08/10).
- Lớp an toàn cho robot thật: giới hạn khớp theo calibration, giới hạn tốc độ, không gian làm việc (tính bằng mô hình SO-101), dừng khẩn cấp, timeout. Dry-run chế độ replay (không cần phần cứng) đã chạy.
- **Lỗi đã tìm và sửa:**
  - gripper mô phỏng không kẹp được vật (thêm đệm ngàm), vật nảy văng khỏi nồi;
  - metadata sai sau khi gộp dataset;
  - dataset mô phỏng ghi dở khi tạm dừng (đã sửa, không phải sinh lại);
  - lệch RNG làm resume không chính xác.

## 4. Kết quả hiện tại

| Hạng mục | Kết quả hiện tại | Trạng thái |
|---|---|---|
| Dataset | 11 task, 551 episode, split 432/55/55, 9 episode loại, validate 112/112 | PASS |
| Simulation | MuJoCo SO-101, 11 task, oracle ở vị trí chưa thấy 65/70 | PASS |
| ACT (task_01) | offline 8,9° (đứng yên 11,1°); mô phỏng 0/20 | PASS (offline) / NOT ACHIEVED (mô phỏng) |
| Evo-1 Stage 1 | val loss 0,035; offline 15,0° (đứng yên ~12,0°); vượt baseline 1/11 task | NOT ACHIEVED |
| REAL_ONLY | là dữ liệu đang dùng cho Evo-1 Stage 1 ở trên | RUNNING (PAUSED) |
| REAL_PLUS_SIM | đã dựng dataset trộn (652 episode); chưa train | PENDING |
| CKA / Sensitivity | chưa triển khai | PENDING |
| CLP | chưa triển khai | PENDING |
| Stage 2 Lite | chưa triển khai | PENDING |
| W8A8 | chưa triển khai | PENDING |
| W4A8 | chưa triển khai | PENDING |
| W4A4 / FoldQuantVLA | repo chính thức không hỗ trợ Evo-1; kernel INT4 nhắm sm_87/sm_89, máy này là sm_86 | PENDING (native trên RTX 3050 chưa xác minh) |
| Đánh giá đủ 11 task trong mô phỏng (Evo-1) | Chưa đánh giá | PENDING |
| Video cuối | mới có video oracle (task_01, task_08) và ACT task_01 | PENDING |
| Deploy SO-101 thật | script + lớp an toàn + dry-run replay; chưa thử với phần cứng | PENDING |

**Chất lượng Evo-1 hiện tại** (đánh giá trên episode test thật, chưa thấy khi train):
- Val loss giảm mạnh (0,327 → 0,035), nhưng **loss thấp chưa có nghĩa là policy tốt**.
- **11 task**, 4 frame mỗi episode: MAE trung bình 15,0°. Robot đứng yên đạt ~12,0°. Evo-1 chỉ tốt hơn ở task_02 (13,0° so với 13,5°). Cosine hướng chuyển động 0,23–0,56.
- **Task_01**, cùng giao thức với ACT (8 frame, horizon 50):
  - Evo-1: 10,5° (val 11,1°), cosine 0,65;
  - ACT: 8,9° (val 7,5°), cosine 0,77;
  - đứng yên: 11,1°.
- **Tỉ lệ thành công trong mô phỏng của Evo-1: chưa đánh giá.**

## 5. Vấn đề và hạn chế

1. **Evo-1 chưa đạt chất lượng policy.** Sau Stage 1, sai số action vẫn cao hơn baseline đứng yên ở 10/11 task và kém ACT train riêng cho task_01. Nguyên nhân chính là train chưa đủ trên GPU 6 GB:
   - VLM đóng băng;
   - mới dùng 1/10 rồi 2/10 số frame;
   - lần 3 đang dừng ở bước 1.000.
2. **Lệch miền giữa video thật và mô phỏng.** ACT tốt khi đánh giá offline nhưng 0/20 trong mô phỏng. Mô phỏng dùng hình khối; robot mô phỏng màu vàng còn robot thật màu xanh cyan; camera top chưa được hiệu chỉnh. Vì vậy tỉ lệ thành công trong mô phỏng của policy train bằng dữ liệu thật hiện chưa phản ánh đúng chất lượng.
3. **Chưa làm phần tối ưu:** CKA, CLP, Stage 2 Lite và toàn bộ lượng tử hoá.
4. **W4A4 native trên RTX 3050 chưa chắc khả thi.** FoldQuantVLA chỉ hỗ trợ GR00T và π₀.₅; kernel INT4 nhắm sm_87/sm_89. Cần tự điều chỉnh cho Evo-1 và build thử cho sm_86. Nếu không build được thì dùng fake quant + W8A8/W4A8.
5. **Suy luận Evo-1 chậm**: ~0,8 s mỗi chunk (đủ dùng cho 50 action ở 30 Hz, nhưng cần nới timeout khi deploy). Ảnh bắt buộc 448 px (256 px không được phép).
6. **Chưa có đánh giá cuối:** chưa có kết quả mô phỏng đủ 11 task, video cuối, hay thử trên robot thật.

## 6. Kế hoạch tiếp theo

| # | Việc | Mục đích | Tiêu chí PASS |
|---|---|---|---|
| 1 | Resume lần 3 của Stage 1 (bước 1.000 → 20.000), đánh giá offline đủ 11 task | biết giới hạn của Stage 1 | sai số offline thấp hơn baseline đứng yên ở ≥ 6/11 task |
| 2 | Train REAL_PLUS_SIM, so sánh với REAL_ONLY trên cùng giao thức | xem dữ liệu mô phỏng có giúp không | có bảng so sánh offline + mô phỏng cho cả hai |
| 3 | Sensitivity / CKA trên tập mẫu phủ 11 task (đầu ra lưu đĩa) | tìm lớp dư thừa, lớp nhạy cảm | có báo cáo theo từng lớp / module |
| 4 | CLP (tỉa từng lớp rồi đánh giá) | giảm kích thước, tăng tốc | mất ≤ 1° MAE và không giảm tỉ lệ thành công |
| 5 | Stage 2 Lite (LoRA / fine-tune chọn lọc theo sensitivity) | cải thiện chất lượng trong giới hạn 6 GB | vượt Stage 1 về MAE; không OOM |
| 6 | W8A8 | lượng tử hoá an toàn đầu tiên | MAE tăng ≤ 0,5° so với BF16; đo VRAM / độ trễ |
| 7 | W4A8 | giảm bộ nhớ thêm | MAE tăng ≤ 1° |
| 8 | W4A4 (theo hướng FoldQuant, fake quant trước) | thử giới hạn nén | báo PASS / FAIL / BLOCKED trung thực, kể cả native trên sm_86 |
| 9 | Chọn BEST_LOCAL | model cuối | cân bằng tỉ lệ thành công, MAE, VRAM, RAM, độ trễ, kích thước |
| 10 | Đánh giá đủ 11 task trong mô phỏng (vị trí đã thấy / chưa thấy) | kết quả cuối | có tỉ lệ thành công từng task, gồm vị trí chưa thấy |
| 11 | Ghi video mô phỏng | minh hoạ | video cho cả 11 task |
| 12 | Dry-run trên SO-101 thật rồi chạy tốc độ thấp | deploy | dry-run PASS, calibration hợp lệ, lớp an toàn hoạt động |

## 7. Tiêu chí hoàn thành

Project chỉ được coi là hoàn thành khi đạt đủ các mục sau. Hiện tại:

| Tiêu chí | Hiện tại |
|---|---|
| Đủ 11 task | ĐẠT (dataset + mô phỏng) |
| Pipeline dataset hoàn chỉnh | ĐẠT |
| MuJoCo SO-101 hoạt động | ĐẠT |
| Policy Evo-1 hoạt động (vượt baseline) | CHƯA |
| Có model cuối (BEST_LOCAL) | CHƯA |
| Benchmark đủ 11 task, có metric từng task | CHƯA (mới có offline) |
| Test vị trí chưa thấy (policy) | CHƯA (mới có oracle) |
| Video mô phỏng | CHƯA |
| Chạy local trên RTX 3050 6 GB / RAM 12 GB | ĐẠT cho train / suy luận Stage 1 (VRAM ≤ 4,5 GB); model cuối chưa có |
| Checkpoint / export | ĐẠT cho Stage 1 (định dạng LeRobot) |
| Đường deploy an toàn lên SO-101 | MỘT PHẦN (script + dry-run replay; chưa thử với phần cứng) |

## 8. Kết luận

Phần nền tảng của project đã xong và có kiểm chứng: dữ liệu 11 task (551 episode, chia theo episode), môi trường mô phỏng MuJoCo của SO-101 với 11 task và vùng vị trí chưa thấy, baseline ACT, và toàn bộ pipeline train / đánh giá / deploy chạy được trên RTX 3050 6 GB. Kết quả đáng chú ý nhất là train Evo-1 Stage 1 trên đầu ra VLM lưu sẵn: tốc độ tăng khoảng 40 lần, val loss từ 0,327 xuống 0,035, sai số offline từ 44,3° xuống 15,0°. Tuy vậy, Evo-1 **vẫn chưa vượt** baseline đứng yên ở 10/11 task và còn kém ACT trên task_01, nên chưa thể coi là một policy dùng được. Thêm vào đó, policy train bằng dữ liệu thật chưa chạy tốt trong mô phỏng do lệch miền hình ảnh. Bước tiếp theo là hoàn thành lần 3 của Stage 1, so sánh REAL_ONLY với REAL_PLUS_SIM, rồi Stage 2 Lite. Sau đó mới làm CKA/CLP và lượng tử hoá theo thứ tự W8A8 → W4A8 → W4A4. Project hoàn thành khi có model cuối vượt baseline, được đánh giá đủ 11 task (kể cả vị trí chưa thấy) kèm video, chạy được trên máy local, và qua được dry-run trên robot thật.

| Thành phần | Trạng thái | Ghi chú |
|---|---|---|
| Dataset + validation + split | PASS | 551 episode; 432/55/55; 9 episode loại |
| Môi trường local | PASS | RTX 3050 6 GB, BF16, LeRobot 0.6.1 |
| Mô phỏng MuJoCo 11 task | PASS | oracle 65/70 ở vị trí chưa thấy |
| Dataset mô phỏng / REAL_PLUS_SIM | PASS (dữ liệu) | 220 episode mô phỏng; dataset trộn chưa được train |
| ACT baseline | PASS (offline) / NOT ACHIEVED (mô phỏng) | 8,9° so với đứng yên 11,1°; mô phỏng 0/20 |
| Evo-1 Stage 1 | NOT ACHIEVED / PAUSED | 15,0° so với đứng yên ~12,0°; lần 3 dừng ở bước 1.000/20.000 |
| Stage 2 Lite | PENDING | |
| CKA / CLP | PENDING | |
| W8A8 / W4A8 | PENDING | |
| W4A4 / FoldQuantVLA | PENDING | native trên sm_86 chưa xác minh |
| Đánh giá 11 task + video | PENDING | |
| Deploy SO-101 | PENDING (một phần) | lớp an toàn + dry-run replay đã có |

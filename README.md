# STAR-RIS–RSMA: phân bổ tài nguyên bằng học tăng cường sâu

Mã nguồn, dữ liệu kết quả và luận văn thạc sĩ:

> **Tối ưu phân bổ tài nguyên sử dụng học tăng cường sâu trong mạng STAR-RIS hỗ trợ RSMA**
> Nguyễn Duy Thành — Trường Đại học Giao thông Vận tải TP. Hồ Chí Minh (2026)

| | |
|---|---|
| Luận văn (PDF) | [`thesis_latex/main.pdf`](thesis_latex/main.pdf) — bản nộp khóa bằng thẻ `thesis-v6` |
| Mã LaTeX | [`thesis_latex/`](thesis_latex/) (biên dịch: xelatex → biber → xelatex → xelatex; nhật ký `thesis_latex/build_v6.log`) |
| Bộ kết quả duy nhất được trích dẫn | [`results/physical_v6_full_r2/`](results/physical_v6_full_r2/) và dữ liệu thô bộ giải không học [`results/six_method_v2/raw/`](results/six_method_v2/raw/) |
| Tài liệu tham khảo | [`references/`](references/) (xem `references/TLTK_README.md`) |

## Giao thức thực nghiệm V6 (giao thức của luận văn)

- Hệ SISO, `K = 4` UE, STAR-RIS chế độ chia năng lượng, `N = 16, 32, 64, 96, 128`.
- Tham số hóa hành động `physical_v6_soft_anchor`: bộ giải mã có cấu trúc vật lý dùng chung cho mọi thuật toán DRL (`src/star_ris_rsma/action.py`).
- Cấu hình: `configs/v3/pilot_v6_soft_anchor_n{16,32,64,96,128}.yaml`.
- Sáu phương pháp: **TD3, DDPG, PPO** (5 hạt giống × 5 giá trị N = 75 lần huấn luyện, mỗi lần 100.000 tương tác) và ba bộ giải không học **AO–SCA, AO–Grid, AnalyticalRIS** (`src/star_ris_rsma/baselines/ao_corrected.py`, `ao_grid_corrected.py`, `analytical_ris.py`).
- Ngân hàng kịch bản khóa: huấn luyện 10.000 (seed 11001), xác thực 1.000 (seed 22001), kiểm thử 1.000 (seed 33001); nguồn gốc và checksum ghi trong `results/physical_v6_full_r2/scenario_banks/BANK_PROVENANCE.json`.
- Chọn điểm lưu chỉ trên tập xác thực; đánh giá xác định trên tập kiểm thử.
- Độ trễ: CPU một luồng, `--warmup 10 --count 100`, báo cáo trung vị.

## Cài đặt và kiểm thử

```bash
python -m pip install -e .[dev]
pytest -q
```

## Tái lập

Repo không chứa ngân hàng kịch bản (`.npz`) và điểm lưu mô hình (`.pt`); chúng được tạo lại như sau.

```bash
# 1. Ngân hàng kịch bản V6 (kiểm checksum với bản đã khóa)
python scripts/prepare_v6_scenario_banks.py --output-dir artifacts/scenario_banks
python scripts/prepare_v6_scenario_banks.py --output-dir artifacts/scenario_banks --verify-existing

# 2. Huấn luyện DRL (lặp cho method ∈ {td3, ddpg, ppo}, seed ∈ {0..4}, N ∈ {16, 32, 64, 96, 128})
python scripts/run_train_drl_v3.py --method td3 \
  --config configs/v3/pilot_v6_soft_anchor_n32.yaml --seed 0 \
  --output results/train/td3/N32/seed_0

# 3. Bộ giải không học trên 1.000 kịch bản kiểm thử
python scripts/run_corrected_baseline.py --method ao_grid --n-ris 32 \
  --start 0 --end 1000 --output results/solvers/N32/ao_grid.csv

# 4. Độ trễ CPU một luồng
python scripts/benchmark_latency_v2.py --method td3 \
  --config configs/v3/pilot_v6_soft_anchor_n32.yaml \
  --checkpoint results/train/td3/N32/seed_0/best.pt \
  --warmup 10 --count 100 --output results/latency/td3_N32.csv

# 5. Sinh Bảng 4.3 của luận văn từ CSV đã khóa (không cần huấn luyện)
python scripts/make_thesis_table_sum_rate.py
```

Lần chạy gốc dùng Kaggle (`scripts/orchestrate_kaggle_v6_full.py`) và được tổng hợp bằng `scripts/build_physical_v6_full_report.py`.

## Truy xuất phiên bản

| Tầng | Commit |
|---|---|
| Huấn luyện DRL (75 lần chạy) | `3e96df18eab0a8b3a6a3f1006d74c31e09add2a1` |
| Bộ giải không học trên tập kiểm thử | `c779fa5d8f171a89242d7e64e8b78356dd2b2542` |
| Đo độ trễ CPU | `346e125e78fb4d3d6a0f6e62b2590b1ea3eaf711` |
| Tổng hợp kết quả (`experiment/physical-v6-full`) | `fde2fe7798eaf080886b121cb7f66492b4c9bc48` |
| Luận văn bản nộp | thẻ `thesis-v6` (`git rev-list -n 1 thesis-v6`) |

## Thư mục lịch sử

`results/six_method_v1/`, `results/final_thesis_paper_bundle/`, `results/FINAL_RESULTS_*`, các cấu hình `configs/v3/constrained_action_n*.yaml`, `configs/siso_n*.yaml` và README cũ ([`docs/history/README_v3_protocol.md`](docs/history/README_v3_protocol.md)) thuộc các giao thức trước (v1–v3) và **không** được trích dẫn trong luận văn. Chúng được giữ lại để truy vết.

# Kiểm tra lại bằng chứng V6 r2 trong GitHub Release

Release: [`physical-v6-r2-evidence-9547c62`](https://github.com/Juliolayme/STAR_RIS_RSMA_TD3/releases/tag/physical-v6-r2-evidence-9547c62) — kết quả: **PASS** (75 lưu trữ huấn luyện, 0 lỗi).

Sinh bằng `scripts/verify_v6_release_evidence.py` (không huấn luyện lại; đánh giá lại trên CPU). Chi tiết từng lưu trữ: `RELEASE_VERIFICATION.json`.

| Kiểm tra | Kết quả |
|---|---|
| SHA-256 của 4 tệp release so với `EVIDENCE_MANIFEST.json` | 4/4 khớp |
| SHA-256 của 15 ScenarioBank lồng bên trong | 15/15 khớp |
| Checksum nội dung 5 ngân hàng kiểm thử so với cột `bank_checksum` của dữ liệu thô | 5/5 khớp |
| SHA-256 của 75 lưu trữ huấn luyện lồng bên trong | 75/75 khớp |
| Đủ 14 tệp bắt buộc trong mỗi lưu trữ (`best.pt`, `initial.pt`, `latest.pt`, …) | 75/75 |
| `best.pt` hạt giống 0 so với `LATENCY_CHECKPOINT_INDEX.json` | 15/15 khớp |
| `test_best_raw.csv` trong lưu trữ so với `raw/DRL_V6_TEST_BEST_RAW_ALL.csv` | 75 000/75 000 dòng; sai lệch tổng tốc độ lớn nhất 3.6e-15 bit/s/Hz |
| Nạp lại `best.pt`, đánh giá lại 1.000 kịch bản kiểm thử trên CPU | sai lệch từng kịch bản lớn nhất 7.6e-05 bit/s/Hz (sai số số học GPU/CPU); 0 kịch bản đổi trạng thái đạt QoS |
| Trung bình và độ lệch chuẩn theo (thuật toán, N) so với `TABLE_V6_SIX_METHOD_PERFORMANCE.csv` (Bảng 4.2–4.4 luận văn) | sai lệch lớn nhất 3.4e-08 bit/s/Hz; khớp đến 4 chữ số thập phân |

## Tái lập kiểm tra

```bash
python results/physical_v6_full_r2/evidence/download_evidence.py --destination artifacts/v6_evidence
python scripts/verify_v6_release_evidence.py --assets artifacts/v6_evidence
```

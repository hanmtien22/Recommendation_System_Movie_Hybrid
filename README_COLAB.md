# Hybrid Movie Recommendation System

Hệ thống khuyến nghị phim kết hợp (Hybrid) sử dụng các phương pháp lọc cộng tác (Collaborative Filtering), lọc dựa trên nội dung (Content-Based Filtering) và thuật toán xếp hạng mức độ phổ biến (Popularity-Based). Hệ thống được đóng gói và triển khai giao diện trực quan bằng Streamlit.

## Tính năng chính
- **Collaborative Filtering:** Sử dụng thuật toán SVD (Singular Value Decomposition) để dự đoán mức độ yêu thích của người dùng dựa trên lịch sử tương tác.
- **Content-Based Filtering:** Sử dụng TF-IDF Vectorizer và tính toán độ tương đồng Cosine (Cosine Similarity) dựa trên metadata của phim (genres, keywords, cast, crew).
- **Hybrid Approach:** Kết hợp điểm số từ các mô hình thành phần (Blending/Weighting) để tối ưu hóa kết quả, giúp giải quyết bài toán khởi đầu lạnh (Cold Start Problem).
- **Giao diện Real-time:** Đóng gói mô hình tĩnh (Artifacts serialization) và deploy ứng dụng web gợi ý thời gian thực.

## Kết quả thực nghiệm (Evaluation)
Hệ thống được đánh giá dựa trên các chỉ số RMSE, MAE, Precision@K. Kết quả so sánh giữa các mô hình:

| Mô hình | Precision@K | Recall@K | NDCG@K |
| ALS implicit| 0.08325 | 0.526456 | 0.337205 |
| Logistic blender| 0.08325 | 0.526012 | 0.331105 |
| BPR | 0.07175 | 0.378576 | 0.260644 |
| MF | 0.04050 | 0.193564 | 0.140479 |
| CBF| 0.01950 | 0.115268 | 0.079145 |
| Popularity-Based | 0.05225 | 0.261312 | 0.191665 |
| Item-KNN baseline | 0.07925 | 0.424347 | 0.297547 |
| User-KNN baseline| 0.07675 | 0.422207 | 0.298856 |
| Random baseline | 0.00425	 | 0.018477 | 0.010052 |
| Bias-only baseline | 0.04875 | 0.239451 | 0.175805 |
| **Hybrid Model (Đề xuất)** | **0.08800** | **0.541267** | **0.361113** |

## Hướng dẫn cài đặt & Chạy Local

### 1. Cài đặt môi trường
Khuyến nghị sử dụng Python 3.10+. Khởi tạo môi trường ảo và cài đặt các thư viện phụ thuộc:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
### 2.Chạy ứng dụng giao diện (Streamlit)
```bash
streamlit run src/app.py
```
### 3. Cấu trúc thư mục dự án 
├── data/                  # Thư mục chứa dữ liệu gốc (.csv)
├── models/                # Lưu trữ trọng số mô hình sau khi train (.pkl)
├── notebooks/             # Thử nghiệm thuật toán & phân tích dữ liệu (EDA)
├── src/                   # Mã nguồn xử lý chính (DataLoader, Evaluation, Recommender)
├── app.py                 # File chạy ứng dụng giao diện Streamlit
├── requirements.txt       # Danh sách thư viện bắt buộc
└── README.md              # Tài liệu dự án

# Hybrid Movie Recommendation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

**Hệ thống khuyến nghị phim kết hợp (Hybrid) sử dụng các phương pháp lọc cộng tác, lọc dựa trên nội dung và xếp hạng phổ biến**

[Tính năng](#-tính-năng-chính) • [Cài đặt](#-cài-đặt) • [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng) • [Kiến trúc](#-kiến-trúc-thuật-toán) • [Kết quả](#-kết-quả-thực-nghiệm)

</div>

---

## 📋 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Tính năng chính](#-tính-năng-chính)
- [Tập dữ liệu](#-tập-dữ-liệu)
- [Kiến trúc thuật toán](#-kiến-trúc-thuật-toán)
- [Kết quả thực nghiệm](#-kết-quả-thực-nghiệm)
- [Cài đặt](#-cài-đặt)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
- [Cấu trúc dự án](#-cấu-trúc-dự án)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)

---

## 🎯 Giới thiệu

**Hybrid Movie Recommendation System** là một hệ thống khuyến nghị phim toàn diện, kết hợp ba phương pháp chính:

- **Collaborative Filtering (CF):** Dựa trên hành vi và sở thích của người dùng tương tự
- **Content-Based Filtering (CBF):** Dựa trên đặc trưng và nội dung của bộ phim
- **Popularity-Based:** Dựa trên mức độ phổ biến và đánh giá chung của phim

Hệ thống sử dụng kỹ thuật **Blending** để tích hợp điểm số từ các mô hình khác nhau, giải quyết bài toán **Cold Start** và tăng cường độ chính xác dự đoán.

---

## ✨ Tính năng chính

### 🔄 Collaborative Filtering
- Sử dụng **SVD (Singular Value Decomposition)** để phân rã ma trận user-item
- Học các tiềm ẩn (latent factors) từ lịch sử tương tác người dùng
- Dự đoán đánh giá tiềm ẩn cho các phim chưa xem

### 📝 Content-Based Filtering
- Trích xuất đặc trưng phim từ metadata (genres, keywords, cast, crew, overview)
- Sử dụng **TF-IDF Vectorizer** để chuyển đổi văn bản thành vector đặc trưng
- Tính toán **Cosine Similarity** để tìm phim có nội dung tương đồng

### ⭐ Popularity-Based
- Xếp hạng phim dựa trên điểm đánh giá trung bình
- Xây dựng baseline mạnh để so sánh hiệu năng
- Hữu ích cho các người dùng mới (Cold Start Problem)

### 🤖 Hybrid Approach
- Tích hợp điểm số từ CF, CBF và Popularity-Based
- Sử dụng **Logistic Blender** để tối ưu hóa trọng số của từng mô hình
- Xử lý hiệu quả **Cold Start Problem**
- Tăng cường độ chính xác tổng thể của hệ thống

### 🚀 Triển khai Real-time
- Serialization mô hình thành file `.pkl` để tối ưu hiệu suất
- Giao diện web interactif với **Streamlit**
- Phản hồi nhanh chóng cho các truy vấn người dùng

---

## 📊 Tập dữ liệu

Dự án kết hợp hai tập dữ liệu công khai:

### MovieLens Dataset
- **Nguồn:** https://grouplens.org/datasets/movielens/
- **Nội dung:** Ma trận tương tác User-Item (ratings, timestamps)
- **Quy mô:** Hàng triệu interactions từ hàng trăm nghìn người dùng
- **Sử dụng cho:** Collaborative Filtering

### TMDB 5000 Movie Dataset
- **Nguồn:** https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
- **Nội dung:** Metadata chi tiết về phim (genres, keywords, cast, crew, overview, ratings)
- **Quy mô:** ~5000 phim với thông tin phong phú
- **Sử dụng cho:** Content-Based Filtering

### ⚠️ Lưu ý về dữ liệu

Các file dữ liệu `.csv` có dung lượng lớn đã được thêm vào `.gitignore`. Để chạy hệ thống cục bộ:

1. Tải dữ liệu từ các nguồn trên
2. Đặt vào thư mục `data/` theo cấu trúc:
   ```
   data/
   ├── ratings.csv          (MovieLens)
   ├── movies_metadata.csv  (TMDB)
   └── keywords.csv         (TMDB)
   ```

---

## 🏗️ Kiến trúc thuật toán

### Phương pháp Collaborative Filtering

```
User Ratings Matrix (m × n)
         ↓
    SVD Decomposition
         ↓
  U × Σ × V^T ≈ R̂
         ↓
  Dự đoán Ratings → Top-K Recommendations
```

**Thuật toán:**
- Matrix Factorization (MF)
- Bayesian Personalized Ranking (BPR)
- Alternating Least Squares (ALS) implicit
- User-KNN / Item-KNN

### Phương pháp Content-Based Filtering

```
Movie Metadata (genres, keywords, etc.)
         ↓
    TF-IDF Vectorization
         ↓
   Feature Vectors
         ↓
  Cosine Similarity Computation
         ↓
  Top-K Similar Movies
```

### Phương pháp Hybrid (Blending)

```
CF Score (s_cf) ┐
                ├→ Blending/Weighting ┐
CBF Score (s_cbf)┤                    ├→ Final Score
                ├→ Logistic Regression
Pop Score (s_pop)┘                    ↓
                           Hybrid Recommendation
```

**Cơ chế tích hợp:**
- Weighted Average: `score = w_cf × s_cf + w_cbf × s_cbf + w_pop × s_pop`
- Logistic Regression: Học trọng số tối ưu từ dữ liệu validation

---

## 📈 Kết quả thực nghiệm

### Kết quả đánh giá trên tập kiểm thử

| Mô hình | Precision@10 | Recall@10 | NDCG@10 | Ghi chú |
|:---|:---:|:---:|:---:|:---|
| Random Baseline | 0.00425 | 0.01848 | 0.01005 | Baseline ngẫu nhiên |
| Bias-only Baseline | 0.04088 | 0.23945 | 0.17581 | Chỉ sử dụng bias term |
| Content-Based (CBF) | 0.01950 | 0.11527 | 0.07914 | Dựa trên nội dung phim |
| Popularity-Based | 0.05225 | 0.26131 | 0.19166 | Dựa trên phổ biến |
| User-KNN | 0.07675 | 0.42221 | 0.29886 | KNN người dùng |
| Item-KNN | 0.07925 | 0.42435 | 0.29755 | KNN phim |
| Matrix Factorization | 0.04050 | 0.19356 | 0.14048 | SVD cơ bản |
| Bayesian Personalized Ranking | 0.07175 | 0.37858 | 0.26064 | Ranking được cá nhân hóa |
| ALS Implicit | 0.08325 | 0.52646 | 0.33720 | ALS với implicit feedback |
| Logistic Blender | 0.08325 | 0.52601 | 0.33110 | Blending bằng logistic |
| **Hybrid Model (Đề xuất)** | **0.08800** | **0.54127** | **0.36111** | ✅ Hiệu năng tốt nhất |

### 📊 Phân tích kết quả

**Nhận xét chính:**

1. **Vượt trội trên tất cả chỉ số:** Mô hình Hybrid đạt hiệu năng cao nhất trên cả 3 chỉ số (Precision, Recall, NDCG)

2. **Tăng cường Recall:** Tăng 2.67% so với ALS Implicit (0.52646 → 0.54127), chứng tỏ hệ thống khuyến nghị tốt hơn

3. **NDCG cao:** Đạt 0.36111 (tăng 7% so với ALS), chứng minh danh sách gợi ý được sắp xếp chính xác hơn

4. **Giải quyết Cold Start:** Kết hợp Content-Based giúp xử lý tốt người dùng/phim mới

5. **Cân bằng:** Phương pháp Hybrid tìm được cân bằng tối ưu giữa personalization và diversity

---

## 🚀 Cài đặt

### Yêu cầu hệ thống

- **Python:** 3.10 trở lên
- **Pip:** Phiên bản mới nhất
- **OS:** Windows, macOS, Linux

### Bước 1: Clone dự án

```bash
git clone https://github.com/yourusername/hybrid-movie-recommendation.git
cd hybrid-movie-recommendation
```

### Bước 2: Tạo môi trường ảo

```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt môi trường ảo
# Linux/macOS:
source venv/bin/activate

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Windows (CMD):
.\venv\Scripts\activate.bat
```

### Bước 3: Cài đặt thư viện phụ thuộc

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Bước 4: Chuẩn bị dữ liệu

1. Tải dữ liệu từ:
   - MovieLens: https://grouplens.org/datasets/movielens/
   - TMDB: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

2. Tạo thư mục và đặt dữ liệu:
   ```bash
   mkdir -p data
   # Sao chép các file .csv vào data/
   ```

3. Chạy preprocessing (nếu cần):
   ```bash
   python src/data_loader.py
   ```

### Bước 5: Huấn luyện mô hình (Tùy chọn)

```bash
python notebooks/train_models.py
```

Mô hình đã huấn luyện sẽ được lưu vào thư mục `models/`

---

## 📖 Hướng dẫn sử dụng

### Chạy ứng dụng Streamlit

```bash
streamlit run src/app.py
```

Ứng dụng sẽ mở tại `http://localhost:8501`

### Giao diện người dùng

1. **Chọn người dùng:** Nhập User ID từ dataset
2. **Chọn phương pháp:** 
   - Collaborative Filtering
   - Content-Based Filtering
   - Hybrid (Khuyến nghị)
3. **Số lượng gợi ý:** Chọn K (mặc định: 10)
4. **Xem kết quả:** Hiển thị top-K phim được gợi ý với điểm số

### Sử dụng API lập trình

```python
from src.recommender import HybridRecommender

# Khởi tạo recommender
recommender = HybridRecommender(
    cf_weight=0.4,
    cbf_weight=0.3,
    pop_weight=0.3
)

# Lấy gợi ý cho người dùng
recommendations = recommender.recommend(
    user_id=1,
    k=10,
    method='hybrid'
)

# Kết quả: DataFrame với columns [movie_id, title, score]
print(recommendations)
```

---

## 📁 Cấu trúc dự án

```
hybrid-movie-recommendation/
│
├── 📂 data/                              # Thư mục dữ liệu [.gitignore]
│   ├── ratings.csv                       # MovieLens ratings
│   ├── movies_metadata.csv               # TMDB metadata
│   └── keywords.csv                      # TMDB keywords
│
├── 📂 models/                            # Thư mục lưu trữ mô hình [.gitignore]
│   ├── svd_model.pkl                     # Mô hình SVD đã huấn luyện
│   ├── tfidf_vectorizer.pkl              # TF-IDF vectorizer
│   ├── similarity_matrix.pkl              # Ma trận similarity
│   └── blender_model.pkl                 # Logistic blender
│
├── 📂 notebooks/                         # Thử nghiệm & phân tích
│   ├── eda.ipynb                         # Exploratory Data Analysis
│   ├── collaborative_filtering.ipynb     # Thử nghiệm CF
│   ├── content_based_filtering.ipynb     # Thử nghiệm CBF
│   ├── hybrid_approach.ipynb              # Thử nghiệm Hybrid
│   └── evaluation.ipynb                  # Đánh giá mô hình
│
├── 📂 src/                               # Mã nguồn chính
│   ├── __init__.py
│   ├── app.py                            # Ứng dụng Streamlit
│   ├── data_loader.py                    # Tải và xử lý dữ liệu
│   ├── collaborative_filtering.py        # Mô hình CF
│   ├── content_based_filtering.py        # Mô hình CBF
│   ├── popularity_based.py               # Mô hình Popularity
│   ├── recommender.py                    # Hybrid recommender
│   └── evaluation.py                     # Metrics đánh giá
│
├── 📄 requirements.txt                   # Thư viện phụ thuộc
├── 📄 .gitignore                         # Ignore files
├── 📄 README.md                          # Tài liệu này
└── 📄 LICENSE                            # MIT License

```

---

## 🛠️ Yêu cầu hệ thống

### Phụ thuộc chính

```
streamlit>=1.28.0           # Giao diện web
pandas>=2.0.0               # Xử lý dữ liệu
numpy>=1.24.0               # Tính toán số
scikit-learn>=1.3.0         # Machine Learning
scipy>=1.11.0               # Khoa học tính toán
surprise>=0.1               # Recommender systems
implicit>=0.7.0             # Implicit feedback
matplotlib>=3.7.0           # Visualization
seaborn>=0.12.0             # Statistical visualization
```

### Cài đặt chi tiết

Xem file `requirements.txt` để danh sách đầy đủ với phiên bản

---

## 📊 Metrics đánh giá

### Precision@K
Tỷ lệ phim được gợi ý có liên quan trong top-K

```
Precision@K = (# relevant recommendations in top-K) / K
```

### Recall@K
Tỷ lệ phim liên quan được tìm thấy trong top-K

```
Recall@K = (# relevant recommendations in top-K) / (# total relevant items)
```

### NDCG@K
Discounted Cumulative Gain - Đo chất lượng sắp xếp

```
NDCG@K = DCG@K / IDCG@K
```

---

## 🔄 Pipeline xử lý

```
┌─────────────────────────────────────────────────────────────┐
│                    RAW DATA                                  │
│         (MovieLens + TMDB datasets)                          │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│                  DATA PREPROCESSING                          │
│     (Cleaning, Merging, Feature Engineering)                 │
└────────────┬────────────────────────────────────────────────┘
             │
    ┌────────┼────────┬────────────┐
    ▼        ▼        ▼            ▼
┌────────┬────────┬────────┬──────────────┐
│   CF   │  CBF   │  POP   │  HYBRID      │
│ Module │ Module │ Module │  BLENDER     │
└────────┴────────┴────────┴──────────────┘
    │        │        │            │
    └────────┼────────┼────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│              MODEL SERIALIZATION (.pkl)                      │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│           STREAMLIT INTERFACE                                │
│      (Real-time Recommendations)                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 💡 Những điểm nổi bật

✅ **Kết hợp đa phương pháp:** Tối ưu hóa ưu điểm của từng thuật toán  
✅ **Xử lý Cold Start:** Giải quyết bài toán người dùng/phim mới  
✅ **Hiệu suất cao:** Tăng 7% NDCG so với các baseline  
✅ **Giao diện thân thiện:** Streamlit app dễ sử dụng  
✅ **Triển khai nhanh:** Mô hình serialized cho production  
✅ **Có tài liệu:** Notebooks chi tiết cho mỗi bước  
✅ **Dễ mở rộng:** Kiến trúc modular cho các thuật toán mới  




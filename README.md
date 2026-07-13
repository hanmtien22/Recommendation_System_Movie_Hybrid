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
- [Cấu trúc dự án](#-cấu-trúc-dự-án)

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

**Nếu bạn chưa có dữ liệu:**

1. Tải dữ liệu từ:
   - MovieLens: https://grouplens.org/datasets/movielens/
   - TMDB: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

2. Sao chép các file vào thư mục `data/`:
   ```bash
   mkdir -p data
   # Đặt các file .csv vào thư mục data/
   ls data/  # Kiểm tra các file
   ```

3. **Nếu bạn đã có dữ liệu xử lý** (như trong dự án):
   - Các file đã được xử lý sẵn trong `data/`
   - `final_ratings_processed.csv` và `movie_meta_processed.csv` đã sẵn sàng sử dụng
   - Không cần preprocessing thêm

### Bước 5: Tải mô hình (Nếu chưa có)

**Nếu bạn chưa huấn luyện mô hình:**

Chạy Jupyter notebook để huấn luyện:
```bash
jupyter notebook notebook/Hybrid_Recommendation_Movie.ipynb
```

Hoặc chạy từ terminal:
```bash
python -c "from notebook.Hybrid_Recommendation_Movie import *; train_models()"
```

Mô hình sẽ được lưu vào `models/`:
- `svd_model.pkl`
- `tfidf_vectorizer.pkl`
- `content_features.pkl`
- `blend_model.pkl`
- Và các file khác...

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
import pickle
from src.recommender import HybridRecommender

# Tải các mô hình đã huấn luyện
with open('models/svd_model.pkl', 'rb') as f:
    svd_model = pickle.load(f)

with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('models/content_features.pkl', 'rb') as f:
    content_features = pickle.load(f)

with open('models/blend_model.pkl', 'rb') as f:
    blend_model = pickle.load(f)

# Khởi tạo Hybrid Recommender
recommender = HybridRecommender(
    svd_model=svd_model,
    tfidf_vectorizer=tfidf_vectorizer,
    content_features=content_features,
    blend_model=blend_model,
    cf_weight=0.4,
    cbf_weight=0.3,
    pop_weight=0.3
)

# Lấy gợi ý cho người dùng
recommendations = recommender.recommend(
    user_id=1,
    k=10,
    method='hybrid'  # 'cf', 'cbf', 'popularity', 'hybrid'
)

# Kết quả: DataFrame với columns [movie_id, title, score, genres, rating]
print(recommendations)
print(f"Top recommendation: {recommendations.iloc[0]['title']}")
```

### Ví dụ nâng cao

```python
# Collaborative Filtering
cf_recommendations = recommender.recommend(user_id=1, k=10, method='cf')

# Content-Based Filtering
cbf_recommendations = recommender.recommend(user_id=1, k=10, method='cbf')

# Popularity-Based
pop_recommendations = recommender.recommend(user_id=1, k=10, method='popularity')

# Hybrid (Recommended)
hybrid_recommendations = recommender.recommend(user_id=1, k=10, method='hybrid')

# So sánh các phương pháp
print("CF Score:", cf_recommendations.iloc[0]['score'])
print("CBF Score:", cbf_recommendations.iloc[0]['score'])
print("Hybrid Score:", hybrid_recommendations.iloc[0]['score'])
```

---

## 📁 Cấu trúc dự án

```
hybrid-movie-recommendation/
│
├── 📂 data/                              # Thư mục dữ liệu [.gitignore]
│   ├── .gitkeep
│   ├── final_ratings_processed.csv       # Dữ liệu ratings đã xử lý
│   ├── links.csv                         # Links giữa MovieLens và IMDB
│   ├── movie_meta_processed.csv          # Metadata phim đã xử lý
│   ├── movies.csv                        # Danh sách phim gốc
│   ├── ratings.csv                       # Đánh giá gốc từ MovieLens
│   ├── tmdb_5000_credits.csv             # Credits từ TMDB
│   └── tmdb_5000_movies.csv              # Metadata từ TMDB
│
├── 📂 models/                            # Lưu trữ mô hình đã huấn luyện [.gitignore]
│   ├── .gitkeep
│   ├── blend_model.pkl                   # Logistic blender model
│   ├── content_features.pkl              # Đặc trưng nội dung
│   ├── hybrid_config.pkl                 # Cấu hình hybrid
│   ├── mappings.pkl                      # Ánh xạ user/movie IDs
│   ├── popularity_table.pkl              # Bảng xếp hạng phổ biến
│   ├── svd_model.pkl                     # Mô hình SVD
│   └── tfidf_vectorizer.pkl              # TF-IDF vectorizer
│
├── 📂 notebook/                          # Thử nghiệm & phân tích
│   └── Hybrid_Recommendation_Movie.ipynb # Notebook huấn luyện & đánh giá
│
├── 📂 src/                               # Mã nguồn chính
│   ├── app.py                            # Ứng dụng Streamlit
│   ├── data_loader.py                    # Tải và xử lý dữ liệu
│   ├── evaluation.py                     # Metrics & đánh giá
│   └── recommender.py                    # Hybrid recommender engine
│
├── 📄 .gitignore                         # Ignore files (data/, models/)
├── 📄 README.md                          # Tài liệu này
├── 📄 requirements.txt                   # Thư viện phụ thuộc
└── 📄 LICENSE                            # MIT License

```

### 📝 Mô tả chi tiết các thư mục

#### `data/` - Dữ liệu
- **final_ratings_processed.csv** - Dữ liệu ratings đã được làm sạch và xử lý
- **movie_meta_processed.csv** - Metadata phim (genres, cast, crew, keywords) đã xử lý
- **tmdb_5000_movies.csv & tmdb_5000_credits.csv** - Nguồn dữ liệu gốc từ TMDB
- **ratings.csv & movies.csv** - Nguồn dữ liệu gốc từ MovieLens
- **links.csv** - Ánh xạ giữa MovieLens và TMDB IDs

#### `models/` - Mô hình đã huấn luyện
- **svd_model.pkl** - Mô hình SVD cho Collaborative Filtering
- **tfidf_vectorizer.pkl** - Vectorizer TF-IDF cho Content-Based Filtering
- **content_features.pkl** - Ma trận đặc trưng phim
- **popularity_table.pkl** - Bảng xếp hạng và điểm phổ biến
- **blend_model.pkl** - Logistic regression blender
- **hybrid_config.pkl** - Cấu hình và trọng số hybrid
- **mappings.pkl** - Ánh xạ giữa user/movie IDs

#### `notebook/` - Thử nghiệm
- **Hybrid_Recommendation_Movie.ipynb** - Notebook toàn bộ quá trình: EDA, huấn luyện, đánh giá

#### `src/` - Mã nguồn
- **app.py** - Giao diện Streamlit cho người dùng
- **data_loader.py** - Module tải và xử lý dữ liệu
- **recommender.py** - Recommender engine chính (CF, CBF, Hybrid)
- **evaluation.py** - Hàm đánh giá (Precision@K, Recall@K, NDCG@K)
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


## 📄 Mô tả chi tiết các module trong `src/`

### `app.py` - Giao diện Streamlit
Ứng dụng web interactif cho người dùng cuối.

**Tính năng:**
- Chọn User ID từ dropdown
- Lựa chọn phương pháp: CF, CBF, Popularity, hoặc Hybrid
- Điều chỉnh số lượng gợi ý (K)
- Hiển thị danh sách phim được gợi ý
- Thông tin chi tiết phim (title, genres, rating, score)

**Chạy ứng dụng:**
```bash
streamlit run src/app.py
```

### `data_loader.py` - Tải & xử lý dữ liệu
Module xử lý dữ liệu từ các nguồn khác nhau.

**Chức năng chính:**
```python
load_movielens_data()      # Tải dữ liệu MovieLens
load_tmdb_data()           # Tải dữ liệu TMDB
merge_datasets()           # Merge các dataset
preprocess_data()          # Làm sạch & xử lý
create_user_item_matrix()  # Tạo ma trận interactions
```

**Ví dụ sử dụng:**
```python
from src.data_loader import load_movielens_data, load_tmdb_data

ratings_df = load_movielens_data('data/ratings.csv')
movies_df = load_tmdb_data('data/tmdb_5000_movies.csv')
```

### `recommender.py` - Engine khuyến nghị chính
Lõi của hệ thống với ba phương pháp khuyến nghị.

**Các lớp chính:**

```python
class CollaborativeFiltering:
    """SVD-based Collaborative Filtering"""
    def __init__(self, svd_model)
    def recommend(user_id, k=10)
    
class ContentBasedFiltering:
    """TF-IDF + Cosine Similarity"""
    def __init__(self, tfidf_vectorizer, content_features)
    def recommend(user_id, k=10)
    
class PopularityBased:
    """Dựa trên đánh giá trung bình"""
    def __init__(self, popularity_table)
    def recommend(k=10)
    
class HybridRecommender:
    """Kết hợp cả ba phương pháp"""
    def __init__(self, cf_weight=0.4, cbf_weight=0.3, pop_weight=0.3)
    def recommend(user_id, k=10, method='hybrid')
    def blend_scores(cf_score, cbf_score, pop_score)
```

**Ví dụ:**
```python
from src.recommender import HybridRecommender

recommender = HybridRecommender()
result = recommender.recommend(user_id=42, k=10, method='hybrid')
```

### `evaluation.py` - Đánh giá mô hình
Module tính toán các chỉ số đánh giá.

**Hàm chính:**
```python
precision_at_k(recommendations, ground_truth, k=10)
recall_at_k(recommendations, ground_truth, k=10)
ndcg_at_k(recommendations, ground_truth, k=10)

# Đánh giá toàn bộ
evaluate_recommender(recommender, test_ratings, k=10)
```

**Ví dụ:**
```python
from src.evaluation import precision_at_k, recall_at_k, ndcg_at_k

# Tính Precision@10
prec = precision_at_k(pred_ratings, true_ratings, k=10)
print(f"Precision@10: {prec:.4f}")
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

---

## 🔧 Troubleshooting & FAQ

### ❓ Vấn đề: Import module không tìm thấy

**Lỗi:**
```
ModuleNotFoundError: No module named 'src'
```

**Giải pháp:**
```bash
# Đảm bảo bạn ở thư mục gốc của dự án
cd /path/to/hybrid-movie-recommendation

# Thêm thư mục gốc vào PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Hoặc chạy từ thư mục gốc
python -m streamlit run src/app.py
```

### ❓ Vấn đề: File .pkl không tìm thấy

**Lỗi:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'models/svd_model.pkl'
```

**Giải pháp:**
1. Kiểm tra thư mục `models/` có tồn tại không
2. Chạy notebook để huấn luyện mô hình:
   ```bash
   jupyter notebook notebook/Hybrid_Recommendation_Movie.ipynb
   ```
3. Hoặc tải mô hình đã huấn luyện từ:
   - [Link Google Drive / Dropbox tùy vào bạn]

### ❓ Vấn đề: Lỗi import pandas/numpy/sklearn

**Giải pháp:**
```bash
# Cập nhật pip
pip install --upgrade pip

# Cài đặt lại tất cả dependencies
pip install -r requirements.txt --upgrade

# Hoặc cài riêng
pip install pandas numpy scikit-learn scipy streamlit surprise implicit
```

### ❓ Vấn đề: Streamlit port đã được sử dụng

**Lỗi:**
```
Error: Addressalreadyinuse[Errno 48]
```

**Giải pháp:**
```bash
# Chạy trên port khác
streamlit run src/app.py --server.port 8502

# Hoặc kill process sử dụng port 8501
lsof -ti:8501 | xargs kill -9  # macOS/Linux
netstat -ano | findstr :8501   # Windows
```

### ❓ Vấn đề: Memory không đủ khi xử lý dữ liệu lớn

**Giải pháp:**
```python
# Trong data_loader.py, sử dụng chunking
df = pd.read_csv('data/ratings.csv', chunksize=100000)

# Hoặc giới hạn số lượng records
df = pd.read_csv('data/ratings.csv', nrows=1000000)

# Sử dụng data type optimal
df['user_id'] = df['user_id'].astype('int32')
df['movie_id'] = df['movie_id'].astype('int32')
df['rating'] = df['rating'].astype('float32')
```

### ❓ Vấn đề: Kết quả khuyến nghị không đa dạng

**Giải pháp:**
```python
# Tăng diversity bằng cách điều chỉnh trọng số
recommender = HybridRecommender(
    cf_weight=0.3,    # Giảm CF (ít personalized)
    cbf_weight=0.5,   # Tăng CBF (đa dạng nội dung)
    pop_weight=0.2
)

# Hoặc thêm re-ranking logic
def diversity_reranking(recommendations, similarity_threshold=0.7):
    """Remove highly similar movies from recommendations"""
    # Implementation logic here
    pass
```

### ❓ Vấn đề: Hiệu suất chậm trên Streamlit

**Giải pháp:**
```python
# Sử dụng @st.cache để cache kết quả
import streamlit as st

@st.cache_resource
def load_models():
    """Load models once and cache them"""
    # Load logic here
    return recommender

# Hoặc cache dữ liệu
@st.cache_data
def load_data():
    return pd.read_csv('data/final_ratings_processed.csv')
```


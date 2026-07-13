# Chạy project trên Google Colab

Upload **nguyên thư mục** `RECOMMENDATION_SYSTEM` lên Google Drive theo cấu trúc:

```text
MyDrive/
└── RECOMMENDATION_SYSTEM/
    ├── data/
    │   ├── ratings.csv
    │   ├── movies.csv
    │   ├── links.csv
    │   ├── tmdb_5000_movies.csv
    │   └── tmdb_5000_credits.csv
    ├── models/
    ├── app.py
    ├── data_loader.py
    ├── evaluation.py
    ├── recommender.py
    └── Hybrid_Movie_Recommendation1_realtime_streamlit_sync.ipynb
```

Sau đó:

1. Mở file notebook từ Google Drive bằng Google Colab.
2. Chạy từ cell đầu tiên hoặc chọn **Runtime → Run all**.
3. Notebook sẽ tự mount Drive và dùng:

```python
/content/drive/MyDrive/RECOMMENDATION_SYSTEM
```

làm thư mục project chuẩn.

## Lưu ý

- Không upload riêng lẻ mỗi file `.ipynb`; hãy upload cả thư mục.
- Nếu đổi tên thư mục `RECOMMENDATION_SYSTEM`, bạn phải sửa lại đường dẫn trong notebook.
- Sau khi train xong, các artifacts sẽ được lưu vào `models/`.

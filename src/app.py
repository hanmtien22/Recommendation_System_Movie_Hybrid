import re
import textwrap
import pandas as pd
import streamlit as st
from src.data_loader import load_data, load_models
from recommender import MovieRecommender


st.set_page_config(page_title="Hybrid Movie Recommendation", layout="wide")


def clean_html(html_str):
    """Remove newlines and redundant spaces to prevent Streamlit's markdown parser
    from interpreting indented HTML as markdown code blocks."""
    html_str = html_str.replace('\n', ' ')
    return re.sub(r'\s+', ' ', html_str).strip()


def inject_custom_css():
    st.markdown(textwrap.dedent("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Font override */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Sleek Custom Metrics */
    .custom-metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease, border-color 0.2s ease;
        margin-bottom: 15px;
    }
    .custom-metric-card:hover {
        transform: translateY(-3px);
        border-color: #6366f1;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }
    .custom-metric-val {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(to right, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .custom-metric-lbl {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
    }

    /* Recommendation Cards Grid */
    .movie-card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
        gap: 20px;
        padding: 15px 0;
    }
    .movie-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 22px;
        position: relative;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 220px;
        transition: all 0.25s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .movie-card:hover {
        transform: translateY(-4px);
        border-color: #6366f1;
        box-shadow: 0 12px 20px -5px rgba(0, 0, 0, 0.3);
    }
    .movie-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(to bottom, #6366f1, #a855f7);
    }
    .movie-rank {
        position: absolute;
        top: 15px;
        right: 15px;
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 20px;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    .movie-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 5px;
        margin-bottom: 10px;
        line-height: 1.4;
        padding-right: 35px;
    }
    .movie-tags {
        font-size: 0.8rem;
        color: #94a3b8;
        line-height: 1.5;
        margin-bottom: 18px;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .movie-footer {
        border-top: 1px solid #334155;
        padding-top: 12px;
        margin-top: auto;
    }
    .score-lbl {
        font-size: 0.7rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    
    /* Star Rating */
    .stars-container {
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .star-icon {
        color: #fbbf24;
        font-size: 1.1rem;
    }
    .star-value-text {
        font-size: 0.95rem;
        color: #f8fafc;
        font-weight: 600;
        margin-left: 6px;
    }
    
    /* Similarity progress */
    .sim-bar-wrapper {
        display: flex;
        flex-direction: column;
        gap: 5px;
    }
    .sim-percent-text {
        font-size: 0.9rem;
        font-weight: 600;
        color: #34d399;
    }
    .sim-bar-outer {
        background-color: #334155;
        border-radius: 6px;
        height: 6px;
        overflow: hidden;
        width: 100%;
    }
    .sim-bar-inner {
        background: linear-gradient(to right, #10b981, #34d399);
        border-radius: 6px;
        height: 100%;
    }

    /* Hybrid badges */
    .hybrid-score-badge {
        background: linear-gradient(135deg, #5b21b6 0%, #4c1d95 100%);
        border: 1px solid #7c3aed;
        color: #ddd6fe;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 8px;
    }
    .hybrid-details-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
    }
    .factor-tag {
        font-size: 0.65rem;
        background: #0f172a;
        color: #94a3b8;
        padding: 2px 6px;
        border-radius: 4px;
        border: 1px solid #1e293b;
    }
    </style>
    """).strip(), unsafe_allow_html=True)


inject_custom_css()


def render_metric_cards(labels_values):
    cols = st.columns(len(labels_values))
    for col, (label, val) in zip(cols, labels_values):
        with col:
            st.markdown(clean_html(f"""
            <div class="custom-metric-card">
                <div class="custom-metric-val">{val}</div>
                <div class="custom-metric-lbl">{label}</div>
            </div>
            """), unsafe_allow_html=True)


def render_recommendation_cards(df, method):
    if df.empty:
        st.info("Không có gợi ý nào phù hợp.")
        return
        
    df_with_tags = df.merge(movie_meta[["movieId", "tags"]], on="movieId", how="left")
    
    cards_html = []
    for _, row in df_with_tags.iterrows():
        rank = row["rank"]
        title = row["title"]
        tags_raw = row.get("tags", "")
        if pd.isna(tags_raw) or not tags_raw:
            tags_snippet = "Không có mô tả chi tiết."
        else:
            tags_cleaned = str(tags_raw).replace('"', '').strip()
            tags_snippet = tags_cleaned[:120] + "..." if len(tags_cleaned) > 120 else tags_cleaned
            
        footer_html = ""
        if method == "content":
            score = row["similarity_score"]
            percentage = int(score * 100)
            footer_html = f"""
            <div class="score-lbl">Độ tương đồng</div>
            <div class="sim-bar-wrapper">
                <div class="sim-percent-text">{percentage}% tương đồng</div>
                <div class="sim-bar-outer">
                    <div class="sim-bar-inner" style="width: {percentage}%"></div>
                </div>
            </div>
            """
        elif method == "cf":
            rating = row["predicted_rating"]
            full_stars = int(round(rating))
            star_str = "★" * full_stars + "☆" * (5 - full_stars)
            footer_html = f"""
            <div class="score-lbl">Dự đoán đánh giá</div>
            <div class="stars-container">
                <span class="star-icon">{star_str}</span>
                <span class="star-value-text">{rating:.2f} / 5.0</span>
            </div>
            """
        elif method == "hybrid":
            hybrid_score = row["hybrid_score"]
            cf = row.get("cf_score", 0.0)
            cbf = row.get("cbf_score", 0.0)
            pop = row.get("popularity_score", 0.0)
            footer_html = f"""
            <div class="score-lbl">Điểm số Hybrid</div>
            <div class="hybrid-score-badge">Score: {hybrid_score:.4f}</div>
            <div class="hybrid-details-grid">
                <span class="factor-tag">CF: {cf:.2f}</span>
                <span class="factor-tag">CB: {cbf:.2f}</span>
                <span class="factor-tag">Popularity: {pop:.2f}</span>
            </div>
            """
            
        card_html = f"""
        <div class="movie-card">
            <span class="movie-rank">#{rank}</span>
            <div>
                <div class="movie-title">{title}</div>
                <div class="movie-tags">{tags_snippet}</div>
            </div>
            <div class="movie-footer">
                {footer_html}
            </div>
        </div>
        """
        cards_html.append(clean_html(card_html))
        
    st.markdown(clean_html(f'<div class="movie-card-grid">{"".join(cards_html)}</div>'), unsafe_allow_html=True)


@st.cache_resource
def load_recommender():
    movie_meta, ratings, movies, links = load_data("data")
    tfidf, content_features, svd_model, mappings, hybrid_config, popularity_table, blend_model = load_models("models")
    recommender = MovieRecommender(
        movie_meta,
        ratings,
        content_features,
        svd_model,
        mappings,
        hybrid_config=hybrid_config,
        popularity_table=popularity_table,
        blend_model=blend_model,
    )
    return recommender, movie_meta, ratings


recommender, movie_meta, ratings = load_recommender()

if "new_ratings" not in st.session_state:
    st.session_state.new_ratings = pd.DataFrame(columns=["userId", "movieId", "rating"])


def current_ratings():
    return pd.concat(
        [ratings[["userId", "movieId", "rating"]], st.session_state.new_ratings],
        ignore_index=True
    )


st.sidebar.title("🎬 Movie Recommendation")
role = st.sidebar.radio("Vai trò", ["Người dùng", "Quản lý hệ thống"])

if role == "Người dùng":
    page = st.sidebar.selectbox(
        "Chức năng",
        ["Trang chủ", "Tìm kiếm phim", "Gợi ý phim tương tự", "Gợi ý cá nhân hóa", "Gợi ý Hybrid", "Đánh giá phim"]
    )

    if page == "Trang chủ":
        st.title("🎬 Hybrid Movie Recommendation System")
        render_metric_cards([
            ("Số phim", f"{len(movie_meta):,}"),
            ("Số user", f"{ratings['userId'].nunique():,}"),
            ("Số rating", f"{len(ratings) + len(st.session_state.new_ratings):,}")
        ])

    elif page == "Tìm kiếm phim":
        st.title("🔍 Tìm kiếm phim")
        keyword = st.text_input("Nhập tên phim")
        if keyword:
            result = movie_meta[movie_meta["title"].str.contains(keyword, case=False, na=False, regex=False)]
            st.dataframe(result[["movieId", "title"]].head(30), use_container_width=True)

    elif page == "Gợi ý phim tương tự":
        st.title("Gợi ý Phim tương tự (Content-Based)")
        col1, col2 = st.columns([3, 1])
        with col1:
            movie_title = st.text_input("Nhập tên phim", "Toy Story")
        with col2:
            top_n = st.slider("Top N", 5, 30, 10, step=5)
            
        view_mode = st.radio("Chế độ hiển thị", ["Thẻ trực quan (Thân thiện)", "Bảng chi tiết (Phân tích)"], horizontal=True)
        
        if st.button("Lấy gợi ý", type="primary"):
            with st.spinner("Đang tìm phim tương tự..."):
                recs = recommender.get_content_based_recommendations(movie_title, top_n)
                if not recs.empty:
                    if view_mode == "Thẻ trực quan (Thân thiện)":
                        render_recommendation_cards(recs, "content")
                    else:
                        st.dataframe(recs, use_container_width=True)
                else:
                    st.warning("Không tìm thấy phim phù hợp.")

    elif page == "Gợi ý cá nhân hóa":
        st.title("Gợi ý Cá nhân hóa (Collaborative Filtering)")
        col1, col2 = st.columns([2, 2])
        with col1:
            user_id = st.number_input("User ID", min_value=1, value=1, step=1)
        with col2:
            top_n = st.slider("Top N", 5, 30, 10, step=5)
            
        view_mode = st.radio("Chế độ hiển thị", ["Thẻ trực quan (Thân thiện)", "Bảng chi tiết (Phân tích)"], horizontal=True)
        
        if st.button("Lấy gợi ý", type="primary"):
            with st.spinner("Đang phân tích hành vi người dùng..."):
                recs = recommender.get_mf_recommendations(int(user_id), top_n, ratings_data=current_ratings())
                if not recs.empty:
                    if view_mode == "Thẻ trực quan (Thân thiện)":
                        render_recommendation_cards(recs, "cf")
                    else:
                        st.dataframe(recs, use_container_width=True)
                else:
                    st.warning("Không có dữ liệu cho người dùng này.")

    elif page == "Gợi ý Hybrid":
        st.title(" Gợi ý kết hợp (Hybrid)")
        col1, col2 = st.columns([2, 2])
        with col1:
            user_id = st.number_input("User ID", min_value=1, value=1, step=1)
        with col2:
            top_n = st.slider("Top N", 5, 30, 10, step=5)
            
        view_mode = st.radio("Chế độ hiển thị", ["Thẻ trực quan (Thân thiện)", "Bảng chi tiết (Phân tích)"], horizontal=True)
        
        if st.button("Lấy gợi ý Hybrid", type="primary"):
            with st.spinner("Đang tính toán điểm Hybrid..."):
                recs = recommender.get_hybrid_recommendations(
                    int(user_id), top_n=top_n, ratings_data=current_ratings()
                )
                if not recs.empty:
                    if view_mode == "Thẻ trực quan (Thân thiện)":
                        render_recommendation_cards(recs, "hybrid")
                    else:
                        st.dataframe(recs, use_container_width=True)
                else:
                    st.warning("Không tìm thấy dữ liệu gợi ý cho người dùng.")

    elif page == "Đánh giá phim":
        st.title("⭐ Đánh giá phim realtime")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Nhập đánh giá mới")
            user_id = st.number_input("User ID", min_value=1, value=1, step=1)
            selected_title = st.selectbox("Chọn phim", movie_meta["title"].sort_values().unique())
            movie_id = int(movie_meta.loc[movie_meta["title"] == selected_title, "movieId"].iloc[0])
            rating = st.slider("Rating (Số sao)", 0.5, 5.0, 5.0, step=0.5)
            
            if st.button("Lưu đánh giá", type="primary"):
                new_row = pd.DataFrame([{"userId": int(user_id), "movieId": movie_id, "rating": rating}])
                mask = (
                    (st.session_state.new_ratings["userId"] == int(user_id)) &
                    (st.session_state.new_ratings["movieId"] == movie_id)
                )
                st.session_state.new_ratings = st.session_state.new_ratings[~mask]
                st.session_state.new_ratings = pd.concat([st.session_state.new_ratings, new_row], ignore_index=True)
                st.success("Đã lưu đánh giá realtime! Các thuật toán gợi ý sẽ được cập nhật lập tức.")
        
        with col2:
            st.subheader("Đánh giá mới lưu trong phiên này")
            if not st.session_state.new_ratings.empty:
                ratings_display = st.session_state.new_ratings.merge(
                    movie_meta[["movieId", "title"]], on="movieId", how="left"
                )[["userId", "title", "rating"]]
                st.dataframe(ratings_display, use_container_width=True)
            else:
                st.info("Chưa có đánh giá nào được thêm trong phiên này.")

else:
    page = st.sidebar.selectbox("Chức năng", ["Dashboard", "Quản lý phim", "Quản lý người dùng"])

    if page == "Dashboard":
        st.title("📊 System Dashboard")
        render_metric_cards([
            ("Movies", f"{len(movie_meta):,}"),
            ("Users", f"{ratings['userId'].nunique():,}"),
            ("Ratings", f"{len(ratings):,}"),
            ("Avg Rating", f"{ratings['rating'].mean():.2f}")
        ])
        st.subheader("Phân bố rating")
        st.bar_chart(ratings["rating"].value_counts().sort_index())

    elif page == "Quản lý phim":
        st.title("Movie Management")
        keyword = st.text_input("Tìm phim")
        df = movie_meta.copy()
        if keyword:
            df = df[df["title"].str.contains(keyword, case=False, na=False, regex=False)]
        st.dataframe(df, use_container_width=True)

    elif page == "Quản lý người dùng":
        st.title("User Management")
        user_stats = ratings.groupby("userId").agg(
            rating_count=("rating", "count"),
            avg_rating=("rating", "mean")
        ).reset_index()
        st.dataframe(user_stats, use_container_width=True)

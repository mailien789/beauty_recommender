import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import random
import os
import gdown

def download_file_from_drive():
    file_id = "1abcDEFgHIJKlmnOPQ1234567"
    url = f"https://drive.google.com/uc?id=1fQUwBibHbJ3qlvbs6NwpRy8obAplXvX4"
    output = "ratings_with_titles.csv"
    
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

# Cấu hình giao diện
st.set_page_config(page_title="Beauty Recommender 💄✨", page_icon="💄", layout="wide")

# CSS nền hồng pastel
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #fff0f5;
}
[data-testid="stHeader"] {
    background-color: rgba(255, 182, 193, 0.6);
}
[data-testid="stSidebar"] {
    background-color: #ffe4e1;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Đọc dữ liệu
@st.cache_data
def load_data():
    download_file_from_drive()

    df = pd.read_csv("ratings_with_titles.csv")

    # Lọc bớt noise: chỉ lấy user và product có nhiều đánh giá
    user_counts = df["UserId"].value_counts()
    df = df[df["UserId"].isin(user_counts[user_counts >= 5].index)]

    product_counts = df["ProductTitle"].value_counts()
    df = df[df["ProductTitle"].isin(product_counts[product_counts >= 10].index)]

    return df

df = load_data()

# App title
st.title("💖 Beauty Recommender System 💖")
st.write("Chào mừng bạn đến với hệ thống gợi ý mỹ phẩm dựa trên sở thích và đánh giá người dùng! 🛍️✨")

# ==== 1. Popularity-based ====
def popularity_based_recommendation(df, top_n=10):
    product_popularity = df.groupby(['ProductId', 'ProductTitle']).size().reset_index(name='RatingCount')
    top_products = product_popularity.sort_values('RatingCount', ascending=False).head(top_n)
    return top_products[['ProductTitle', 'RatingCount']]

# ==== 2. Collaborative Filtering ====
def collaborative_filtering_recommendation(df, user_id, top_n=10):
    user_product_matrix = df.pivot_table(index='UserId', columns='ProductTitle', values='Rating').fillna(0)
    if user_id not in user_product_matrix.index:
        return pd.DataFrame({'ProductTitle': ['User không tồn tại'], 'PredictedRating': [0]})

    similarity = cosine_similarity(user_product_matrix)
    similarity_df = pd.DataFrame(similarity, index=user_product_matrix.index, columns=user_product_matrix.index)

    similar_users = similarity_df[user_id].sort_values(ascending=False).drop(user_id).head(5).index
    similar_ratings = user_product_matrix.loc[similar_users]
    mean_ratings = similar_ratings.mean().sort_values(ascending=False)

    already_rated = user_product_matrix.loc[user_id]
    unrated_products = already_rated[already_rated == 0].index
    recommended = mean_ratings[unrated_products].head(top_n)

    if recommended.empty:
        return pd.DataFrame({'ProductTitle': ['User đã đánh giá tất cả sản phẩm'], 'PredictedRating': [5]})

    return pd.DataFrame({'ProductTitle': recommended.index, 'PredictedRating': recommended.values})

# ==== 3. Cold Start ====
def cold_start_recommendation(df, top_n=10):
    product_titles = df['ProductTitle'].unique()
    random_products = random.sample(list(product_titles), top_n)
    return pd.DataFrame(random_products, columns=['ProductTitle'])

# ==== 4. Content-based Filtering (mới thêm & tối ưu bằng scipy) ====
def content_based_recommendation(df, selected_product, top_n=10):
    df_filtered = df[df['ProductTitle'].isin(df['ProductTitle'].value_counts()[df['ProductTitle'].value_counts() >= 10].index)]
    if selected_product not in df_filtered['ProductTitle'].unique():
        return pd.DataFrame({'ProductTitle': ['Sản phẩm không tồn tại hoặc quá ít lượt đánh giá'], 'SimilarityScore': [0]})

    product_user_matrix = df_filtered.pivot_table(index='ProductTitle', columns='UserId', values='Rating').fillna(0)
    product_user_sparse = csr_matrix(product_user_matrix.values)

    similarity_matrix = cosine_similarity(product_user_sparse)
    similarity_df = pd.DataFrame(similarity_matrix, index=product_user_matrix.index, columns=product_user_matrix.index)

    if selected_product not in similarity_df.index:
        return pd.DataFrame({'ProductTitle': ['Không tìm thấy sản phẩm tương tự'], 'SimilarityScore': [0]})

    similar_scores = similarity_df[selected_product].sort_values(ascending=False).drop(selected_product).head(top_n)

    if similar_scores.empty or np.all(similar_scores == 0):
        return pd.DataFrame({'ProductTitle': ['Không tìm thấy sản phẩm tương tự'], 'SimilarityScore': [0]})

    return pd.DataFrame({'ProductTitle': similar_scores.index, 'SimilarityScore': similar_scores.values})

# ==== Sidebar ====
st.sidebar.title("✨ Menu Đề Xuất")
option = st.sidebar.radio("Chọn phương pháp đề xuất:", [
    "💫 Popularity-based",
    "👩‍❤️‍👩 Collaborative Filtering",
    "🌸 Cold Start",
    "🎯 Content-based Filtering"
])

st.markdown("---")

# ==== Giao diện chính theo lựa chọn ====
if option == "💫 Popularity-based":
    st.subheader("📌 Top sản phẩm phổ biến nhất")
    top_n = st.slider("Chọn số lượng sản phẩm hiển thị:", 5, 20, 10, key="popularity_slider")
    result = popularity_based_recommendation(df, top_n)
    st.dataframe(result, use_container_width=True)

elif option == "👩‍❤️‍👩 Collaborative Filtering":
    st.subheader("📌 Gợi ý sản phẩm dành riêng cho User")
    user_ids = df['UserId'].unique().tolist()
    user_id = st.selectbox("Chọn UserId:", user_ids)
    if st.button("🔍 Gợi ý ngay", key="collab_button"):
        result = collaborative_filtering_recommendation(df, user_id)
        st.dataframe(result, use_container_width=True)

elif option == "🌸 Cold Start":
    st.subheader("📌 Gợi ý cho khách hàng mới (chưa có đánh giá)")
    top_n = st.slider("Chọn số lượng sản phẩm hiển thị:", 5, 20, 10, key="cold_start_slider")
    result = cold_start_recommendation(df, top_n)
    st.dataframe(result, use_container_width=True)

elif option == "🎯 Content-based Filtering":
    st.subheader("📌 Gợi ý sản phẩm tương tự")
    product_titles = sorted(df['ProductTitle'].dropna().unique().tolist())
    selected_product = st.selectbox("Chọn một sản phẩm:", product_titles)
    if st.button("🔍 Gợi ý sản phẩm tương tự", key="content_button"):
        result = content_based_recommendation(df, selected_product)
        st.dataframe(result, use_container_width=True)

st.markdown("---")
st.write("© 2025 | Beauty Recommender by Liên Tiên 😎")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random

# Cấu hình giao diện
st.set_page_config(page_title="Beauty Recommender 💄✨", page_icon="💄", layout="wide")

# CSS nhẹ cho app nền hồng pastel
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
    df = pd.read_csv('ratings_with_titles.csv')
    return df

df = load_data()

# Tựa đề app
st.title("💖 Beauty Recommender System 💖")
st.write("Chào mừng bạn đến với hệ thống gợi ý mỹ phẩm dựa trên sở thích và đánh giá người dùng! 🛍️✨")

# Part I: Popularity-based
def popularity_based_recommendation(df, top_n=10):
    product_popularity = df.groupby(['ProductId', 'ProductTitle']).size().reset_index(name='RatingCount')
    top_products = product_popularity.sort_values('RatingCount', ascending=False).head(top_n)
    return top_products[['ProductTitle', 'RatingCount']]

# Part II: Collaborative Filtering (dùng sklearn cosine similarity)
def collaborative_filtering_recommendation(df, user_id, top_n=10):
    user_product_matrix = df.pivot_table(index='UserId', columns='ProductTitle', values='Rating').fillna(0)

    if user_id not in user_product_matrix.index:
        return pd.DataFrame({'ProductTitle': ['User không tồn tại'], 'PredictedRating': [0]})

    similarity_matrix = cosine_similarity(user_product_matrix)
    similarity_df = pd.DataFrame(similarity_matrix, index=user_product_matrix.index, columns=user_product_matrix.index)

    similar_users = similarity_df[user_id].sort_values(ascending=False).drop(user_id).head(5).index

    similar_ratings = user_product_matrix.loc[similar_users]
    mean_ratings = similar_ratings.mean().sort_values(ascending=False)

    already_rated = user_product_matrix.loc[user_id]
    unrated_products = already_rated[already_rated == 0].index
    recommended = mean_ratings[unrated_products].head(top_n)

    if recommended.empty:
        return pd.DataFrame({'ProductTitle': ['User đã đánh giá tất cả sản phẩm'], 'PredictedRating': [5]})

    return pd.DataFrame({'ProductTitle': recommended.index, 'PredictedRating': recommended.values})

# Part III: Cold start
def cold_start_recommendation(df, top_n=10):
    product_titles = df['ProductTitle'].unique()
    random_products = random.sample(list(product_titles), top_n)
    return pd.DataFrame(random_products, columns=['ProductTitle'])

# Sidebar điều hướng
st.sidebar.title("✨ Menu Đề Xuất")
option = st.sidebar.radio("Chọn phương pháp đề xuất:", ["💫 Popularity-based", "👩‍❤️‍👩 Collaborative Filtering", "🌸 Cold Start"])

st.markdown("---")

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

st.markdown("---")
st.write("© 2025 | Beauty Recommender by Liên Tiên 😎")

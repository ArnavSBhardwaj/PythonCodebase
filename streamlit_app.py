import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

st.set_page_config(page_title="Intelligent Product Recommender", layout="centered")

st.title("🛍️ Intelligent Product Recommender")

# Load resources
with open("/content/drive/My Drive/Intelligent Product Recommendation System for E-commerce/app/product_enc.pkl", "rb") as f:
    product_enc = pickle.load(f)

with open("/content/drive/My Drive/Intelligent Product Recommendation System for E-commerce/app/product_id_map.pkl", "rb") as f:
    product_id_map = pickle.load(f)

with open("/content/drive/My Drive/Intelligent Product Recommendation System for E-commerce/app/product_similarity.npy", "rb") as f:
    product_similarity = np.load(f)

with open("/content/drive/My Drive/Intelligent Product Recommendation System for E-commerce/app/content_similarity.npy", "rb") as f:
    content_similarity = np.load(f)

# Create set of known product IDs (as strings)
known_products = set(str(pid) for pid in product_enc.classes_)

# Input UI
product_input = st.text_input("🔍 Enter a Product ID:", value="")
model_type = st.selectbox("📊 Choose Recommendation Type:", ["Collaborative", "Content-Based", "Hybrid"])
top_n = st.slider("📌 How many recommendations?", min_value=1, max_value=10, value=5)

# Show sample product IDs to assist users
with st.expander("💡 Sample Product IDs (try one of these)"):
    sample_ids = list(known_products)[:5]
    for sid in sample_ids:
        st.code(sid)

# Recommendation button logic
if st.button("🔎 Recommend"):
    product_input = product_input.strip()
    
    if product_input not in known_products:
        st.error("❌ Product not found in the dataset. Try one of the sample IDs above.")
    else:
        idx = product_enc.transform([product_input])[0]

        if model_type == "Collaborative":
            scores = product_similarity[idx]
        elif model_type == "Content-Based":
            scores = content_similarity[idx]
        else:
            scores = 0.6 * product_similarity[idx] + 0.4 * content_similarity[idx]

        sim_scores = list(enumerate(scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

        st.success("✅ Recommendations generated!")
        st.subheader("🧠 Recommended Products:")
        for i, (prod_idx, score) in enumerate(sim_scores, 1):
            rec_product = product_id_map[prod_idx]
            st.write(f"{i}. Product ID: `{rec_product}` | Score: `{score:.4f}`")

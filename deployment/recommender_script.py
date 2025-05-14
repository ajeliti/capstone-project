import numpy as np
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("recommender.pkl")

# Load data
@st.cache_data
def load_products():
    return pd.read_csv("cleaned data.csv", on_bad_lines='skip')

product_df = load_products()

# Skin tone descriptions
skin_tone_descriptions = {
    'light': "Light skin that may burn easily and rarely tans.",
    'fair': "Very light or pale skin, often with freckles and light hair.",
    'light medium': "A balance between light and medium skin tones.",
    'medium': "Beige or olive skin that tans gradually.",
    'mediumTan': "Warmer tone, tans more easily and rarely burns.",
    'fairLight': "Lighter than light medium, often cool-toned.",
    'tan': "Naturally tan skin tone that rarely burns.",
    'deep': "Dark skin that almost never burns.",
    'rich': "Very dark, melanin-rich skin, often with cool undertones.",
    'porcelain': "Extremely fair, delicate skin, often very sensitive.",
    'olive': "Neutral or greenish undertones, typically tans easily.",
    'dark': "Dark brown skin that resists sunburn.",
    'ebony': "Deepest skin tone with cool or neutral undertones."
}

# Skin type descriptions
skin_type_descriptions = {
    'dry': "Skin that tends to feel tight and rough, often flaking or peeling.",
    'oily': "Skin that produces excess sebum, often with shiny areas, especially on the forehead, nose, and chin.",
    'combination': "Skin that has both dry and oily areas, typically oily in the T-zone and dry on the cheeks.",
    'normal': "Balanced skin that isn’t too oily or dry, with few imperfections.",
    'sensitive': "Skin that is prone to irritation, redness, or burning from products or environmental factors."
}

# App Title
st.title("Skincare Product Recommender App")
st.markdown("Get personalized skincare suggestions based on your skin type, tone, and budget.")

# Sidebar for input
st.sidebar.header("Customize Your Preferences")
skin_type_options = sorted(product_df['skin_type'].dropna().unique())
skin_tone_options = sorted(product_df['skin_tone'].dropna().unique())

# Skin type selection
skin_type = st.sidebar.selectbox("Skin Type", skin_type_options)
if skin_type in skin_type_descriptions:
    st.sidebar.markdown(f"<small>{skin_type_descriptions[skin_type]}</small>", unsafe_allow_html=True)

# Skin tone selection
skin_tone = st.sidebar.selectbox("Skin Tone", skin_tone_options)
if skin_tone in skin_tone_descriptions:
    st.sidebar.markdown(f"<small>{skin_tone_descriptions[skin_tone]}</small>", unsafe_allow_html=True)

# Price slider
price_budget = st.sidebar.slider("Price Budget ($)", min_value=5, max_value=200, value=25)

# Recommend button
if st.sidebar.button("Recommend Products"):
    with st.spinner("Finding the best products for you..."):
        # Filter dataset based on user input
        filtered_df = product_df[
            (product_df['price_usd'] <= price_budget) &
            (product_df['skin_type'] == skin_type) &
            (product_df['skin_tone'] == skin_tone)
        ].copy()

        if filtered_df.empty:
            st.warning("No products matched your preferences. Try changing your filters or increasing your budget.")
        else:
            # Predict
            preds = model.predict(filtered_df)
            filtered_df['score'] = preds
            recommended = filtered_df[filtered_df['score'] == 1]

            # Top results
            top_products = (
                recommended
                .groupby(['product_name', 'brand_name'], as_index=False)
                .agg({'price_usd': 'first', 'rating': 'mean'})
                .sort_values(by='rating', ascending=False)
                .head(10)
            )

            if not top_products.empty:
                st.success(f"Found {len(top_products)} top-rated products within your budget!")
                for _, row in top_products.iterrows():
                    st.markdown(f"""
                        <div style="border:1px solid #ddd; border-radius:10px; padding:15px; margin-bottom:15px;">
                            <h4 style="margin-bottom:5px;">{row['product_name']}</h4>
                            <p style="margin:0;"><strong>Brand:</strong> {row['brand_name']}</p>
                            <p style="margin:0;"><strong>Price:</strong> ${row['price_usd']}</p>
                            <p style="margin:0;"><strong>Rating:</strong> ⭐ {row['rating']:.1f}</p>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No recommended products found for these preferences.")
else:
    st.info("⬅ Adjust your preferences in the sidebar and click *Recommend Products* to begin.")

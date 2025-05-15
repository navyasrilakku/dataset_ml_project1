import numpy as np
import pickle
import streamlit as st
import pandas as pd

# Load model
with open("database_linear_model.pkl", "rb") as file:
    model = pickle.load(file)

# Prediction function
def predict_hpi(article_id, latitude, longitude, page_views, average_views):
    input_data = pd.DataFrame([[article_id, latitude, longitude, page_views, average_views]],
                              columns=["article_id", "latitude", "longitude", "page_views", "average_views"])
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit App
def main():
    st.set_page_config(page_title="HPI Predictor", layout="centered")

    # Branding Header
    st.markdown("""
        <div style="text-align: center; padding-top: 30px;">
            <h1 style="color: #222222; font-size: 2.5em;">🔮 HPI Predictor</h1>
            <p style="color: #555555;">A lightweight ML-powered tool to estimate an article's popularity index in real time</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Input Section
    st.markdown("### Enter Article Metadata")
    st.markdown("*Keep it short and simple. All fields are required.*")

    article_id = st.number_input("🆔 Article ID", min_value=0, step=1)
    latitude = st.number_input("📍 Latitude", format="%.6f")
    longitude = st.number_input("📍 Longitude", format="%.6f")
    page_views = st.number_input("👁️ Page Views", min_value=0, step=1)
    average_views = st.number_input("📊 Average Views", min_value=0.0, format="%.2f")

    st.markdown("")

    # Predict Button
    if st.button("🚀 Predict Now"):
        try:
            prediction = predict_hpi(article_id, latitude, longitude, page_views, average_views)
            st.markdown("---")
            st.markdown(f"""
                <div style="text-align: center; padding: 20px;">
                    <h2 style="color: #006400;">🎯 Predicted HPI: {prediction:.2f}</h2>
                    <p style="color: #777;">This score is based on the current metadata input.</p>
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: gray; font-size: 0.85em;">
            Built with ❤️ using Streamlit · Model: Scikit-learn Linear Regression
        </div>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()

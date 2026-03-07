"""
ML Predictions page — Train model, view metrics, predict cancellations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from src.ml_model import (
    train_model, predict_single, cluster_cancellation_hotspots,
    save_model, load_model,
)


def render():
    st.markdown("## 🤖 ML Cancellation Prediction")

    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first from the Home page or click 'Load Demo Data' in the sidebar.")
        return

    df = st.session_state.df_clean

    tab1, tab2, tab3 = st.tabs(["🎯 Train & Evaluate", "🔮 Predict New Booking", "📍 Cluster Analysis"])

    # ── Tab 1: Train & Evaluate ───────────────────────────────────────────
    with tab1:
        st.markdown("### Model Training")

        col1, col2 = st.columns([1, 2])

        with col1:
            model_type = st.selectbox(
                "Algorithm",
                ["random_forest", "gradient_boosting"],
                format_func=lambda x: "Random Forest" if x == "random_forest" else "Gradient Boosting",
            )
            test_size = st.slider("Test Split", 0.1, 0.4, 0.2, 0.05)

            train_btn = st.button("🚀 Train Model", use_container_width=True)

        if train_btn:
            with st.spinner("Training model... This may take a moment."):
                result = train_model(df, model_type=model_type, test_size=test_size)
                st.session_state.ml_result = result
            st.success("✅ Model trained successfully!")

        if st.session_state.ml_result is not None:
            result = st.session_state.ml_result
            metrics = result["metrics"]

            with col2:
                st.markdown("#### Model Performance")
                m1, m2, m3, m4, m5 = st.columns(5)
                with m1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                with m2:
                    st.metric("Precision", f"{metrics['precision']:.2%}")
                with m3:
                    st.metric("Recall", f"{metrics['recall']:.2%}")
                with m4:
                    st.metric("F1 Score", f"{metrics['f1_score']:.2%}")
                with m5:
                    st.metric("ROC AUC", f"{metrics['roc_auc']:.2%}")

                st.caption(f"Cross-validation F1: {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}")

            # Feature importance
            st.markdown("#### Feature Importance")
            feat_imp = result["feature_importance"]
            fig = px.bar(
                feat_imp, x="Importance", y="Feature",
                orientation="h",
                title="What Drives Cancellations?",
                color="Importance",
                color_continuous_scale=["#1B9E3E", "#ffa502", "#ff6b6b"],
            )
            fig.update_layout(yaxis=dict(autorange="reversed"), height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Confusion matrix
            col1, col2 = st.columns(2)
            with col1:
                cm = result["confusion_matrix"]
                fig = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=["Completed", "Canceled"],
                    y=["Completed", "Canceled"],
                    title="Confusion Matrix",
                    color_continuous_scale="Greens",
                    text_auto=True,
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # ROC-like probability distribution
                y_test = result["y_test"]
                y_proba = result["y_proba"]
                prob_df = pd.DataFrame({"Probability": y_proba, "Actual": y_test})
                prob_df["Actual"] = prob_df["Actual"].map({0: "Completed", 1: "Canceled"})

                fig = px.histogram(
                    prob_df, x="Probability", color="Actual",
                    title="Predicted Probability Distribution",
                    nbins=40, barmode="overlay", opacity=0.7,
                    color_discrete_sequence=["#1B9E3E", "#ff6b6b"],
                )
                st.plotly_chart(fig, use_container_width=True)

            # Save model
            if st.button("💾 Save Model", use_container_width=True):
                path = save_model(result)
                st.success(f"Model saved to {path}")

    # ── Tab 2: Predict New Booking ────────────────────────────────────────
    with tab2:
        st.markdown("### Predict Cancellation Risk for a New Booking")

        if st.session_state.ml_result is None:
            st.info("👆 Please train a model first in the 'Train & Evaluate' tab.")
            return

        result = st.session_state.ml_result
        model = result["model"]
        encoders = result["encoders"]
        features = result["features"]

        col1, col2, col3 = st.columns(3)

        with col1:
            booking_hour = st.slider("Booking Hour", 0, 23, 18)
            ride_distance = st.number_input("Ride Distance (km)", 1.0, 50.0, 8.0, 0.5)
            booking_value = st.number_input("Booking Value (₹)", 20.0, 2000.0, 150.0, 10.0)
            is_weekend = st.checkbox("Weekend?", value=False)

        with col2:
            driver_rating = st.slider("Driver Rating", 1.0, 5.0, 4.0, 0.1)
            customer_rating = st.slider("Customer Rating", 1.0, 5.0, 4.2, 0.1)
            eta_pickup = st.number_input("ETA to Pickup (min)", 1.0, 45.0, 8.0, 0.5)

        with col3:
            vehicle_options = encoders.get("VehicleType", None)
            vehicle_list = list(vehicle_options.classes_) if vehicle_options else ["Mini", "Sedan", "SUV", "Auto", "Bike"]
            vehicle_type = st.selectbox("Vehicle Type", vehicle_list)

            payment_options = encoders.get("PaymentMode", None)
            payment_list = list(payment_options.classes_) if payment_options else ["UPI", "Cash", "Credit Card"]
            payment_mode = st.selectbox("Payment Mode", payment_list)

            location_options = encoders.get("PickupLocation", None)
            loc_list = sorted(list(location_options.classes_)) if location_options else ["Koramangala"]
            pickup_location = st.selectbox("Pickup Location", loc_list)

        if st.button("🔮 Predict", use_container_width=True):
            prediction = predict_single(
                model, encoders, features,
                booking_hour, ride_distance, booking_value,
                driver_rating, customer_rating, eta_pickup,
                is_weekend, vehicle_type, payment_mode, pickup_location,
            )

            st.markdown("---")

            prob = prediction["cancellation_probability"]
            risk = prediction["risk_level"]

            # Risk display
            if risk == "Low":
                color = "#1B9E3E"
                emoji = "✅"
            elif risk == "Medium":
                color = "#ffa502"
                emoji = "⚠️"
            else:
                color = "#ff6b6b"
                emoji = "🚨"

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    f"<div style='text-align:center; padding:1.5rem; background:rgba(0,0,0,0.3); "
                    f"border-radius:12px; border:2px solid {color};'>"
                    f"<div style='font-size:3rem;'>{emoji}</div>"
                    f"<div style='font-size:1.5rem; font-weight:700; color:{color};'>{risk} Risk</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    f"<div style='text-align:center; padding:1.5rem; background:rgba(0,0,0,0.3); "
                    f"border-radius:12px;'>"
                    f"<div style='font-size:2.5rem; font-weight:700; color:{color};'>{prob:.1%}</div>"
                    f"<div style='color:#aaa;'>Cancellation Probability</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with col3:
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    title={"text": "Risk Score"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": color},
                        "steps": [
                            {"range": [0, 30], "color": "rgba(27,158,62,0.2)"},
                            {"range": [30, 60], "color": "rgba(255,165,2,0.2)"},
                            {"range": [60, 100], "color": "rgba(255,107,107,0.2)"},
                        ],
                    },
                    number={"suffix": "%"},
                ))
                fig.update_layout(height=250, margin=dict(t=50, b=0, l=30, r=30))
                st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: Cluster Analysis ───────────────────────────────────────────
    with tab3:
        st.markdown("### Cancellation Hotspot Clustering (KMeans)")

        n_clusters = st.slider("Number of Clusters", 2, 8, 5)

        if st.button("🔍 Run Clustering", use_container_width=True):
            with st.spinner("Clustering..."):
                clusters = cluster_cancellation_hotspots(df, n_clusters=n_clusters)

            if clusters.empty:
                st.warning("Not enough canceled rides for clustering.")
            else:
                st.success(f"✅ {n_clusters} clusters identified")
                st.dataframe(clusters, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    fig = px.scatter(
                        clusters, x="AvgETA", y="CancelCount",
                        color="Cluster", size="AvgFare",
                        hover_data=["PickupLocation"],
                        title="Clusters: ETA vs Cancel Count",
                        color_continuous_scale="Viridis",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    fig = px.bar(
                        clusters.sort_values("CancelCount"),
                        y="PickupLocation", x="CancelCount",
                        color="Cluster",
                        orientation="h",
                        title="Cancellations by Location (Colored by Cluster)",
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)

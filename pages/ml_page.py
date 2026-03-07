"""
ML Predictions page - Train model, view metrics, predict cancellations.
Dark theme styled.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from src.ml_model import train_model, predict_single, cluster_cancellation_hotspots, save_model, load_model


def _apply_dark(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#A0AEC0"), title_font=dict(color="#FFFFFF"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
        legend=dict(font=dict(color="#A0AEC0")),
    )
    return fig


def render():
    st.markdown("""
    <div style="background:#16213E;border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:14px 28px;margin-bottom:20px;display:flex;align-items:center;gap:12px;">
        <svg viewBox="0 0 24 24" fill="#00C853" width="24" height="24"><path d="M21 10.12h-6.78l2.74-2.82c-2.73-2.7-7.15-2.8-9.88-.1-2.73 2.71-2.73 7.08 0 9.79s7.15 2.71 9.88 0C18.32 15.65 19 14.08 19 12.1h2c0 1.98-.88 4.55-2.64 6.29-3.51 3.48-9.21 3.48-12.72 0-3.5-3.47-3.5-9.11 0-12.58 3.51-3.47 9.14-3.49 12.65-.06l2.72-2.63V10.12z"/></svg>
        <div>
            <div style="font-size:20px;font-weight:700;color:#FFF;">ML Cancellation Prediction</div>
            <div style="font-size:12px;color:#A0AEC0;">Train models, predict risk, and discover patterns</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.warning("Please load data first from the Overview Dashboard or click 'Load Demo Data' in the sidebar.")
        return

    df = st.session_state.df_clean

    tab1, tab2, tab3 = st.tabs(["Train & Evaluate", "Predict New Booking", "Cluster Analysis"])

    # Tab 1: Train
    with tab1:
        st.markdown('<div style="font-size:15px;font-weight:700;color:#FFF;margin-bottom:14px;">Model Training</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown('<div style="background:#16213E;border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:22px;">', unsafe_allow_html=True)
            model_type = st.selectbox("Algorithm", ["random_forest", "gradient_boosting"],
                format_func=lambda x: "Random Forest" if x == "random_forest" else "Gradient Boosting")
            test_size = st.slider("Test Split", 0.1, 0.4, 0.2, 0.05)
            train_btn = st.button("Train Model", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if train_btn:
            with st.spinner("Training model... This may take a moment."):
                result = train_model(df, model_type=model_type, test_size=test_size)
                st.session_state.ml_result = result
            st.success("Model trained successfully!")

        if st.session_state.ml_result is not None:
            result = st.session_state.ml_result
            metrics = result["metrics"]

            with col2:
                st.markdown('<div style="background:#16213E;border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:22px;">', unsafe_allow_html=True)
                st.markdown('<div style="font-size:14px;font-weight:600;color:#FFF;margin-bottom:12px;">Model Performance</div>', unsafe_allow_html=True)
                m1, m2, m3, m4, m5 = st.columns(5)
                with m1: st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                with m2: st.metric("Precision", f"{metrics['precision']:.2%}")
                with m3: st.metric("Recall", f"{metrics['recall']:.2%}")
                with m4: st.metric("F1 Score", f"{metrics['f1_score']:.2%}")
                with m5: st.metric("ROC AUC", f"{metrics['roc_auc']:.2%}")
                st.caption(f"Cross-validation F1: {metrics['cv_f1_mean']:.4f} +/- {metrics['cv_f1_std']:.4f}")
                st.markdown('</div>', unsafe_allow_html=True)

            # Feature importance
            st.markdown('<div style="background:#16213E;border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:22px;margin-top:18px;">', unsafe_allow_html=True)
            st.markdown('<div style="font-size:14px;font-weight:600;color:#FFF;margin-bottom:12px;">Feature Importance - What Drives Cancellations?</div>', unsafe_allow_html=True)
            feat_imp = result["feature_importance"]
            fig = px.bar(feat_imp, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale=["#00C853","#FFB300","#F44336"])
            fig.update_layout(yaxis=dict(autorange="reversed"), height=400)
            _apply_dark(fig)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Confusion matrix + prob dist
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div style="background:#16213E;border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:22px;margin-top:18px;">', unsafe_allow_html=True)
                cm = result["confusion_matrix"]
                fig = px.imshow(cm, labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=["Completed","Canceled"], y=["Completed","Canceled"],
                    title="Confusion Matrix", color_continuous_scale=[[0,"#0D0D1A"],[1,"#00C853"]], text_auto=True)
                _apply_dark(fig)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div style="background:#16213E;border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:22px;margin-top:18px;">', unsafe_allow_html=True)
                y_test = result["y_test"]
                y_proba = result["y_proba"]
                prob_df = pd.DataFrame({"Probability": y_proba, "Actual": y_test})
                prob_df["Actual"] = prob_df["Actual"].map({0: "Completed", 1: "Canceled"})
                fig = px.histogram(prob_df, x="Probability", color="Actual",
                    title="Predicted Probability Distribution", nbins=40, barmode="overlay",
                    opacity=0.7, color_discrete_sequence=["#00C853","#F44336"])
                _apply_dark(fig)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            if st.button("Save Model", use_container_width=True):
                path = save_model(result)
                st.success(f"Model saved to {path}")

    # Tab 2: Predict
    with tab2:
        st.markdown('<div style="font-size:15px;font-weight:700;color:#FFF;margin-bottom:14px;">Predict Cancellation Risk for a New Booking</div>', unsafe_allow_html=True)

        if st.session_state.ml_result is None:
            st.info("Please train a model first in the 'Train & Evaluate' tab.")
            return

        result = st.session_state.ml_result
        model = result["model"]
        encoders = result["encoders"]
        features = result["features"]

        col1, col2, col3 = st.columns(3)
        with col1:
            booking_hour = st.slider("Booking Hour", 0, 23, 18)
            ride_distance = st.number_input("Ride Distance (km)", 1.0, 50.0, 8.0, 0.5)
            booking_value = st.number_input("Booking Value (INR)", 20.0, 2000.0, 150.0, 10.0)
            is_weekend = st.checkbox("Weekend?", value=False)
        with col2:
            driver_rating = st.slider("Driver Rating", 1.0, 5.0, 4.0, 0.1)
            customer_rating = st.slider("Customer Rating", 1.0, 5.0, 4.2, 0.1)
            eta_pickup = st.number_input("ETA to Pickup (min)", 1.0, 45.0, 8.0, 0.5)
        with col3:
            vehicle_options = encoders.get("VehicleType", None)
            vehicle_list = list(vehicle_options.classes_) if vehicle_options else ["Mini","Sedan","SUV","Auto","Bike"]
            vehicle_type = st.selectbox("Vehicle Type", vehicle_list)
            payment_options = encoders.get("PaymentMode", None)
            payment_list = list(payment_options.classes_) if payment_options else ["UPI","Cash","Credit Card"]
            payment_mode = st.selectbox("Payment Mode", payment_list)
            location_options = encoders.get("PickupLocation", None)
            loc_list = sorted(list(location_options.classes_)) if location_options else ["Koramangala"]
            pickup_location = st.selectbox("Pickup Location", loc_list)

        if st.button("Predict", use_container_width=True):
            prediction = predict_single(model, encoders, features,
                booking_hour, ride_distance, booking_value,
                driver_rating, customer_rating, eta_pickup,
                is_weekend, vehicle_type, payment_mode, pickup_location)

            prob = prediction["cancellation_probability"]
            risk = prediction["risk_level"]

            if risk == "Low":
                color, icon_path = "#00C853", "M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"
            elif risk == "Medium":
                color, icon_path = "#FFB300", "M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"
            else:
                color, icon_path = "#F44336", "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"

            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div style="text-align:center;padding:1.5rem;background:#16213E;border-radius:12px;border:2px solid {color};">
                    <svg viewBox="0 0 24 24" fill="{color}" width="48" height="48"><path d="{icon_path}"/></svg>
                    <div style="font-size:1.5rem;font-weight:700;color:{color};margin-top:8px;">{risk} Risk</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div style="text-align:center;padding:1.5rem;background:#16213E;border-radius:12px;">
                    <div style="font-size:2.5rem;font-weight:700;color:{color};">{prob:.1%}</div>
                    <div style="color:#A0AEC0;">Cancellation Probability</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                fig = go.Figure(go.Indicator(mode="gauge+number", value=prob*100,
                    title={"text": "Risk Score", "font": {"color": "#FFF"}},
                    gauge={"axis": {"range": [0,100], "tickcolor": "#A0AEC0"},
                           "bar": {"color": color},
                           "bgcolor": "#0D0D1A",
                           "steps": [{"range":[0,30],"color":"rgba(0,200,83,0.2)"},
                                     {"range":[30,60],"color":"rgba(255,179,0,0.2)"},
                                     {"range":[60,100],"color":"rgba(244,67,54,0.2)"}]},
                    number={"suffix": "%", "font": {"color": color}}))
                fig.update_layout(height=250, margin=dict(t=50,b=0,l=30,r=30),
                    paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#A0AEC0"))
                st.plotly_chart(fig, use_container_width=True)

    # Tab 3: Clusters
    with tab3:
        st.markdown('<div style="font-size:15px;font-weight:700;color:#FFF;margin-bottom:14px;">Cancellation Hotspot Clustering (KMeans)</div>', unsafe_allow_html=True)
        n_clusters = st.slider("Number of Clusters", 2, 8, 5)
        if st.button("Run Clustering", use_container_width=True):
            with st.spinner("Clustering..."):
                clusters = cluster_cancellation_hotspots(df, n_clusters=n_clusters)
            if clusters.empty:
                st.warning("Not enough canceled rides for clustering.")
            else:
                st.success(f"{n_clusters} clusters identified")
                st.dataframe(clusters, use_container_width=True)
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.scatter(clusters, x="AvgETA", y="CancelCount", color="Cluster",
                        size="AvgFare", hover_data=["PickupLocation"],
                        title="Clusters: ETA vs Cancel Count", color_continuous_scale="Viridis")
                    _apply_dark(fig)
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fig = px.bar(clusters.sort_values("CancelCount"), y="PickupLocation",
                        x="CancelCount", color="Cluster", orientation="h",
                        title="Cancellations by Location (Colored by Cluster)")
                    fig.update_layout(height=500)
                    _apply_dark(fig)
                    st.plotly_chart(fig, use_container_width=True)

"""Interactive img2GPS demo.

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from img2gps.inference import load_inference_model, predict_image  # noqa: E402


def probability_message(top1_prob: float, topk_mass: float, top_k: int) -> None:
    """Explain grid probability concentration without calling it calibrated accuracy."""
    if top1_prob >= 0.40:
        st.success(
            "The model distribution is strongly concentrated on one grid cell. "
            "This means the grid classifier has a clear preferred region, although GPS accuracy still needs ground truth."
        )
    elif topk_mass >= 0.35:
        st.info(
            f"The top-{top_k} cells together contain a substantial share of probability mass. "
            "The exact cell is uncertain, but the plausible region is relatively concentrated."
        )
    elif topk_mass >= 0.20:
        st.warning(
            f"The model is moderately uncertain: probability is spread beyond the top-{top_k} cells."
        )
    else:
        st.warning(
            "The model is highly uncertain. The image may be ambiguous, outside the target campus region, "
            "or visually different from the training data."
        )


def render_topk_map(result) -> None:
    """Render final prediction and top-k candidate grid centers on one map."""
    topk_df = pd.DataFrame([cell.__dict__ for cell in result.top_k_cells])
    topk_df = topk_df.rename(columns={"center_latitude": "lat", "center_longitude": "lon"})
    topk_df["radius"] = (topk_df["probability"] / topk_df["probability"].max()).clip(lower=0.35) * 90

    pred_df = pd.DataFrame(
        [{"lat": result.latitude, "lon": result.longitude, "label": "soft top-k prediction"}]
    )

    view_state = pdk.ViewState(
        latitude=result.latitude,
        longitude=result.longitude,
        zoom=16,
        pitch=0,
    )

    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            data=topk_df,
            get_position="[lon, lat]",
            get_radius="radius",
            get_fill_color="[255, 140, 0, 140]",
            pickable=True,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=pred_df,
            get_position="[lon, lat]",
            get_radius=28,
            get_fill_color="[0, 120, 255, 220]",
            pickable=True,
        ),
    ]

    st.pydeck_chart(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/dark-v10",
            initial_view_state=view_state,
            layers=layers,
            tooltip={
                "html": "<b>Cell:</b> {cell_id}<br/><b>Probability:</b> {probability}",
                "style": {"color": "white"},
            },
        )
    )
    st.caption(
        "Orange circles are the top-k candidate grid centers, scaled by probability. "
        "The blue point is the final soft top-k GPS prediction. These markers show model uncertainty, "
        "not a measured error radius."
    )


st.set_page_config(page_title="img2GPS Demo", page_icon="📍", layout="wide")
st.title("img2GPS: Campus Image Geolocation")
st.write(
    "Upload a campus image and predict its GPS coordinate using a grid-based geolocation model. "
    "The demo reports top-k grid probabilities to show spatial uncertainty."
)

with st.sidebar:
    checkpoint_path = st.text_input("Checkpoint path", value="checkpoints/best.pt")
    top_k = st.slider("Top-k cells for soft decoding", min_value=1, max_value=10, value=5)
    st.caption(
        "The trained checkpoint is not included in the repository. "
        "Place it under checkpoints/ or provide a custom path."
    )

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    col_img, col_pred = st.columns([1, 1])
    with col_img:
        st.image(uploaded_file, caption="Input image", use_column_width=True)

    checkpoint = ROOT / checkpoint_path
    if not checkpoint.exists():
        st.error(f"Checkpoint not found: {checkpoint}")
        st.stop()

    with st.spinner("Running inference..."):
        model = load_inference_model(checkpoint)
        result = predict_image(model, uploaded_file, top_k=top_k)

    with col_pred:
        st.metric("Predicted latitude", f"{result.latitude:.6f}")
        st.metric("Predicted longitude", f"{result.longitude:.6f}")
        st.metric("Top-1 grid probability", f"{result.confidence:.3f}")
        st.metric(f"Top-{top_k} probability mass", f"{result.top_k_probability_mass:.3f}")
        st.metric("Prediction entropy", f"{result.entropy:.3f}")
        st.caption(
            "Top-1 grid probability is the softmax probability of the single most likely grid cell. "
            f"Top-{top_k} probability mass is the total probability assigned to the cells used for soft top-k decoding. "
            "These values describe spatial concentration, not calibrated GPS accuracy."
        )
        probability_message(result.confidence, result.top_k_probability_mass, top_k)

    st.subheader("Uncertainty-aware map")
    render_topk_map(result)

    st.subheader("Top-k candidate grid cells")
    topk_table = pd.DataFrame([cell.__dict__ for cell in result.top_k_cells])
    topk_table["probability"] = topk_table["probability"].map(lambda x: f"{x:.4f}")
    st.dataframe(topk_table, use_container_width=True)
else:
    st.info("Upload an image to run the demo.")

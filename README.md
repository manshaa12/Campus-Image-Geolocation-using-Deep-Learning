# img2GPS: Campus-Scale Image Geolocation

img2GPS is a computer vision project for predicting GPS coordinates from campus images. Instead of directly regressing latitude and longitude, the model converts geolocation into a structured spatial prediction problem. The target campus region is divided into a 10 by 10 grid, and the model predicts a probability distribution over grid cells before decoding the final GPS coordinate using soft top-k prediction.

This repository is organized as a reproducible ML engineering project with training, evaluation, prediction, uncertainty-aware inference, a Streamlit demo, a FastAPI inference service, Docker support, and basic tests.

## Demo

![Streamlit demo](assets/demo_streamlit.png)

The Streamlit demo shows the full inference workflow:

- upload an input image
- predict latitude and longitude
- visualize the prediction on a map
- display top-k candidate grid cells
- report top-1 grid probability, top-k probability mass, and prediction entropy

The orange circles show top-k candidate grid centers scaled by probability. The blue marker shows the final soft top-k GPS prediction. These markers represent model uncertainty, not a measured error radius.

## Project Highlights

- Predicts GPS coordinates from campus images
- Uses a ResNet-50 backbone for visual feature extraction
- Reformulates GPS prediction as grid-based spatial localization
- Uses soft top-k decoding instead of hard argmax prediction
- Reports Haversine distance for geographic error evaluation
- Provides top-k grid probability outputs for uncertainty interpretation
- Includes Streamlit and FastAPI interfaces for interactive and API-based inference
- Uses Hugging Face Datasets for external dataset loading
- Keeps raw data, reports, and checkpoints outside the GitHub repository

## Quick Start

Create a clean Python environment:

```bash
conda create -n img2gps python=3.11 -y
conda activate img2gps
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

Run tests:

```bash
pytest -q
```

Place the trained checkpoint at:

```text
checkpoints/best.pt
```

For example:

```bash
mkdir -p checkpoints
cp /path/to/model.pt checkpoints/best.pt
```

Run prediction on an example image:

```bash
python -m img2gps.predict \
  --checkpoint checkpoints/best.pt \
  --image examples/images/example_01.jpg
```

## Streamlit Demo

Run the interactive demo:

```bash
streamlit run app/streamlit_app.py
```

The demo allows users to upload an image and view:

- predicted GPS coordinate
- map visualization
- top-k candidate grid cells
- top-1 grid probability
- top-k probability mass
- prediction entropy

## FastAPI Inference Service

Run the API server:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Example request:

```bash
curl -X POST "http://localhost:8000/predict?top_k=5" \
  -F "file=@examples/images/example_01.jpg"
```

Example response format:

```json
{
  "latitude": 39.9520,
  "longitude": -75.1930,
  "confidence": 0.0863,
  "entropy": 3.42,
  "top_k_cells": [
    {
      "cell_id": 73,
      "probability": 0.0863,
      "latitude": 39.9519,
      "longitude": -75.1932
    }
  ]
}
```

In the API output, `confidence` is kept for compatibility, but it should be interpreted as the top-1 grid probability rather than a calibrated GPS accuracy score.

## Docker Usage

Build the Docker image:

```bash
docker build -t img2gps .
```

Run the FastAPI service:

```bash
docker run -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/checkpoints:ro \
  img2gps
```

Or use Docker Compose:

```bash
docker compose up --build
```

## Method Overview

The model follows a structured geolocation pipeline:

```text
Input image
    ↓
Image preprocessing and normalization
    ↓
ResNet-50 visual backbone
    ↓
10 by 10 grid-cell classification
    ↓
Soft top-k grid decoding
    ↓
Predicted latitude and longitude
```

Direct latitude and longitude regression treats geolocation as two independent continuous values. In contrast, grid-based localization allows the model to first reason about a coarse spatial region and then decode a coordinate from the most likely candidate regions.

During inference, the model does not rely only on the single most likely grid cell. Instead, it uses the top-k predicted grid cells and computes a probability-weighted GPS prediction. This reduces the risk of large errors when multiple nearby locations look visually similar.

## Dataset

The dataset is hosted externally on Hugging Face:

```text
yyss114/CIS-5190-project-6
```

Dataset link:

```text
https://huggingface.co/datasets/yyss114/CIS-5190-project-6
```

The dataset contains campus images paired with GPS coordinates. The expected fields include:

```text
image
latitude
longitude
source
```

The full dataset is not stored in this GitHub repository because of file size and privacy considerations. The repository only includes a small number of anonymized example images under:

```text
examples/images/
```

These images are provided only for quick testing and demonstration.

## Checkpoint Policy

The trained checkpoint is not included in this repository. Large model files should not be committed directly to GitHub.

Expected local checkpoint path:

```text
checkpoints/best.pt
```

The `.gitignore` file excludes checkpoint files by default.

## Training

Train with the default configuration:

```bash
python -m img2gps.train --config configs/default.yaml
```

The training pipeline supports:

- Hugging Face dataset loading
- image augmentation
- 10 by 10 grid label construction
- location-aware validation splitting
- checkpoint saving
- Haversine distance evaluation

Training outputs are saved under:

```text
outputs/
```

These generated files are ignored by Git.

## Evaluation

Evaluate a trained checkpoint:

```bash
python -m img2gps.evaluate \
  --checkpoint checkpoints/best.pt \
  --csv reference/metadata.csv \
  --image-dir reference/images \
  --predictions-out outputs/predictions.csv
```

The main metric is average Haversine distance in meters.

## Error Analysis

After generating a prediction CSV, run:

```bash
python scripts/analyze_errors.py \
  --predictions outputs/predictions.csv \
  --out-dir outputs/error_analysis
```

This can generate:

```text
error_histogram.png
grid_error_heatmap.png
worst_20_predictions.csv
summary.json
```

Error analysis is useful because a low training loss does not always mean better geographic accuracy. Haversine distance should be used as the main evaluation criterion.

## Interpreting Top-k Probabilities

The model predicts a probability distribution over 100 grid cells. The top-1 grid probability is the softmax probability of the most likely grid cell. It measures how concentrated the model's grid prediction is, but it is not a calibrated GPS accuracy score.

Because the final coordinate is decoded using soft top-k prediction, the top-k probability mass is often more informative than the top-1 probability alone. For example, a top-1 probability around 0.08 may still be meaningful if the top-5 cells together contain a large portion of the probability mass and are spatially close.

The final prediction quality should always be evaluated using Haversine distance against ground-truth GPS coordinates.

## Repository Structure

```text
img2gps/
├── app/
│   └── streamlit_app.py
├── api/
│   └── main.py
├── assets/
│   └── demo_streamlit.png
├── configs/
│   └── default.yaml
├── examples/
│   └── images/
│       ├── example_01.jpg
│       ├── example_02.jpg
│       └── README.md
├── scripts/
│   ├── analyze_errors.py
│   └── plot_grid_distribution.py
├── src/
│   └── img2gps/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── evaluate.py
│       ├── inference.py
│       ├── metrics.py
│       ├── model.py
│       ├── predict.py
│       └── train.py
├── tests/
├── checkpoints/
│   └── README.md
├── README.md
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── .gitignore
```

## Example Images

Two small anonymized public-campus example images are included under:

```text
examples/images/
```

These images are intended only for quick testing. They are not a replacement for the full training dataset.

Before adding any new example images, remove EXIF metadata and avoid images containing clear faces, license plates, private spaces, or sensitive information.

## Privacy and Data Policy

This repository does not include:

- full image datasets
- raw GPS records
- model checkpoints
- reports or private documents
- API keys or credentials

The full dataset is hosted externally on Hugging Face. This repository is intended to contain only code, documentation, small demonstration images, and reproducible project configuration.

## Limitations

This model is region-specific. It is designed for a fixed campus-scale area and should not be expected to generalize directly to another city, campus, or geographic region without retraining.

Other limitations include:

- visually similar locations may produce ambiguous predictions
- GPS labels may contain noise
- grid-cell boundaries can introduce discretization errors
- top-k probability is not the same as calibrated location confidence
- model performance should be evaluated with ground-truth GPS labels whenever possible

## Future Work

Potential extensions include:

- adding a calibrated uncertainty model
- incorporating the offset regression head into final decoding
- improving map-based visualization of top-k candidate regions
- adding a lightweight web deployment
- comparing grid-based localization with direct coordinate regression and retrieval-based methods
- expanding the dataset to more diverse campus regions
- adding model cards and dataset cards for clearer documentation

## License

This repository is provided for educational and portfolio purposes. See `LICENSE` for details.

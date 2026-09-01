# Error Analysis

This project evaluates geolocation quality using Haversine distance in meters. In addition to reporting the mean distance, the repository includes tools for inspecting where and why the model fails.

## Generate predictions

```bash
python -m img2gps.evaluate \
  --checkpoint checkpoints/best.pt \
  --dataset yyss114/CIS-5190-project-6 \
  --split train \
  --predictions-out outputs/predictions.csv
```

## Generate error-analysis plots

```bash
python scripts/analyze_errors.py \
  --predictions outputs/predictions.csv \
  --out-dir outputs/error_analysis
```

This produces:

- `error_histogram.png`: distribution of image-level Haversine errors
- `grid_error_heatmap.png`: mean error by ground-truth grid cell
- `worst_20_predictions.csv`: examples with the largest errors
- `summary.json`: mean, median, P75, P90, and maximum error

## Why this matters

For image geolocation, average error alone can hide important failure patterns. A model may perform well around visually distinctive buildings but fail near sidewalks, intersections, or visually similar campus regions. Grid-level error analysis helps identify whether mistakes are geographically concentrated and whether additional data collection should target specific regions.


## Probability interpretation

The top-1 grid probability reported by the demo is a softmax probability over 100 grid cells, not a calibrated correctness score. Error analysis should therefore focus on Haversine error, top-k spatial candidates, entropy, and grid-level error patterns rather than expecting image-classification-style probabilities such as 0.8 or 0.9.


## Probability outputs

For inference demos, top-1 grid probability should not be interpreted as calibrated GPS accuracy. The top-k probability mass is the sum of the probabilities assigned to the candidate grid cells used by soft top-k decoding. Reporting both values is useful because image geolocation can be spatially ambiguous: a low top-1 value may still correspond to a coherent local region when the top-k candidates are nearby.

<div align="center">
  <img src="frontend/public/sir_kalam.png" alt="Project K.A.L.A.M." width="160" />
  <h1>K.A.L.A.M.</h1>
  <h3>Knowledge-Augmented Learning for Atmospheric Metamodels</h3>
  <p><strong>Team DOMinators</strong> — National Runner-up (2nd Place), ISRO Bhartiya Antariksh Hackathon 2025</p>
</div>

---

## Research Overview

**K.A.L.A.M.: Efficient Probabilistic Cloud Nowcasting from INSAT-3DR on a Single Consumer GPU**

K.A.L.A.M. is a two-stage probabilistic cloud-nowcasting framework designed for six-band INSAT-3D/3DR satellite imagery. It predicts three nominal future frames from four observed frames by combining a physics-regularized deterministic forecaster with a conditional residual diffusion model.

The deterministic stage learns corrections to the most recent observed frame through a residual-from-persistence architecture incorporating CBAM attention, FiLM-based motion conditioning, and gated refinement. A differentiable continuity-inspired transport regularizer encourages consistency in predicted image evolution.

The generative stage models the deterministic forecast's residual using conditional diffusion. Multiple samples provide alternative possible future states and pixelwise uncertainty estimates.

On an index-ordered, gap-buffered evaluation of 210 held-out sequences, the deterministic model achieves **SSIM 0.728 and PSNR 22.31 dB**, outperforming persistence, optical flow, and a retrained SimVP baseline.

The diffusion stage recovers **84.3% of target high-frequency spectral power**, compared with 46.6% for the deterministic forecast, and improves the ensemble CRPS relative to a deterministic reference.

The model operates on a single **NVIDIA RTX 3050 laptop GPU with 6 GB VRAM**, demonstrating that probabilistic satellite-image forecasting can be implemented without cluster-scale computing infrastructure.

**Important evaluation distinctions:** Forecast horizons are reported as nominal frame indices because verified acquisition timestamps are unavailable in the derived dataset. The diffusion ensemble provides informative uncertainty rankings but is underdispersed and is not claimed to be absolutely calibrated.

The repository also includes a full-stack application for map-based visualization, satellite-frame animation, inference, and performance reporting.

---

## Table of Contents

1. [Monorepo Layout](#monorepo-layout)
2. [Tech Stack](#tech-stack)
3. [Pipeline At A Glance](#pipeline-at-a-glance)
4. [Visual Results](#visual-results)
5. [ML Architecture](#ml-architecture-python-backend)
6. [Measured Results](#measured-results)
7. [Reproducibility](#reproducibility)
8. [Python-Backend Structure](#python-backend-structure)
9. [Experimental / Future Work](#experimental--future-work)
10. [Known Limitations](#known-limitations)
11. [Full-Stack Quick Start](#full-stack-quick-start)
12. [API Overview](#api-overview-backend)
13. [Frontend Highlights](#frontend-highlights)
14. [Troubleshooting](#troubleshooting)
15. [Citation](#citation)
16. [Acknowledgements & License](#acknowledgements--license)

---

## Monorepo Layout

* `frontend/` — React/Vite application for visualization, animations, maps, and reports.
* `backend/` — Express API, WebSocket server, Cloudflare R2 integration, and MongoDB persistence.
* `Python-Backend/` — Machine-learning research, training, evaluation, diffusion, and serving pipelines.
* `Inference/` — Example satellite-input sequences, predicted images, and evaluation metrics.

## Tech Stack

* **Frontend:** React 19, Vite, Tailwind CSS, React Router, Lucide React.
* **Backend:** Node.js, Express.js, WebSocket, Multer, MongoDB, Mongoose.
* **Storage:** Cloudflare R2 using AWS SDK v3.
* **Machine Learning:** Python, PyTorch 2.6, CUDA 12.4, OpenCV, NumPy, scikit-image.
* **Model Serving:** FastAPI and ONNX.
* **Architecture:** Residual encoder–decoder, CBAM, FiLM, conditional DDPM, DDIM sampling.

---

## Pipeline At A Glance

```mermaid
flowchart LR
    A["INSAT-3D/3DR<br/>Six spectral bands"] --> B["Preprocessing<br/>Normalization and resizing"]
    B --> C["Four observed frames<br/>T-3 to T"]
    C --> D["Deterministic forecaster<br/>Residual + CBAM + FiLM"]
    D --> E["Deterministic forecast<br/>T+1 to T+3"]
    E --> F["Conditional residual diffusion<br/>DDIM sampling"]
    F --> G["Probabilistic samples<br/>and uncertainty maps"]
    E --> H["Evaluation and visualization"]
    G --> H
    H --> I["FastAPI / Express"]
    I --> J["React dashboard"]
```

### Multi-Spectral Representation

K.A.L.A.M. processes six INSAT spectral bands:

| Band | Description         | Atmospheric Information                      |
| ---- | ------------------- | -------------------------------------------- |
| TIR1 | Thermal Infrared 1  | Cloud-top temperature and thermal structure  |
| TIR2 | Thermal Infrared 2  | Complementary thermal information            |
| WV   | Water Vapour        | Atmospheric moisture distribution            |
| VIS  | Visible             | Cloud morphology and boundaries              |
| MIR  | Mid-Infrared        | Additional cloud and surface characteristics |
| SWIR | Short-Wave Infrared | Complementary spectral information           |

Each observation is represented as a six-channel tensor.

The forecasting task is:

**Input:** Four consecutive six-band observations.

**Output:** Three predicted six-band observations.

The model therefore processes:

`[4, 6, 128, 128] → [3, 6, 128, 128]`

The output indices T+1, T+2, and T+3 indicate future positions in the derived sequence. They should not be interpreted as verified 15-, 30-, and 45-minute horizons.

---

## Visual Results

Representative future-frame predictions are available in [`Inference/output/sequence_0010`](Inference/output/sequence_0010).

<div align="center">
  <img src="Inference/output/sequence_0010/predicted_t5_IMG_TIR1.png" alt="Predicted future frame 1 — TIR1" width="30%" />
  <img src="Inference/output/sequence_0010/predicted_t6_IMG_TIR1.png" alt="Predicted future frame 2 — TIR1" width="30%" />
  <img src="Inference/output/sequence_0010/predicted_t7_IMG_TIR1.png" alt="Predicted future frame 3 — TIR1" width="30%" />
</div>

Training and model-comparison figures:

<div align="center">
  <img src="Python-Backend/results/fig_training_curves.png" alt="Training Curves" width="46%" />
  <img src="Python-Backend/results/fig_sota_comparison.png" alt="Model Comparison" width="46%" />
</div>

Additional results, including uncertainty analysis, model ablations, spectral fidelity, and qualitative forecasts, are available in [`Python-Backend/results/`](Python-Backend/results/).

---

## ML Architecture (Python-Backend)

K.A.L.A.M. combines a deterministic forecasting model with conditional residual diffusion.

### Stage 1 — Deterministic Forecaster

**Implementation:** `src/pidm_model.py`

**Model:** `PIConditionalForecaster`

The deterministic architecture contains:

* A multi-scale input-processing stem.
* A four-level residual encoder.
* GroupNorm and GELU activation functions.
* CBAM channel and spatial attention.
* A 512-channel bottleneck.
* A skip-connected decoder.
* A motion-conditioned FiLM meta-learner.
* A residual-from-persistence prediction head.
* A gated iterative refinement module.

Rather than directly predicting future satellite frames, the model estimates corrections to the last observed frame.

The prediction is formulated as:

$$
\hat{y}_{t+k}=x_t+g_k\Delta_\theta(X)
$$

where:

* \(x_t\) is the last observed satellite frame.
* \(\Delta_\theta(X)\) is the learned residual correction.
* \(g_k\) is a learnable horizon-specific gain.
* \(k \in \{1,2,3\}\) represents the nominal forecast index.

The gain is initialized to zero, allowing training to begin from the persistence baseline.

**Deterministic model size:** Approximately 43.2 million parameters.

### Stage 2 — Conditional Residual Diffusion

**Implementation:** `src/pidm_diffusion.py`

The second stage models the residual between the deterministic prediction and ground truth.

$$
r_0=Y-\hat{Y}
$$

A conditional denoising diffusion probabilistic model learns the residual distribution conditioned on the observed frames and deterministic predictions.

The diffusion architecture includes:

* A conditional U-Net denoiser.
* Diffusion timestep embeddings.
* Residual-noise prediction.
* DDPM training.
* DDIM sampling for accelerated inference.
* Ensemble-based pixelwise uncertainty estimation.

The model uses 1,000 training diffusion steps.

For spectral-fidelity evaluation, inference uses 25 DDIM steps.

The separate uncertainty experiment uses 20 DDIM steps and an ensemble of 12 samples.

Given an ensemble of predictions, pixelwise uncertainty is estimated through the standard deviation across ensemble members.

The resulting uncertainty is useful for ranking relatively difficult predictions. However, the measured ensemble is underdispersed, so its raw standard deviation is not presented as an absolutely calibrated prediction interval.

**Diffusion model size:** Approximately 3.7 million parameters.

**Combined system:** Approximately 47.0 million parameters.

### Physics-Informed Regularization

**Implementation:** `src/pidm_losses.py`

The deterministic model incorporates a differentiable continuity-inspired transport regularizer.

The residual is based on:

$$
\frac{\partial I}{\partial t}+\nabla\cdot(I\mathbf{u})
$$

where:

* \(I\) represents satellite-image intensity.
* \(\mathbf{u}\) is an estimated two-dimensional velocity field.
* The velocity field is obtained using a differentiable Lucas–Kanade estimator.

The regularizer encourages consistency between predicted intensity evolution and estimated image transport.

**This is a soft image-transport constraint, not a demonstration of atmospheric mass conservation.**

The complete training objective combines several terms.

| Loss Component                  | Weight |
| ------------------------------- | -----: |
| L1                              |    1.0 |
| MSE                             |    0.6 |
| SSIM                            |    1.0 |
| MS-SSIM, fine-tuning            |    2.0 |
| Multi-scale gradient            |   0.15 |
| Temporal consistency            |   0.15 |
| Total variation                 |   0.02 |
| Advection consistency           |   0.10 |
| Continuity-inspired regularizer |   0.03 |
| SSL flip consistency            |   0.30 |

The horizon-specific training weights are:

`T+1: 0.6 | T+2: 1.0 | T+3: 1.4`

### Training Configuration

Training uses AdamW optimization, gradient clipping, mixed precision, and learning-rate scheduling.

The research pipeline incorporates:

1. Initial deterministic model training.
2. Augmentation and exponential-moving-average refinement.
3. Additional continuity-regularized fine-tuning.
4. Conditional residual diffusion training.
5. Test-time augmentation for deterministic inference.

The training scripts and their configurable parameters are available in [`Python-Backend/src/`](Python-Backend/src/).

---

## Measured Results

### Evaluation Protocol

The primary deterministic evaluation uses **210 held-out sequences**.

The dataset is partitioned into ordered training, validation, and testing subsets using a seven-window buffer between partitions.

This prevents overlapping windows from sharing frames across the train, validation, and test partitions.

The evaluation subsets contain:

| Partition  | Sequences |
| ---------- | --------: |
| Training   |     1,042 |
| Validation |       223 |
| Testing    |       210 |

The split follows the stored sequence indices. Source-level acquisition timestamps are unavailable in the derived cache.

Because neighboring test windows overlap, individual test-window scores are not statistically independent.

### Deterministic Forecasting Performance

The following results are calculated over the 210 held-out sequences.

| Method         |    PSNR ↑ |    SSIM ↑ |     MAE ↓ |    RMSE ↓ | LPIPS ↓ |
| -------------- | --------: | --------: | --------: | --------: | ------: |
| Persistence    |     20.12 |     0.630 |     0.081 |     0.130 |   0.077 |
| Optical Flow   |     19.58 |     0.587 |     0.085 |     0.135 |   0.112 |
| SimVP          |     21.00 |     0.648 |     0.072 |     0.126 |   0.073 |
| **K.A.L.A.M.** | **22.31** | **0.728** | **0.059** | **0.093** |   0.179 |

K.A.L.A.M. achieves:

* **+0.099 absolute SSIM** over persistence.
* **+0.142 absolute SSIM** over optical flow.
* **+0.080 absolute SSIM** over retrained SimVP.
* Approximately **27% reduction in MAE** relative to persistence.
* Approximately **28% reduction in RMSE** relative to persistence.

The per-window Wilcoxon signed-rank test reports:

`p ≈ 3.3 × 10⁻³⁶`

Because the held-out windows overlap, this p-value is descriptive and should not be interpreted as evidence from 210 statistically independent observations.

The deterministic model has a higher LPIPS value than the baselines. Diffusion improves spectral power recovery, but a corresponding diffusion-sample LPIPS improvement has not been established by the reported experiment.

### Forecast Performance by Nominal Lead Index

| Forecast Index | SSIM ↑ | PSNR ↑ |
| -------------- | -----: | -----: |
| T+1            |  0.804 |  24.00 |
| T+2            |  0.725 |  22.06 |
| T+3            |  0.657 |  20.87 |

Forecasting accuracy decreases as the prediction index increases.

These indices describe future frames in the derived sequence, not verified minute-based lead times.

### Performance by Spectral Band

| Band | SSIM ↑ | PSNR ↑ | Persistence SSIM |
| ---- | -----: | -----: | ---------------: |
| TIR1 |  0.631 |  21.73 |            0.461 |
| TIR2 |  0.751 |  22.20 |            0.697 |
| WV   |  0.690 |  21.85 |            0.511 |
| VIS  |  0.765 |  22.53 |            0.684 |
| MIR  |  0.764 |  22.68 |            0.681 |
| SWIR |  0.770 |  22.86 |            0.742 |

The largest absolute improvements over persistence occur in:

* **TIR1:** Approximately +0.170 SSIM.
* **WV:** Approximately +0.178 SSIM.

### Component Ablation Study

**Results:** `results/ablation_full.json`

| Configuration                     | Epochs | SSIM ↑ |  ΔSSIM | PSNR ↑ |
| --------------------------------- | -----: | -----: | -----: | -----: |
| Full model                        |     45 |  0.707 |      — |  21.89 |
| Without residual-from-persistence |     45 |  0.220 | −0.487 |  11.55 |
| Without motion meta-learner       |     45 |  0.630 | −0.077 |  20.12 |
| Without gated refinement          |     28 |  0.694 | −0.013 |  22.22 |
| Without physics regularizer       |     28 |  0.697 | −0.010 |  21.98 |

The results identify residual-from-persistence prediction as a major architectural contribution.

Removing the motion meta-learner also substantially reduces performance.

The gated-refinement and physics-regularizer variants were trained for fewer epochs. Therefore, their performance differences cannot be attributed exclusively to removing those components.

### Diffusion Spectral Fidelity

**Results:** `results/tgrs_analysis.json`

**Evaluation subset:** 70 held-out sequences.

The spectral experiment compares high-frequency power against the ground-truth image spectrum.

| Forecast                     | High-Frequency Power Ratio |
| ---------------------------- | -------------------------: |
| Deterministic K.A.L.A.M.     |                      0.466 |
| Conditional diffusion sample |                  **0.843** |

Diffusion recovers substantially more high-frequency spectral power than the deterministic prediction.

The ratio describes spectral-power agreement rather than direct pixelwise reconstruction accuracy.

### Probabilistic Forecasting and Uncertainty

**Results:** `results/uncertainty_calibration.json`

**Evaluation subset:** 42 held-out sequences.

**Ensemble size:** 12.

**DDIM sampling:** 20 steps.

| Quantity                        |             Value |
| ------------------------------- | ----------------: |
| Diffusion ensemble CRPS         |            0.0462 |
| Deterministic reference score   |            0.0577 |
| Relative score reduction        | Approximately 20% |
| Spread–skill correlation        |             0.612 |
| Binned spread–error correlation |             0.945 |
| Mean ensemble spread            |             0.056 |
| Mean observed RMSE              |             0.087 |

The ensemble demonstrates informative uncertainty ranking.

Higher predicted uncertainty generally corresponds to larger observed errors.

However, the ensemble spread is smaller than the observed RMSE.

Consequently, the current predictions should be considered **underdispersed rather than absolutely calibrated**.

The reported uncertainty values are most appropriately interpreted as relative indicators of forecasting difficulty.

### Computational Efficiency

**Benchmark:** `results/efficiency_benchmark.json`

**GPU:** NVIDIA GeForce RTX 3050 Laptop GPU, 6 GB VRAM.

**Input:** `[1, 4, 6, 128, 128]`

**Output:** `[1, 3, 6, 128, 128]`

| Measurement                          |       Value |
| ------------------------------------ | ----------: |
| Deterministic model parameters       |      43.21M |
| Diffusion model parameters           |       3.73M |
| Total parameters                     |      46.95M |
| Deterministic inference              |     0.684 s |
| Deterministic inference with 4× TTA  |     2.730 s |
| One diffusion sample, DDIM-25        |     3.859 s |
| Deterministic + one diffusion sample | **4.543 s** |
| Deterministic peak GPU memory        |     1.87 GB |
| Diffusion-stage peak GPU memory      |     0.80 GB |

**Important:** The measured 4.54-second latency corresponds to deterministic inference followed by one 25-step diffusion sample.

It does **not** represent the execution time of a complete 12-member diffusion ensemble.

The complete ensemble requires additional sampling, and its end-to-end latency is not established by the published benchmark.

---

## Reproducibility

### Dataset

The research uses INSAT-3DR Imager Level-1C multispectral imagery obtained through ISRO's MOSDAC data services.

The six spectral channels are:

`VIS, SWIR, MIR, WV, TIR1, TIR2`

The original imagery is approximately `1616 × 1737` pixels.

The derived dataset contains **1,489 overlapping seven-frame examples**, each comprising:

* Four observed frames.
* Three prediction targets.
* Six spectral channels.

Each band is normalized to `[0,1]`, and the images are resized to `128 × 128`.

The approximate effective sampling scale is 48 km per pixel.

The current evaluation is therefore oriented toward mesoscale image patterns rather than individual convective cells.

### Temporal Metadata

The derived dataset preserves ordered sequence indices but does not retain authenticated UTC acquisition timestamps.

Some filenames contain placeholder timestamps associated with nominal spacing.

These placeholders are not treated as verified acquisition times.

Accordingly, all forecasting results are reported using nominal frame indices.

### Data Splitting

**Implementation:** `src/data_split.py`

The dataset uses ordered 70/15/15 partitioning with seven-window buffers between splits.

This prevents shared frames across training, validation, and testing partitions.

The existing split does not establish generalization across independent acquisition days or seasons.

### Availability

The repository provides:

* Research implementation.
* Model architecture definitions.
* Training scripts.
* Evaluation scripts.
* Baseline implementations.
* Configuration options.
* Saved numerical results.
* Generated evaluation figures.
* Example inference outputs.

The complete derived training cache and trained research checkpoints are not publicly distributed in the repository.

Raw INSAT Level-1C products are available through the relevant ISRO/MOSDAC access mechanisms and are not redistributed here.

### Research Pipeline

```bash
cd Python-Backend

pip install -r requirements.txt

# Preprocess source sequences into the training cache
python src/prep_cache.py

# Train the deterministic forecaster
python src/train_pidm.py --epochs 260 --batch 12 --augment --ema 0.999 \
       --w_ssl 0.3 --w_pde 0.03 --w_msssim 2.0 \
       --lead_weights 0.6,1.0,1.4

# Train conditional residual diffusion
python src/train_diffusion.py --epochs 40

# Evaluate deterministic forecasting
python src/evaluate_and_plot.py

# Train the SimVP baseline
python src/baseline_simvp.py --epochs 70

# Generate model comparisons
python src/make_sota_comparison.py

# Evaluate probabilistic uncertainty
python src/uncertainty_calibration.py

# Evaluate spectral characteristics
python src/tgrs_analysis.py

# Run ablation experiments
python src/run_ablations.py --epochs 28
python src/ablation_eval.py

# Benchmark computational efficiency
python src/efficiency_benchmark.py
```

These commands document the available research workflow. Reproduction requires access to the appropriately prepared source dataset and compatible model-training environment.

---

## Python-Backend Structure

```text
Python-Backend/
├── requirements.txt
├── src/
│   ├── pidm_model.py
│   ├── pidm_diffusion.py
│   ├── pidm_losses.py
│   ├── train_pidm.py
│   ├── train_diffusion.py
│   ├── data_split.py
│   ├── evaluate_and_plot.py
│   ├── baseline_simvp.py
│   ├── uncertainty_calibration.py
│   ├── tgrs_analysis.py
│   ├── run_ablations.py
│   └── efficiency_benchmark.py
├── serving/
├── experimental/
└── results/
```

### Model Serving

The `serving/` directory includes:

* `main.py` — FastAPI application.
* `inference.py` — Inference execution and prediction generation.
* `train4.py` — Earlier EnhancedUNet implementation.
* `train5.py` — CBAM-based alternative implementation.
* `conv_2_onnx.py` — ONNX export.
* `npy2png.py` — Data conversion.
* `stack.py` — Multi-band tensor preparation.
* `pt2np.py` — Tensor-to-NumPy conversion.
* `view.py` — Data visualization.
* `test1.py`, `test2.py` — Inference tests.

The research model in `src/` and the application-serving implementations are maintained as separate development tracks.

The published research metrics correspond to the evaluated research pipeline, not automatically to every model variant in `serving/`.

---

## Experimental / Future Work

The `experimental/` directory contains exploratory model-development work that is not included in the main GRSL evaluation.

### Transformer-Based Forecaster

`train6.py` explores a larger architecture incorporating:

* Transformer-based temporal modeling.
* Residual convolutional blocks.
* Time conditioning.
* Additional forecasting losses.

This model has not been validated under the complete research evaluation protocol.

### Stable Diffusion Experiments

`stabdiff.py` explores adapting a pretrained diffusion architecture to multispectral satellite imagery.

`fine_tune.py` contains an additional experimental fine-tuning implementation.

These experiments are retained for research transparency but do not support the reported GRSL results.

### Future Research Directions

Potential extensions include:

* Independent-day and multi-season evaluation.
* Acquisition-time-aware forecasting.
* Improved uncertainty calibration.
* Additional spectral and perceptual evaluation of diffusion outputs.
* Higher-resolution satellite nowcasting.
* Comparison against additional geostationary satellite forecasting architectures.
* Improved computational efficiency for ensemble inference.

---

## Known Limitations

The current study has several documented limitations.

**1. Acquisition-period coverage**

The evaluation uses data from one collection period. Generalization across different days, seasons, and atmospheric conditions remains untested.

**2. Temporal metadata**

Authenticated acquisition timestamps are not available in the derived dataset. Forecast indices therefore do not establish verified real-time forecasting horizons.

**3. Spatial resolution**

The effective sampling scale of approximately 48 km per pixel captures relatively large-scale patterns but does not resolve individual convective cells.

**4. Probabilistic calibration**

The diffusion ensemble is underdispersed. Its uncertainty values are informative for relative ranking but are not demonstrated to be calibrated absolute error estimates.

**5. Physics regularization**

The continuity-inspired objective encourages image-transport consistency. It does not prove atmospheric mass conservation or establish superior physical consistency over every baseline.

**6. Ablation budgets**

Two component-removal experiments use fewer training epochs than the full model, limiting causal interpretation of their differences.

**7. Perceptual evaluation**

The deterministic model has weaker LPIPS performance than the compared baselines. Diffusion demonstrates stronger high-frequency spectral-power recovery, but diffusion LPIPS improvements are not established in the reported results.

**8. Ensemble inference**

The measured 4.54-second latency is for deterministic prediction plus one diffusion sample, not the complete 12-member ensemble.

**9. Reproduction resources**

The complete derived cache and trained research checkpoints are not included in the public repository.

---

## Full-Stack Quick Start

### Prerequisites

* Node.js 18+
* Python 3.10+
* MongoDB
* Cloudflare R2 credentials
* PyTorch-compatible environment

### Backend Setup

```bash
cd backend

npm install

npm run dev
```

The Express backend runs on port `3000`.

The WebSocket server runs on port `3001`.

### Frontend Setup

```bash
cd frontend

npm install

npm run dev
```

The frontend runs on port `5173`.

### Python Setup

```bash
cd Python-Backend

pip install -r requirements.txt
```

### Environment Variables

Create `backend/.env` with the required configuration.

```env
PORT=3000
NODE_ENV=development

MONGO_URL=mongodb://localhost:27017/project-kalam

R2_ENDPOINT=
R2_ACCESS_KEY_ID=
R2_SECRET_ACCESS_KEY=
R2_BUCKET_NAME=
R2_PUBLIC_URL=
```

### Local Paths To Update

Some backend components contain development-machine-specific file paths.

Update these paths before running the application on another machine.

Relevant files include:

* `backend/controllers/modelTest.controller.js`
* `backend/app.js`

The Python inference script and prediction-output directories must point to locations available on the host system.

### Application Workflow

1. Start the Python inference service when required.
2. Start the Express backend and WebSocket server.
3. Start the React frontend.
4. Open the application in a browser.
5. Select available satellite sequences.
6. Generate and visualize predictions.
7. Explore forecast metrics and imagery.

---

## API Overview (backend)

**Base URL:** `http://localhost:3000`

### Core Endpoints

| Endpoint                                     | Description               |
| -------------------------------------------- | ------------------------- |
| `GET /`                                      | Service information       |
| `GET /api/prediction-images/<relative-path>` | Retrieve predicted images |

### Cloudflare R2 Endpoints

| Endpoint            | Description                |
| ------------------- | -------------------------- |
| `POST /upload`      | Upload files               |
| `POST /test-upload` | Test file uploads          |
| `GET /health`       | Service and storage status |

### Prediction Endpoints

| Endpoint                   | Description               |
| -------------------------- | ------------------------- |
| `POST /folder-path`        | Trigger Python validation |
| `POST /predict-frames`     | Request forecast results  |
| `GET /available-sequences` | List available sequences  |

### FastAPI Research Serving

The Python serving application exposes:

`POST /run_inference`

The endpoint supports the model-serving workflow using available input sequences and configured checkpoints.

---

## Frontend Highlights

The React application provides:

* A multispectral satellite-image viewer.
* Frame-by-frame satellite animation.
* Forecast and ground-truth comparison.
* Interactive band selection.
* Model-inference execution.
* Performance reporting.
* Map-based satellite visualization.
* WebSocket-based processing updates.
* Responsive dashboard components.

### Frontend Structure

```text
frontend/
├── src/
│   ├── App.jsx
│   ├── components/
│   ├── pages/
│   ├── hooks/
│   ├── libs/
│   └── styles/
└── public/
```

### Main Components

* `SatelliteAnimationPage.jsx` — Satellite-frame visualization.
* `ModelTestAndTerminalPreview.jsx` — Model testing and logs.
* `MapComponent.jsx` — Map integration.
* `ReportPage.jsx` — Forecast results and reporting.
* `EnhancedFileBrowser.jsx` — File selection.
* `FileAssignmentSection.jsx` — Data assignment workflow.

---

## Troubleshooting

**Images not loading**

Verify the prediction-output directory and corresponding backend file paths.

**WebSocket connection errors**

Ensure the WebSocket service is running on the configured port.

**Cloudflare R2 upload failures**

Check the R2 endpoint, access keys, bucket name, and permissions.

**MongoDB errors**

Verify the MongoDB connection string and database availability.

**Python inference errors**

Ensure the correct Python environment, model checkpoint, input directory, and required dependencies are configured.

**CUDA errors**

Verify PyTorch compatibility with the installed GPU driver and CUDA environment.

**Missing research data**

The complete research cache is not distributed through this repository. Reproducing training requires obtaining the relevant satellite data and preparing the derived sequences.

---

## Citation

### Research Manuscript

**K.A.L.A.M.: Efficient Probabilistic Cloud Nowcasting from INSAT-3DR on a Single Consumer GPU**

Authors:

Nisarg Vashi, Heli Shah, Hensi Patel, Krish Vaghasia, Aparna Kumari, and Madhukant Patel.

Manuscript prepared for submission to *IEEE Geoscience and Remote Sensing Letters*.

This manuscript has not been identified as an accepted or published IEEE article.

### BibTeX

```bibtex
@misc{vashi2026kalam,
  title  = {K.A.L.A.M.: Efficient Probabilistic Cloud
            Nowcasting from INSAT-3DR on a Single
            Consumer GPU},
  author = {Vashi, Nisarg and
            Shah, Heli and
            Patel, Hensi and
            Vaghasia, Krish and
            Kumari, Aparna and
            Patel, Madhukant},
  year   = {2026},
  note   = {Research manuscript prepared for
            submission to IEEE Geoscience and
            Remote Sensing Letters},
  url    = {https://github.com/info-gallary/K.A.L.A.M.}
}
```

### Project Recognition

K.A.L.A.M. was developed by **Team DOMinators**, National Runner-up (2nd Place) at the **ISRO Bhartiya Antariksh Hackathon 2025**.

---

## Acknowledgements & License

The authors acknowledge the Indian Space Research Organisation (ISRO), the National Remote Sensing Centre (NRSC), Hyderabad, and the organizers of Bhartiya Antariksh Hackathon 2025 for providing the opportunity and access to satellite data.

The authors also acknowledge Nirma University for institutional support.

Claude (Anthropic) was used for language refinement in the research manuscript. Responsibility for the scientific content, methodology, experiments, and conclusions remains with the authors.

**License:** MIT.

See [`LICENSE`](LICENSE) for details.

---

**K.A.L.A.M. — Knowledge-Augmented Learning for Atmospheric Metamodels**

*Efficient Probabilistic Cloud Nowcasting from INSAT-3DR on a Single Consumer GPU.*

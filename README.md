<div align="center">

# LEAD-AI: DPWH Infrastructure Risk Profiling

An unsupervised machine learning system designed to discover risk archetypes and operational anomalies in Philippine public infrastructure projects using from-scratch clustering algorithms.

**Project Webpage:** [https://randomchaffee.github.io/lead-ai/](https://randomchaffee.github.io/lead-ai/) *(To be changed to webpage URL once deployed)*

</div>

---

## Table of Contents

- [LEAD-AI: DPWH Infrastructure Risk Profiling](#lead-ai-dpwh-infrastructure-risk-profiling)
  - [Table of Contents](#table-of-contents)
  - [About the Project](#about-the-project)
  - [System Architecture](#system-architecture)
  - [Dataset \& Preprocessing](#dataset--preprocessing)
  - [Engineered Feature Metrics](#engineered-feature-metrics)
  - [Machine Learning Methods](#machine-learning-methods)
  - [Configuration \& Hyperparameters](#configuration--hyperparameters)
  - [Project Structure](#project-structure)
    - [Backend Engine (Python)](#backend-engine-python)
    - [Frontend Application (React + TypeScript)](#frontend-application-react--typescript)
  - [Installation \& Setup](#installation--setup)
    - [Prerequisites](#prerequisites)
    - [1. Clone the Repository](#1-clone-the-repository)
    - [2. Configure Python Virtual Environment](#2-configure-python-virtual-environment)
    - [3. Prepare the Dataset](#3-prepare-the-dataset)
  - [Running the Pipeline](#running-the-pipeline)
    - [Running Individual Model Evaluations](#running-individual-model-evaluations)
    - [Pipeline Outputs](#pipeline-outputs)
  - [Running the Frontend](#running-the-frontend)
  - [Task Log](#task-log)
  - [References \& Disclosures](#references--disclosures)

---

## About the Project

Public infrastructure projects in the Philippines face persistent challenges with budget overruns, project abandonment, and contractual delays. With over 248,000 project records available across the country, identifying patterns of systemic risk manually is impractical.

**LEAD-AI** applies **unsupervised machine learning** to analyze DPWH infrastructure contract data without predefined class labels. By projecting infrastructure projects into a multi-dimensional metric space, the system uses distance-based clustering algorithms to partition projects into distinct **risk profiles** (such as low-risk routine maintenance vs. high-exposure delayed contracts) and isolate anomalous cases for prioritized audit review.

All preprocessing, model training, and evaluation metrics are implemented entirely from scratch using Python standard libraries and NumPy, strictly adhering to course constraints prohibiting high-level machine learning libraries (`scikit-learn`, `pandas`, `scipy`).

---

## System Architecture

```text
[ BetterGov.ph Dataset ] 
         |
         v
[ Step 1: load_and_sample.py ] -------- Filter 2021-2023 & Active Contractors (1,600 rows)
         |
         v
[ Step 2: feature_quantifying.py ] ---- Compute 5 Metrics & Min-Max Normalization (1600x5 Matrix)
         |
         |---> [ kmeans.py ] ---------- Centroid-based partitioning (Lloyd's Algorithm)
         |---> [ kmedoids.py ] -------- Partitioning Around Medoids (PAM)
         |---> [ dbscan.py ] ---------- Density-based clustering & anomaly/noise detection (-1)
         |
         v
[ Step 3: compare_all.py ] ------------ Master comparison workflow running all 3 algorithms
         |                              (Silhouette Score, Davies-Bouldin Index, JSON export)
         v
[ Visualizations & Outputs ] ---------- Comparative 3-panel scatter plots (methods_comparison.png)
         |                              and individual model evaluation charts
         v
[ Static Webpage Deployment ] --------- React + TypeScript Single-Page Application on GitHub Pages
```

---

## Dataset & Preprocessing

- **Source:** DPWH Infrastructure Transparency Dataset via [BetterGov.ph Open Data Portal](https://data.bettergov.ph/).
- **Original Scale:** 248,220 contract records.
- **Filtering Rules:**
  1. **Year Filter:** Restrict to projects starting in **2021, 2022, or 2023** to focus on mature project cycles unaffected by 2020 pandemic lockdowns.
  2. **Contractor Activity:** Retain projects where the contractor appears at least **3 times** to ensure sufficient track record data.
  3. **Sampling:** A random sample of **1,600 projects** is selected using a fixed random seed (`42`) for experimental reproducibility.

> **Dataset Setup Guide:** Due to GitHub's file size limits, the raw dataset is not stored in this repository. Follow the step-by-step instructions in [READ-FIRST.md](backend/data/raw/READ-FIRST.md) to download and convert the data locally.

---

## Engineered Feature Metrics

From the 23 raw columns, 5 numerical features are computed to represent each project:

| Feature Metric | Source Column(s) | Description |
| :--- | :--- | :--- |
| **`budget`** | `budget` | Total contract budget in Philippine Pesos (PHP). |
| **`duration_days`** | `startDate`, `completionDate` | Calendar duration between project commencement and completion. |
| **`contractor_count`** | `contractor` | Total volume of infrastructure projects awarded to this contractor. |
| **`contractor_avg_budget`** | `contractor`, `budget` | Mean allocation per project handled by this contractor. |
| **`contractor_delay_rate`** | `contractor`, `completionDate`, `status` | Proportion of contractor projects with recorded delays or termination. |

All features are normalized to the range `[0.0, 1.0]` using manual Min-Max scaling:

$$x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

---

## Machine Learning Methods

Three from-scratch unsupervised clustering methods are implemented and compared:

1. **K-Means Clustering:** Centroid-based algorithm that partitions observations into spherical clusters by minimizing squared Euclidean distance to arithmetic cluster centers.
2. **K-Medoids (PAM):** Medoid-based partitioning using Kaufman & Rousseeuw's Partitioning Around Medoids. Cluster centers are restricted to actual observed data points, offering greater resistance to extreme budget outliers.
3. **DBSCAN:** Density-based spatial clustering that discovers clusters of arbitrary shapes and explicitly labels sparse points as **Noise ($-1$)** to flag high-risk anomalies.

---

## Configuration & Key parameters

The project utilizes several key parameters across the data processing and modeling stages. Users can modify these values in their respective backend scripts to experiment with different sample distributions, cluster granularities, or density sensitivities:

| Parameter | Default Value | Location | Description |
| :--- | :--- | :--- | :--- |
| **`SAMPLE_SIZE`** | `1600` | `backend/src/load_and_sample.py` | Target random sample size drawn from the filtered DPWH dataset. |
| **`NUM_OF_PROJS`** | `3` | `backend/src/load_and_sample.py` | Minimum project appearances required for a contractor to be included. |
| **`valid_years`** | `['2021', '2022', '2023']` | `backend/src/load_and_sample.py` | Allowed contract start years to ensure project lifecycle maturity. |
| **`random.seed` / `np.random.seed`** | `42` | `load_and_sample.py`, `kmeans.py` | Random seed ensuring deterministic sampling and initial centroid selection. |
| **`k` (K-Means & K-Medoids)** | `4` | `kmeans.py`, `kmedoids.py`, `compare_all.py` | Number of clusters to partition the project observations into. |
| **`eps` (DBSCAN)** | `0.15` | `dbscan.py`, `compare_all.py` | Maximum Euclidean distance radius for neighborhood reachability. |
| **`min_pts` (DBSCAN)** | `5` | `dbscan.py`, `compare_all.py` | Minimum neighbor count (including core point) to form a dense cluster. |

---

## Project Structure

### Backend Engine (Python)

```text
backend/
|-- data/
|   |-- raw/
|   |   |-- convert_parquet.py     # Parquet to CSV conversion script
|   |   \-- READ-FIRST.md          # Dataset download and setup guide
|   \-- processed/                 # Directory for processed data artifacts
|-- src/
|   |-- load_and_sample.py         # Step 1: Data extraction and sampling (1,600 rows)
|   |-- feature_quantifying.py     # Step 2: 5-feature calculation and normalization
|   |-- kmeans.py                  # Step 3A: From-scratch K-Means clustering (K=4)
|   |-- kmeans_evaluate.py         # K-Means evaluation script & 2D cluster visualization
|   |-- kmedoids.py                # Step 3B: From-scratch K-Medoids (PAM) clustering (K=4)
|   |-- kmedoids_evaluate.py       # K-Medoids evaluation script & cluster visualization
|   |-- dbscan.py                  # Step 3C: From-scratch DBSCAN density clustering
|   |-- dbscan_evaluate.py         # DBSCAN evaluation script & noise visualization
|   |-- evaluate.py                # Evaluation module (Silhouette & Davies-Bouldin)
|   \-- compare_all.py             # Step 3: Master comparison pipeline for all 3 algorithms
|-- outputs/                       # Exported charts and JSON evaluation outputs
|   |-- comparison_results.json    # Structured JSON metrics for web dashboard
|   |-- methods_comparison.png     # 3-panel comparative scatter plot (Budget vs Duration)
|   \-- kmeans_clusters.png        # Individual K-Means scatter plot
\-- requirements.txt               # Backend dependencies (numpy, matplotlib, pyarrow)
```

### Frontend Application (React + TypeScript)

```text
frontend/
|-- public/                        # Static assets and icons
|-- src/
|   |-- components/                # UI layout and report sections
|   |-- types/                     # TypeScript interfaces for algorithm results
|   |-- App.tsx                    # Main single-page report view
|   \-- main.tsx                   # Frontend entry point
|-- package.json                   # Node.js dependencies
\-- vite.config.ts                 # Vite configuration with GitHub Pages base path
```

---

## Installation & Setup

### Prerequisites

- Python 3.10+
- Node.js 18+ and `npm`

### 1. Clone the Repository

```bash
git clone https://github.com/randomchaffee/lead-ai.git
cd lead-ai
```

### 2. Configure Python Virtual Environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements.txt
```

*(If you are a linux user)*:
```bash
python3 -m venv venv
source venv/bin/activate 
pip install -r backend/requirements.txt
```

### 3. Prepare the Dataset

Follow the download and conversion steps in [backend/data/raw/READ-FIRST.md](backend/data/raw/READ-FIRST.md).

---

## Running the Pipeline

Ensure your virtual environment is active before running backend scripts:

```powershell
# 1. Run data extraction and sampling (generates 1,600 sampled rows)
python backend/src/load_and_sample.py

# 2. Run feature extraction and normalization (outputs 1600x5 normalized matrix)
python backend/src/feature_quantifying.py

# 3. Run the Master Comparison Pipeline (evaluates all 3 algorithms and exports outputs)
python backend/src/compare_all.py
```

### Running Individual Model Evaluations

You can also run and inspect each algorithm's standalone evaluation and visualization independently:

```powershell
# Run K-Means evaluation (generates backend/outputs/kmeans_clusters.png)
python backend/src/kmeans_evaluate.py

# Run K-Medoids (PAM) evaluation
python backend/src/kmedoids_evaluate.py

# Run DBSCAN evaluation
python backend/src/dbscan_evaluate.py
```

### Pipeline Outputs

Running `compare_all.py` generates two artifacts inside `backend/outputs/`:
- **`comparison_results.json`**: Contains cluster counts, Silhouette Scores, Davies-Bouldin Indices, execution runtimes, and cluster distributions formatted for easy integration into the frontend dashboard.
- **`methods_comparison.png`**: A high-resolution 3-panel scatter plot (Budget vs. Duration) visually contrasting the cluster assignments of K-Means, K-Medoids, and DBSCAN side by side.

---

## Running the Frontend

To run the project report page locally:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173/` in your browser.

---

## Task Log

All member contributions across milestones are tracked in [docs/TASK-LOG.md](docs/TASK-LOG.md).

---

## References & Disclosures

- **Dataset:** DPWH Infrastructure Transparency Dataset, [BetterGov.ph](https://data.bettergov.ph/)
- **Course Reference:** CMSC 170 Introduction to Artificial Intelligence Laboratory Handouts (University of the Philippines Los Baños)
- **AI Assistance Disclosure:** All data processing and algorithm implementations are written from scratch by the group using NumPy and Python standard libraries.
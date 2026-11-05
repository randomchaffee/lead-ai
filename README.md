<div align="center">

# LEAD-AI: DPWH Infrastructure Risk Profiling

![Course](https://img.shields.io/badge/CMSC_170-Intro_to_Artificial_Intelligence-8A2BE2?style=for-the-badge)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![React](https://img.shields.io/badge/React_19-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=for-the-badge&logo=typescript&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white)

An unsupervised machine learning system designed to discover risk archetypes and operational anomalies in Philippine public infrastructure projects using clustering algorithms.

**🌐 Project Webpage:** [https://randomchaffee.github.io/lead-ai/](https://randomchaffee.github.io/lead-ai/) *(To be changed once deployed)*

</div>

---

## 📖 Table of Contents

- [About the Project](#about-the-project)
- [System Architecture](#system-architecture)
- [Dataset & Preprocessing](#dataset--preprocessing)
- [Engineered Feature Metrics](#engineered-feature-metrics)
- [Machine Learning Methods](#machine-learning-methods)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Running the Pipeline](#running-the-pipeline)
- [Running the Frontend](#running-the-frontend)
- [Task Log](#task-log)
- [References & Disclosures](#references--disclosures)

---

## About the Project

Public infrastructure projects in the Philippines face persistent challenges with budget overruns, project abandonment, and contractual delays. With over 248,000 project records available across the country, identifying patterns of systemic risk manually is impractical.

**LEAD-AI** applies **unsupervised machine learning** to analyze DPWH infrastructure contract data without predefined class labels. By projecting infrastructure projects into a multi-dimensional metric space, the system uses distance-based clustering algorithms to partition projects into distinct **risk profiles** (such as low-risk routine maintenance vs. high-exposure delayed contracts) and isolate anomalous cases for prioritized audit review.

All preprocessing, model training, and evaluation metrics are implemented entirely from scratch using Python standard libraries and NumPy, strictly adhering to course constraints prohibiting high-level machine learning libraries (`scikit-learn`, `pandas`, `scipy`).

---

## System Architecture

```text
[ BetterGov.ph Dataset ] 
         │
         ▼
[ Step 1: load_and_sample.py ] ── Filter 2021-2023 & Active Contractors (1,600 rows)
         │
         ▼
[ Step 2: feature_quantifying.py ] ── Compute 5 Metrics & Min-Max Normalization (1600x5 Matrix)
         │
         ├───► [ kmeans.py ]     ── Centroid-based partitioning (Lloyd's Algorithm)
         ├───► [ kmedoids.py ]   ── Partitioning Around Medoids (PAM)
         └───► [ dbscan.py ]     ── Density-based clustering & anomaly/noise detection (-1)
         │
         ▼
[ Evaluation & Plotting ] ── Silhouette Score, Davies-Bouldin Index & Matplotlib Charts
         │
         ▼
[ Static Webpage Deployment ] ── React + TypeScript Single-Page Application on GitHub Pages
```

---

## Dataset & Preprocessing

- **Source:** DPWH Infrastructure Transparency Dataset via [BetterGov.ph Open Data Portal](https://data.bettergov.ph/).
- **Original Scale:** 248,220 contract records.
- **Filtering Rules:**
  1. **Year Filter:** Restrict to projects starting in **2021, 2022, or 2023** to focus on mature project cycles unaffected by 2020 pandemic lockdowns.
  2. **Contractor Activity:** Retain projects where the contractor appears at least **3 times** to ensure sufficient track record data.
  3. **Sampling:** A random sample of **1,600 projects (can be changed by user)** is selected using a fixed random seed (`42`) for experimental reproducibility.

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

1. **K-Means Clustering:** Centroid-based algorithm that partitions observations into $k$ spherical clusters by minimizing squared Euclidean distance to cluster centers.
2. **K-Medoids (PAM):** Medoid-based partitioning where cluster centers must be actual observed data points from the dataset, offering greater resistance to extreme budget outliers.
3. **DBSCAN:** Density-based spatial clustering that groups closely packed observations based on radius ($\epsilon$) and minimum points ($\text{minPts}$), explicitly labeling isolated points as **Noise ($-1$)** to flag anomalies.

---

## Project Structure

```text
lead-ai/
├── .github/
│   └── workflows/
│       └── deploy.yml            # Automated GitHub Pages build workflow
├── docs/
│   └── TASK-LOG.md               # Group member contribution and task tracking
├── backend/
│   ├── data/
│   │   ├── raw/
│   │   │   ├── convert_parquet.py # Parquet to CSV conversion script
│   │   │   └── READ-FIRST.md      # Dataset download and setup guide
│   │   └── processed/             # Directory for processed data artifacts
│   ├── src/
│   │   ├── load_and_sample.py     # Step 1: Data extraction and sampling
│   │   └── feature_quantifying.py # Step 2: 5-feature calculation and normalization
│   ├── outputs/                   # Exported charts and JSON evaluation outputs
│   └── requirements.txt           # Minimal backend dependencies (numpy, matplotlib)
├── frontend/
│   ├── public/                    # Static assets and icons
│   ├── src/
│   │   ├── components/            # UI components and section layouts
│   │   ├── types/                 # TypeScript interfaces for algorithm results
│   │   ├── App.tsx                # Main single-page report view
│   │   └── main.tsx               # Frontend entry point
│   ├── package.json               # Node.js dependencies
│   └── vite.config.ts             # Vite configuration with GitHub Pages base path
├── .gitignore                     # Git ignore rules
└── README.md                      # Primary project documentation
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

### 3. Prepare the Dataset

Follow the download and conversion steps in [backend/data/raw/READ-FIRST.md](backend/data/raw/READ-FIRST.md).

---

## Running the Pipeline

Ensure your virtual environment is active before running backend scripts:

```powershell
# 1. Run data extraction and sampling
python backend/src/load_and_sample.py

# 2. Run feature extraction and normalization
python backend/src/feature_quantifying.py
```

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

All member contributions across milestones are recorded in [docs/TASK-LOG.md](docs/TASK-LOG.md).

---

## References & Disclosures

- **Dataset:** DPWH Infrastructure Transparency Dataset, [BetterGov.ph](https://data.bettergov.ph/)
- **Course Reference:** CMSC 170 Introduction to Artificial Intelligence Laboratory Handouts (University of the Philippines Los Baños)
- **AI Assistance Disclosure:** Generative AI tools were consulted for architecture guidance, syntax verification, and debugging. All data processing and algorithm implementations are written from scratch by the group using NumPy and Python standard libraries.
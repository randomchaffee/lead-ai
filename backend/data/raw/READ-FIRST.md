# Dataset Instructions

The dataset files are **not tracked in this repository** due to GitHub's file size limit.
Every group member must download and convert it locally before running any backend scripts.

---

## Step 1: Download the Dataset

1. Go to the [BetterGov.ph Open Data Portal](https://data.bettergov.ph/)
2. Search for **"DPWH"** and open **"DPWH Infrastructure Transparency Dataset"**
3. Download the **Parquet** file (approximately 20.8 MB)
4. Save it inside this folder as:
   `backend/data/raw/dpwh_transparency_data.parquet`

---

## Step 2: Install Required Libraries

Install the project dependencies and the conversion tool:

```powershell
pip install -r backend/requirements.txt
```

---

## Step 3: Run the Conversion Script

Run the provided script to generate the CSV:

```powershell
python backend/data/raw/convert_parquet.py
```

---

## Step 4: Verify

After running the script, you should see:
`backend/data/raw/dpwh_transparency_data.csv` (~102 MB)

Both `dpwh_transparency_data.parquet` and `dpwh_transparency_data.csv` are gitignored — do not commit either file.

---

## Dataset Facts

| Property | Value |
| :--- | :--- |
| Total rows | ~247,000+ records |
| Parquet file size | ~20.8 MB |
| CSV file size | ~102 MB |
| Key columns used | contractId, contractor, budget, startDate, completionDate, status |
| Source | [BetterGov.ph Open Data Portal](https://data.bettergov.ph/) |
# Dataset Instructions

The dataset files are **not tracked in this repository** due to GitHub's file size limit.
Every group member must download and convert the data locally before running backend scripts.

---

## Step 1: Download the Dataset

1. Go to the [BetterGov.ph Open Data Portal](https://data.bettergov.ph/)
2. Search for **"DPWH"** and open **"DPWH Infrastructure Transparency Dataset"**
3. Download the **Parquet** file (approximately 20.8 MB)
4. Save it inside this directory as:
   `backend/data/raw/dpwh_transparency_data.parquet`

---

## Step 2: Setup Virtual Environment & Install Dependencies

It is recommended to run this project inside a local Python virtual environment to avoid installing packages globally:

1. Open PowerShell and navigate to the root repository folder:
   ```powershell
   cd lead-ai
   ```

2. Create a virtual environment:
   ```powershell
   python -m venv venv
   ```

3. Activate the virtual environment:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
   *(A `(venv)` prefix should appear at the start of your terminal prompt).*

4. Install the backend dependencies:
   ```powershell
   python -m pip install -r backend/requirements.txt
   ```

---

## Step 3: Convert Parquet to CSV

Run the conversion script to generate the CSV locally:

```powershell
python backend/data/raw/convert_parquet.py
```

---

## Step 4: Verification

After conversion completes, verify that the CSV file exists:
`backend/data/raw/dpwh_transparency_data.csv` (~102 MB)

Both `.parquet` and `.csv` files are ignored by git and will not be pushed to GitHub.

---

## Dataset Quick Facts

| Property | Value |
| :--- | :--- |
| Total records | ~248,220 rows |
| Parquet file size | ~20.8 MB |
| CSV file size | ~102 MB |
| Target sample size | 1,600 projects (2021-2023) |
| Source | [BetterGov.ph Open Data Portal](https://data.bettergov.ph/) |
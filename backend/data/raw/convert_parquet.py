import csv
import os
import pyarrow.parquet as pq

# paths relative to this script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parquet_path = os.path.join(current_dir, "dpwh_transparency_data.parquet")
csv_path = os.path.join(current_dir, "dpwh_transparency_data.csv")

if not os.path.exists(parquet_path):
    print(f"Error: Could not find {parquet_path}")
    print("Please download the Parquet file first from BetterGov.ph and place it here.")
    exit(1)

print("Reading Parquet file...")
table = pq.read_table(parquet_path)

print(f"Converting to CSV: {csv_path}...")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(table.schema.names)
    for batch in table.to_batches():
        for row in zip(*batch.columns):
            writer.writerow([str(v) for v in row])

print("Conversion complete!")
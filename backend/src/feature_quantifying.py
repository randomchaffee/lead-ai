import numpy as np
from datetime import datetime
from load_and_sample import load_and_sample


# function to calculate project duration in calendar days
def compute_duration(start_str, end_str):
    # check if either date string is empty
    if len(start_str) == 0 or len(end_str) == 0:
        return 0

    try:
        start_date = datetime.strptime(start_str.strip(), "%Y-%m-%d")
        end_date = datetime.strptime(end_str.strip(), "%Y-%m-%d")
        day_diff = end_date - start_date # get differnce in days
        
        # only return positive duration
        if day_diff.days > 0:
            return day_diff.days
        else:
            return 0
    except ValueError:
        return 0 # fallback if date format is invalid


# build dictionary statistics per contractor
def build_contractor_profiles(rows):
    stats = {} # dictionary mapping contrator name to stats

    for row in rows:
        contractor = row.get("contractor", "").strip()
        raw_budget = row.get("budget", "0")
        
        # convert string budget to float
        try:
            budget_val = float(raw_budget)
        except ValueError:
            budget_val = 0.0 # fallback for bad budget valud

        # check if proejct missed deadline or was terminated
        status = row.get("status", "").strip()
        completion_date = row.get("completionDate", "").strip()
        
        is_delayed = 0 # flag if delayed or not
        if status == "Terminated" or completion_date == "":
            is_delayed = 1

        # update existing entry or create new entry
        if contractor in stats:
            stats[contractor]["total_projects"] = stats[contractor]["total_projects"] + 1
            stats[contractor]["total_budget"] = stats[contractor]["total_budget"] + budget_val
            stats[contractor]["delayed_projects"] = stats[contractor]["delayed_projects"] + is_delayed
        else:
            stats[contractor] = {
                "total_projects": 1,
                "total_budget": budget_val,
                "delayed_projects": is_delayed
            }

    return stats


# build 2d numpy array with all 5 features
def create_feature_matrix(rows, contractor_stats):
    matrix_rows = [] # list to hold all feature rows

    for row in rows:
        contractor = row.get("contractor", "").strip()

        # feature 1 project budget
        try:
            budget = float(row.get("budget", "0"))
        except ValueError:
            budget = 0.0

        # feature 2 duration in days
        start = row.get("startDate", "")
        end = row.get("completionDate", "")
        duration = compute_duration(start, end)

        # get contractor stats from precomputed dictionary
        c_stats = contractor_stats.get(contractor, {"total_projects": 1, "total_budget": 0.0, "delayed_projects": 0})
        
        # feature 3 contractor project volume
        count = c_stats["total_projects"]
        
        # feature 4 contractor average budget
        avg_budget = c_stats["total_budget"] / count
        
        # feature 5 contractor delay rate
        delay_rate = c_stats["delayed_projects"] / count

        # append row with 5 values
        matrix_rows.append([budget, duration, count, avg_budget, delay_rate])

    # convert list to numpy array with float data type
    feature_matrix = np.array(matrix_rows, dtype=float)
    return feature_matrix

# manual min max scaling using numpy operations
def scale_features(matrix):
    min_vals = np.min(matrix, axis=0) # array of column minimums
    max_vals = np.max(matrix, axis=0) # array of column maximums

    val_range = max_vals - min_vals # difference between max and min

    # prevent division by zero if all values in column are equal
    for i in range(len(val_range)):
        if val_range[i] == 0:
            val_range[i] = 1.0

    # apply min max formula: (x - min) / (max - min)
    scaled_matrix = (matrix - min_vals) / val_range
    return scaled_matrix

# master function to load and prepare the data for all algorithms
def prepare_dataset():
    # get sampled data from previous step
    sampled_rows = load_and_sample()

    print("\n\nstarting feature engineering process")
    
    # build contractor stats dictionary
    contractor_stats = build_contractor_profiles(sampled_rows)
    print("contractor profiles created:", len(contractor_stats))
    
    # create raw 5 feature matrix
    raw_matrix = create_feature_matrix(sampled_rows, contractor_stats)
    print("raw matrix shape:", raw_matrix.shape)
    
    # normalize matrix to 0 to 1 range
    normalized_matrix = scale_features(raw_matrix)
    print("normalized matrix shape:", normalized_matrix.shape)
    print("normalized value range:", np.min(normalized_matrix), "to", np.max(normalized_matrix))
    
    # display sample comparison for verification
    print("\nsample feature comparison for first project (raw vs normalized):")
    feature_labels = ["budget", "duration_days", "contractor_count", "contractor_avg_budget", "contractor_delay_rate"]
    for i in range(len(feature_labels)):
        name = feature_labels[i]
        raw_val = raw_matrix[0][i]
        norm_val = normalized_matrix[0][i]
        print(" ", name, "raw:", raw_val, "normalized:", round(norm_val, 4))

    return normalized_matrix, sampled_rows

if __name__ == "__main__":
    X, rows = prepare_dataset()
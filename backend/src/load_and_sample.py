import csv
import random

from config import RANDOM_SEED, SAMPLE_SIZE, NUM_OF_PROJS, VALID_YEARS

# seed for random sampling to keep things consistent
random.seed(RANDOM_SEED)

csv_path = "backend/data/raw/dpwh_transparency_data.csv" # relative path to raw datset


# function to read csv and filter years
def filter_by_year(file_path):
    year_filtered_rows = [] # list to hold valid rows
    valid_years = VALID_YEARS
    record_count = 0 # counter for number of records

    # open file with utf 8 encoding
    with open(file_path, mode="r", encoding="utf-8-sig", errors="replace") as file:
        reader = csv.DictReader(file)
        
        # loop through every row in the csv
        for row in reader:
            start_date = row.get("startDate", "")
            
            # check if date exists and is long enough for year
            if len(start_date) >= 4:
                year = start_date[0:4] # extract first 4 chars for year
                
                # check if year matches the target list
                if year in valid_years:
                    year_filtered_rows.append(row)
            
            record_count += 1 # for counting the total number of rows/record

    return record_count, year_filtered_rows


# function to count and filter by contractor frequency
def filter_by_contractor(rows, min_count):
    contractor_counts = {} # dictionary to count projects per contractor
    
    # count appearances of each contrator
    for row in rows:
        contractor = row.get("contractor", "").strip()
        
        # increment count if exists, else start at 1
        if contractor in contractor_counts:
            contractor_counts[contractor] = contractor_counts[contractor] + 1
        else:
            contractor_counts[contractor] = 1

    freq_filtered_rows = [] # list for filtered output
    
    # only keep row if contractor has enough projects
    for row in rows:
        contractor = row.get("contractor", "").strip()
        count = contractor_counts.get(contractor, 0)
        
        # check threshold
        if count >= min_count:
            freq_filtered_rows.append(row)

    return freq_filtered_rows


# function to get random sample
def get_sample(rows, sample_size):
    # check if rows count is smaller than sample target
    if len(rows) <= sample_size:
        return rows
    
    # use random sample from random library to get samples from the dataset (row)
    sampled_rows = random.sample(rows, sample_size)
    return sampled_rows


# main loader function
def load_and_sample():
    print("starting data loading and filtering")
    
    # filter by year
    record_count, year_rows = filter_by_year(csv_path)
    print("total number of rows/records:", record_count)
    print("rows after year filter:", len(year_rows))
    
    #filter by contractor
    contractor_rows = filter_by_contractor(year_rows, NUM_OF_PROJS)
    print("rows after contractor filter:", len(contractor_rows))
    
    #get random sample
    final_rows = get_sample(contractor_rows, SAMPLE_SIZE)
    print("final sample size:", len(final_rows))
    
    print("first sampled row preview:") # for checking only
    # print sample row for verifcation
    if len(final_rows) > 0:
        first = final_rows[0]
        print("contract id:", first.get("contractId"))
        print("contractor:", first.get("contractor"))
        print("budget:", first.get("budget"))
        print("start date:", first.get("startDate"))
        print("status:", first.get("status"))

    return final_rows


if __name__ == "__main__":
    data = load_and_sample()
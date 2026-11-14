# centralized configuration constants for the lead-ai backend pipeline
# modify these values to experiment with different parameters

RANDOM_SEED = 42 # seed for reproducible sampling and centroid initialization

SAMPLE_SIZE = 1600 # number of projects randomly sampled from the filtered dataset
NUM_OF_PROJS = 3 # minimum number of projects for a contractor to be included
VALID_YEARS = ["2021", "2022", "2023"] # allowed contract start years for filtering

K = 4 # number of clusters for k-means and k-medoids
MAX_ITER = 300 # maximum iterations allowed for k-means convergence

EPS = 0.15 # dbscan neighborhood radius in euclidean distance units
MIN_PTS = 5 # minimum points required to form a dense region in dbscan

FEATURE_NAMES = [
    "budget",
    "duration_days",
    "contractor_count",
    "contractor_avg_budget",
    "contractor_delay_rate"
]
import json
import os
import numpy as np
import pandas as pd

# Path to directory containing JSON files
data_dir = "data"

# Collect all timing values from all files
all_timings = []

for filename in os.listdir(data_dir):
    if filename.endswith(".json"):
        file_path = os.path.join(data_dir, filename)
        with open(file_path, "r") as f:
            data = json.load(f)

        ngram_times = data.get("ngram_times", {})
        for timings in ngram_times.values():
            all_timings.extend(timings)

# Skip if no data
if not all_timings:
    print("No timing data found.")
else:
    timings_array = np.array(all_timings)

    # Calculate overall stats
    std_dev = np.std(timings_array)
    stats = {
        "count": len(timings_array),
        "min": np.min(timings_array),
        "0.5% low": np.percentile(timings_array, 0.5),
        "1% low": np.percentile(timings_array, 1),
        "5% low": np.percentile(timings_array, 5),
        "median": np.median(timings_array),
        "mean": np.mean(timings_array),
        "95% high": np.percentile(timings_array, 95),
        "99% high": np.percentile(timings_array, 99),
        "max": np.max(timings_array),
        "std_dev": std_dev,
        "variance": std_dev ** 2
    }
    # Display the results without scientific notation
    pd.set_option("display.float_format", "{:.5f}".format)

    stats_df = pd.DataFrame(stats, index=["value"]).T
    print(stats_df)

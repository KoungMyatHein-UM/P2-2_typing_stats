import os
import json
from knn import score_live

# Assumes these are already defined in your environment or imported
# - classify_live
# - all_keys
# - timeout
# - knn_index

test_dir = "../test"
results = []

for fname in os.listdir(test_dir):
    if not fname.endswith(".json"):
        continue

    with open(os.path.join(test_dir, fname), "r") as f:
        data = json.load(f)
        ngram_times = data.get("ngram_times", {})

    result = score_live(ngram_times)
    result["file"] = fname
    results.append(result)

# Print results
for r in results:
    print(r)

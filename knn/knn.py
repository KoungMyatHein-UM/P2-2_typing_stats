import glob
import os
import json
from collections import defaultdict


def collect_ngram_times(data_dir):
    ngram_times = defaultdict(list)

    for fname in os.listdir(data_dir):
        if not fname.endswith(".json"):
            continue

        path = os.path.join(data_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "ngram_times" in data and isinstance(data["ngram_times"], dict):
            for pattern, timings in data["ngram_times"].items():
                ngram_times[pattern].extend(timings)

    return dict(ngram_times)


def normalise(max, min, value):
    return 0.5 if max == min else (value - min) / (max - min)


def get_score(model, pattern_data):
    mean_squared_error = 0
    total_patterns = 0

    for pattern, timings in pattern_data.items():
        if pattern not in model:
            continue  # Skip if the pattern doesn't exist in the model

        model_timings = model[pattern]
        if not model_timings or not timings:
            continue  # Skip if either list is empty

        combined_timings = model_timings + timings
        max_timing = max(combined_timings)
        min_timing = min(combined_timings)

        squared_errors = 0
        for timing in timings:
            timing = normalise(max_timing, min_timing, timing)
            for model_timing in model_timings:
                model_timing = normalise(max_timing, min_timing, model_timing)
                squared_errors += (timing - model_timing) ** 2

        mean_squared_error += squared_errors / (len(timings) * len(model_timings))
        total_patterns += 1

    if total_patterns == 0:
        return None  # Or float('inf'), depending on how you want to handle this

    return mean_squared_error / total_patterns


def get_weighted_score(model, pattern_data, fast_penalty_factor=5.0, soft_fast_penalty=2.0, hard_fast_threshold=25, soft_fast_threshold=40, max_timing_cap=600):
    mean_squared_error = 0
    total_patterns = 0

    for pattern, timings in pattern_data.items():
        if pattern not in model:
            continue

        model_timings = model[pattern]
        if not model_timings or not timings:
            continue

        # Cap timings to reduce the influence of long outliers
        capped_model = [min(t, max_timing_cap) for t in model_timings]
        capped_test = [min(t, max_timing_cap) for t in timings]
        combined = capped_model + capped_test

        max_timing = max(combined)
        min_timing = min(combined)

        squared_errors = 0
        for t in capped_test:
            norm_t = normalise(max_timing, min_timing, t)

            for mt in capped_model:
                norm_mt = normalise(max_timing, min_timing, mt)
                error = norm_t - norm_mt

                # Penalise faster timings more
                if t < hard_fast_threshold:
                    squared_errors += (error ** 2) * fast_penalty_factor
                elif t < soft_fast_threshold:
                    squared_errors += (error ** 2) * soft_fast_penalty
                else:
                    squared_errors += error ** 2

        mean_squared_error += squared_errors / (len(capped_test) * len(capped_model))
        total_patterns += 1

    if total_patterns == 0:
        return None

    return mean_squared_error / total_patterns


def get_relative_score(unweighted_score, weighted_score, max_ratio=5.0):
    """
    Normalise weighted score relative to unweighted baseline.
    Clamps extreme values to keep scale between 0 and 1.

    Args:
        unweighted_score (float): baseline MSE from get_score()
        weighted_score (float): penalised MSE from get_weighted_score()
        max_ratio (float): upper bound to clip extreme penalty inflation

    Returns:
        float: relative score between 0 (perfectly human) and 1 (very bot-like)
    """
    if unweighted_score is None or weighted_score is None:
        return None

    ratio = weighted_score / (unweighted_score + 1e-6)  # avoid zero div
    ratio = min(ratio, max_ratio)  # clamp to cap crazy outliers
    return (ratio - 1) / (max_ratio - 1)


if __name__ == "__main__":
    data_dir = "../data"
    test_dir = "../test"
    
    all_patterns = collect_ngram_times(data_dir)
    
    test_files = glob.glob(os.path.join(test_dir, "*.json"))
    test_files.reverse()
    
    for test_file in test_files:
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                test_data = json.load(f)

            # Extract timing data
            ngram_times = test_data.get('ngram_times', {})

            score = get_score(all_patterns, ngram_times)
            print(f"MSE Score for {os.path.basename(test_file)}: {score}")

            weighted_score = get_weighted_score(all_patterns, ngram_times)
            print(f"Weighted score for {os.path.basename(test_file)}: {weighted_score}")

            combined_score = get_relative_score(score, weighted_score)
            print(f"Combined score for {os.path.basename(test_file)}: {combined_score}")


        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading test file {test_file}: {str(e)}")
            continue

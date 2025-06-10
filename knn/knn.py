import glob
import os
import json
from collections import defaultdict
import numpy as np


def trim_ngram_times(raw_ngram_times, lower_percentile=15, upper_percentile=100):
    if lower_percentile == 0:
        return raw_ngram_times

    trimmed = {}

    total_before = sum(len(v) for v in raw_ngram_times.values())
    print(f"Before: {total_before}")

    for pattern, timings in raw_ngram_times.items():
        if not timings:
            continue

        arr = np.array(timings)
        low = np.percentile(arr, lower_percentile)
        high = np.percentile(arr, upper_percentile)
        trimmed_arr = arr[(arr >= low) & (arr <= high)]
        trimmed[pattern] = trimmed_arr.tolist()

    total_after  = sum(len(v) for v in trimmed.values())
    print(f"After: {total_after}")

    return trimmed


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

    trimmed_ngram_times = trim_ngram_times(ngram_times)

    return dict(trimmed_ngram_times)


def normalise(max, min, value):
    return 0.5 if max == min else (value - min) / (max - min)


def get_score(model, pattern_data, n_neighbors=3):
    """
    Compute an average MSE over patterns by comparing each timing
    only to its n_neighbors closest model timings (after normalisation).
    """
    total_mse = 0.0
    total_patterns = 0

    for pattern, timings in pattern_data.items():
        if pattern not in model:
            continue  # Skip if the pattern doesn't exist in the model

        model_timings = model[pattern]
        if not model_timings or not timings:
            continue  # Skip if either list is empty

        # Combine to find min/max for normalisation
        combined = np.array(model_timings + timings, dtype=float)
        min_t, max_t = combined.min(), combined.max()

        # Normalise both lists into [0, 1]
        mt_arr = (np.array(model_timings, dtype=float) - min_t) / (max_t - min_t)
        t_arr  = (np.array(timings, dtype=float)      - min_t) / (max_t - min_t)

        pattern_mse = 0.0
        for t_norm in t_arr:
            # Compute distances to all model points in normalised space
            dists = np.abs(mt_arr - t_norm)
            # Select up to n_neighbors nearest neighbours
            k = min(n_neighbors, len(mt_arr))
            nearest_idxs = np.argpartition(dists, k - 1)[:k]
            nearest_vals = mt_arr[nearest_idxs]

            # Squared errors against those k neighbours
            sq_errs = (t_norm - nearest_vals) ** 2
            pattern_mse += sq_errs.mean()  # average of those k squared‐errors

        # Now average over all timings in this pattern
        pattern_mse /= len(t_arr)
        total_mse += pattern_mse
        total_patterns += 1

    if total_patterns == 0:
        return None  # or float('inf') if you prefer

    return total_mse / total_patterns

def get_weighted_score(
        model,
        pattern_data,
        n_neighbors=3,
        fast_penalty_factor=5.0,
        soft_fast_penalty=2.0,
        hard_fast_threshold=25,
        soft_fast_threshold=40,
        max_timing_cap=600
):
    """
    Compute an average weighted MSE over patterns by comparing each test timing
    only to its n_neighbors closest model timings (after normalisation and capping).
    """
    total_weighted_mse = 0.0
    total_patterns = 0

    for pattern, timings in pattern_data.items():
        if pattern not in model:
            continue

        model_timings = model[pattern]
        if not model_timings or not timings:
            continue

        # 1) Cap both model and test timings to reduce outlier influence
        capped_model = [min(t, max_timing_cap) for t in model_timings]
        capped_test  = [min(t, max_timing_cap) for t in timings]
        combined     = np.array(capped_model + capped_test, dtype=float)

        # 2) Find overall min/max for normalisation
        min_t, max_t = combined.min(), combined.max()

        # 3) Normalise model and test lists into [0, 1]
        mt_arr = (np.array(capped_model, dtype=float) - min_t) / (max_t - min_t)
        t_arr  = (np.array(capped_test,  dtype=float) - min_t) / (max_t - min_t)

        pattern_mse = 0.0
        for raw_t, t_norm in zip(capped_test, t_arr):
            # Compute absolute distances to all normalised model points
            dists = np.abs(mt_arr - t_norm)

            # Pick up to n_neighbors nearest neighbours
            k = min(n_neighbors, len(mt_arr))
            nearest_idxs = np.argpartition(dists, k - 1)[:k]
            nearest_mt_norms = mt_arr[nearest_idxs]

            # For each neighbour, calculate weighted squared error
            sq_errs = []
            for mt_norm in nearest_mt_norms:
                err = t_norm - mt_norm

                # Decide penalty factor based on raw (capped) timing
                if raw_t < hard_fast_threshold:
                    weight = fast_penalty_factor
                elif raw_t < soft_fast_threshold:
                    weight = soft_fast_penalty
                else:
                    weight = 1.0

                sq_errs.append((err ** 2) * weight)

            # Average the k weighted squared-errors for this t
            pattern_mse += np.mean(sq_errs)

        # Average over all test timings in this pattern
        pattern_mse /= len(t_arr)
        total_weighted_mse += pattern_mse
        total_patterns += 1

    if total_patterns == 0:
        return None  # or float('inf'), if you’d rather

    return total_weighted_mse / total_patterns


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


def get_flatness_score(test_data, model_data=None, expected_variability=80.0, penalty_weight=0.3, structure_weight=0.3):
    """
    Penalises:
    - Low inter-pattern variance
    - Low intra-pattern variance
    - Optional: flattened per-pattern structure vs model
    Returns a value between 0 and penalty_weight + structure_weight
    """
    # Inter-pattern: mean timings across patterns
    pattern_means = [np.mean(t) for t in test_data.values() if t]
    inter_std = np.std(pattern_means) if pattern_means else 0

    # Intra-pattern: jitter within each pattern
    intra_jitters = [np.std(t) for t in test_data.values() if len(t) > 1]
    intra_std = np.mean(intra_jitters) if intra_jitters else 0

    flatness_component = 1.0 - min((inter_std + intra_std) / expected_variability, 1.0)
    flatness_score = flatness_component * penalty_weight

    # Optional: structure comparison (only if model_data provided)
    structure_score = 0
    if model_data:
        shared_patterns = set(test_data.keys()) & set(model_data.keys())
        if len(shared_patterns) >= 3:  # need at least 3 to make shape meaningful
            test_means = []
            model_means = []
            for pattern in shared_patterns:
                if test_data[pattern] and model_data[pattern]:
                    test_means.append(np.mean(test_data[pattern]))
                    model_means.append(np.mean(model_data[pattern]))

            test_norm = (np.array(test_means) - np.mean(test_means)) / (np.std(test_means) + 1e-6)
            model_norm = (np.array(model_means) - np.mean(model_means)) / (np.std(model_means) + 1e-6)

            structure_score = np.mean((test_norm - model_norm) ** 2) * structure_weight

    return min(flatness_score + structure_score, 1.0)


def get_final_score(model_patterns, test_patterns, alpha=0.35):
    flatness_score = get_flatness_score(test_patterns, model_data=model_patterns, expected_variability=30.0, penalty_weight=1.0)
    weighted_score = get_weighted_score(model_patterns, test_patterns)
    score = get_score(model_patterns, test_patterns)
    hybrid_score = get_relative_score(score, weighted_score)
    final_score = (hybrid_score * (1 - alpha)) + (flatness_score * alpha)

    print(f"\n=====\nhybrid score: {hybrid_score:.6f}, flatness score: {flatness_score:.6f}, final score: {final_score:.6f}\n=====\n")


    return final_score


if __name__ == "__main__":
    # Colour constants (ANSI escape codes)
    RESET   = "\033[0m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    BOLD    = "\033[1m"

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
            score = get_final_score(all_patterns, ngram_times)
            print(
                f"{CYAN}Final score for {os.path.basename(test_file)}:{RESET} "
                f"{GREEN if score < 0.3 else (YELLOW if score < 0.7 else RED)}"
                f"{score:.4f}{RESET}"
            )


        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading test file {test_file}: {str(e)}")
            continue

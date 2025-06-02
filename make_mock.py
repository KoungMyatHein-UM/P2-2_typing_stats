import json
import random

def generate_mock_ngram_data(pattern_count=50, timings_per_pattern=5,
                             timing_min=30, timing_max=300, seed=None):
    if seed is not None:
        random.seed(seed)

    mock_data = {
        "ngram_times": {}
    }

    for i in range(pattern_count):
        # Generate fake pattern like "[<key1><key2>]->[<key3>]"
        left = "".join(random.choices(["<a>", "<b>", "<c>", "<d>", "<e>", "<space>", "<shift>"], k=random.randint(1, 3)))
        right = random.choice(["<f>", "<g>", "<h>", "<i>", "<j>"])
        pattern = f"[{left}]->[{right}]"

        timings = [random.randint(timing_min, timing_max) for _ in range(timings_per_pattern)]
        mock_data["ngram_times"][pattern] = timings

    return mock_data

if __name__ == "__main__":
    mock_json = generate_mock_ngram_data(
        pattern_count=100,
        timings_per_pattern=5,
        timing_min=75,
        timing_max=85,
        seed=42  # For reproducibility
    )

    with open("test/generated_mock_test_data.json", "w", encoding="utf-8") as f:
        json.dump(mock_json, f, indent=2)

    print("Mock ngram data saved to generated_mock_test_data.json")

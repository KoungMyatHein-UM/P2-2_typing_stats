import json
import random
import string

def generate_mock_ngram_data(pattern_count=50, timings_per_pattern=5,
                             timing_min=30, timing_max=300, seed=None):
    if seed is not None:
        random.seed(seed)

    mock_data = {
        "ngram_times": {}
    }

    # Build full printable character set plus special keys
    base_chars = list(string.ascii_letters + string.digits + string.punctuation)
    specials = ["<space>", "<enter>", "<tab>", "<shift>", "<ctrl>", "<alt>", "<capslock>", "<esc>"]

    # Wrap base characters in angle brackets for consistency
    wrapped_chars = [f"<{c}>" for c in base_chars if c != ">"]  # Avoid nested >
    all_keys = wrapped_chars + specials

    for _ in range(pattern_count):
        left = "".join(random.choices(all_keys, k=random.randint(1, 3)))
        right = random.choice(all_keys)
        pattern = f"[{left}]->[{right}]"

        timings = [random.randint(timing_min, timing_max) for _ in range(timings_per_pattern)]
        mock_data["ngram_times"][pattern] = timings

    return mock_data

if __name__ == "__main__":
    mock_json = generate_mock_ngram_data(
        pattern_count=50000,
        timings_per_pattern=50,
        timing_min=100,
        timing_max=200,
        seed=42  # For reproducibility
    )

    with open("test/generated_mock_test_data.json", "w", encoding="utf-8") as f:
        json.dump(mock_json, f, indent=2)

    print("Mock ngram data saved to generated_mock_test_data.json")

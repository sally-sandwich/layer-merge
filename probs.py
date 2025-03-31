import json
import random

def generate_distribution(n):
    # Generate even distribution where all values sum to 1
    interval = 1.0 / n
    return [interval] * n

data = {
    "1": 85,
    "2": 45,
    "3": 31,
    "4": 75,
    "5": 130,
    "6": 40,
    "7": 38,
    "8": 34,
    "9": 24,
    "10": 22,
    "11": 14,
}

result = {key: generate_distribution(count) for key, count in data.items()}

# Save results to file
with open("probs.json", "w") as f:
    json.dump(result, f, indent=2)

print("JSON file 'probs.json' has been created.")

# Validate that each set of probabilities sums to 1
for key, probs in result.items():
    total = sum(probs)
    print(f"{key}: sum = {total:.10f}, count = {len(probs)}")
    if abs(total - 1) > 1e-10:  # Check if the sum is very close to 1
        print(f"Warning: {key} probabilities do not sum exactly to 1")

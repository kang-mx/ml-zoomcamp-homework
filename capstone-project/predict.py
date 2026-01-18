import requests

url = "http://127.0.0.1:5000/predict"

raw_row = "1 1 0.0023 0.0003 100.0 518.67 643.02 1585.29 1398.21 14.62 21.61 553.90 2388.04 9050.17 1.30 47.20 521.72 2388.03 8125.55 8.4052 0.03 392 2388 100.00 38.86 23.3735"

row = list(map(float, raw_row.split()))

FEATURE_IDX = [
    1, 6, 7, 8, 10, 11, 12, 13,
    15, 16, 17, 18, 19, 21, 24, 25
]

# Build one timestep
timestep = [row[i] for i in FEATURE_IDX]

# Repeat to form sequence
sequence = [timestep] * 30

response = requests.post(url, json={"sequence": sequence})

print("Status:", response.status_code)
print("Raw response:", response.text)

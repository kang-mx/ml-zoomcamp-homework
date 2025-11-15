import requests

url = "https://repair-cost-api.onrender.com/predict"

data = {
    "defect_type": "Structural",
    "defect_location": "Component",
    "severity": "Minor",
    "inspection_method": "Visual Inspection",
    "product_id": "15",
    "month": 6,
    "day_of_week": 3
}

response = requests.post(url, json=data)
print(response.json())
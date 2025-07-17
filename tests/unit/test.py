import requests

data = {
    "cityname": "Raleigh",
    "state": "NC",
    "bedrooms": 3,
    "bathrooms":2,
    "square_feet": 1300
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=data)
print(response.json())
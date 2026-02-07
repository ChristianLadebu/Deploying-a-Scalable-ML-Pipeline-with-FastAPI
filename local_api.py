import requests

# TODO: send a GET using the URL http://127.0.0.1:8000
r = requests.get("http://127.0.0.1:8000")

# TODO: print the status code
print("GET status code:", r.status_code)
# TODO: print the welcome message
print("GET response:", r.json())



data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# TODO: send a POST using the data above
r = requests.post("http://127.0.0.1:8000/data/", json=data)

print("POST status code:", r.status_code)
print("POST response headers:", r.headers.get("content-type"))
print("POST response body (first 1000 chars):", r.text[:1000])

# Only try JSON if it looks like JSON
if "application/json" in (r.headers.get("content-type") or ""):
    print("POST response JSON:", r.json())




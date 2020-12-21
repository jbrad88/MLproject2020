import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'windspeed':2})

print(r.json())
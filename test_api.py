import requests
import json

def send_chat_message(message, lat, lng, model_name):
    url = 'http://127.0.0.1:5000/chat'
    headers = {
        'Content-Type': 'application/json; charset=UTF-8'
    }
    data = {
        'mensaje': message,
        'lat': lat,
        'lng': lng,
        'model_name': model_name
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
        return response.json()  # Assumes the response is in JSON format
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

# Example usage:
text = "Me duele el estómago, que podría tomar? Además, que farmacias tengo cerca?"
latitude = -37.444513
longitude = -72.336370

opcion_modelo = {
    1: "gpt-4o",
    2: "gpt-4o-mini",
    3: "llama-3.1-70b-versatile",
    4: "llama3-8b-8192",
}
model_name = opcion_modelo[4]

result = send_chat_message(text, latitude, longitude, model_name)

if result:
    print("Response received:")
    print(json.dumps(result, indent=2))  # Pretty print the JSON response
else:
    print("Failed to get a response")
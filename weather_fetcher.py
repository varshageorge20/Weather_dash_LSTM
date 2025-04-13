import requests
import pandas as pd

def get_weather_data(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()

    forecasts = []
    for entry in data['list']:
        forecasts.append({
            "timestamp": entry["dt_txt"],
            "temp": entry["main"]["temp"],
            "humidity": entry["main"]["humidity"],
            # "description": entry["weather"][0]["description"]
        })

    df = pd.DataFrame(forecasts)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

"""
Build a weekly weather+leakage CSV for leakage forecasting.

Steps:
1. Pull recent historical hourly weather (last LOOKBACK_DAYS) from archive API.
2. Pull hourly weather forecast (next FORECAST_DAYS) from forecast API.
3. Combine hourly streams, aggregate to weekly max/min, and align feature names.
4. Merge historical weekly leakage by week-ending date.

Future weeks keep missing water_leakage values by design.
"""

import os

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry


LOOKBACK_DAYS = 365
FORECAST_DAYS = 16
LATITUDE = 51.579881
LONGITUDE = 0.77565183

HOURLY_VARS = [
	"temperature_2m",
	"soil_temperature_28_to_100cm",
	"soil_moisture_28_to_100cm",
	"precipitation",
]


def response_to_hourly_dataframe(response, variable_names):
	"""Convert Open-Meteo response to a standard hourly dataframe."""
	hourly = response.Hourly()
	data = {
		"date": pd.date_range(
			start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
			end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
			freq=pd.Timedelta(seconds=hourly.Interval()),
			inclusive="left",
		)
	}
	for i, name in enumerate(variable_names):
		data[name] = hourly.Variables(i).ValuesAsNumpy()

	return pd.DataFrame(data=data)


def get_openmeteo_client():
	cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
	retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
	return openmeteo_requests.Client(session=retry_session)


def get_hourly_history(client):
	now_utc = pd.Timestamp.now(tz="UTC")
	start_date = (now_utc - pd.Timedelta(days=LOOKBACK_DAYS)).date().isoformat()
	end_date = now_utc.date().isoformat()

	url = "https://archive-api.open-meteo.com/v1/archive"
	params = {
		"latitude": LATITUDE,
		"longitude": LONGITUDE,
		"start_date": start_date,
		"end_date": end_date,
		"hourly": HOURLY_VARS,
	}
	response = client.weather_api(url, params=params)[0]
	print(f"Archive coordinates: {response.Latitude()}°N {response.Longitude()}°E")
	print(f"Archive elevation: {response.Elevation()} m asl")
	return response_to_hourly_dataframe(response, HOURLY_VARS)


def get_hourly_forecast(client):
	url = "https://api.open-meteo.com/v1/forecast"
	params = {
		"latitude": LATITUDE,
		"longitude": LONGITUDE,
		"hourly": HOURLY_VARS,
		"forecast_days": FORECAST_DAYS,
	}
	response = client.weather_api(url, params=params)[0]
	print(f"Forecast coordinates: {response.Latitude()}°N {response.Longitude()}°E")
	print(f"Forecast elevation: {response.Elevation()} m asl")
	return response_to_hourly_dataframe(response, HOURLY_VARS)


def build_weekly_weather(hourly_history, hourly_forecast):
	hourly_dataframe = (
		pd.concat([hourly_history, hourly_forecast], ignore_index=True)
		.drop_duplicates(subset=["date"], keep="last")
		.sort_values("date")
	)

	non_zero_precip = (hourly_dataframe["precipitation"].fillna(0) > 0).sum()
	print(f"Combined hourly rows: {len(hourly_dataframe)}")
	print(f"Rows with precipitation > 0: {non_zero_precip}")

	weekly_dataframe = (
		hourly_dataframe.set_index("date").resample("W").agg(["max", "min"])
	)
	weekly_dataframe.columns = [f"{col}_{stat}" for col, stat in weekly_dataframe.columns]

	# Keep the exact training-feature column order.
	weekly_dataframe = weekly_dataframe[
		[
			"temperature_2m_min",
			"soil_temperature_28_to_100cm_min",
			"soil_moisture_28_to_100cm_min",
			"precipitation_min",
			"temperature_2m_max",
			"soil_temperature_28_to_100cm_max",
			"soil_moisture_28_to_100cm_max",
			"precipitation_max",
		]
	]

	return weekly_dataframe


def load_historical_leakage():
	"""Read known historical leakage values from the training-ready weekly file."""
	leakage_path = os.path.join(os.getcwd(), "data", "weekly_min_max.csv")
	leakage_df = pd.read_csv(leakage_path, parse_dates=["date"])
	leakage_df["date"] = pd.to_datetime(leakage_df["date"], utc=True)
	leakage_df = leakage_df.set_index("date")
	return leakage_df[["water_leakage"]]


def main():
	client = get_openmeteo_client()

	hourly_history = get_hourly_history(client)
	hourly_forecast = get_hourly_forecast(client)
	weekly_weather = build_weekly_weather(hourly_history, hourly_forecast)

	leakage_history = load_historical_leakage()
	output_df = weekly_weather.join(leakage_history, how="left")

	output_path = os.path.join(os.getcwd(), "data", "forecast_min_max.csv")
	output_df.to_csv(output_path, index=True)

	print("\nWeekly weather + leakage window")
	print(output_df)
	print(f"\nSaved: {output_path}")


if __name__ == "__main__":
	main()
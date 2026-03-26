import os

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry


def weekly_weather_past(lat, long, location_name="temporary"):
    """Return weekly min,mean,max data of parameters of interest, if no file with its name already exists."""

    # File to save/load data
    weekly_csv = f"weather_data_weekly_{location_name}.csv"
    if os.path.exists(weekly_csv):
        print("File already exists.")
        return (None, None)
    else:
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        # Make sure all required weather variables are listed here
        # The order of variables in hourly or daily is important to assign them correctly below
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": long,
            "start_date": "2001-12-24",
            "end_date": "2026-02-22",
            "hourly": [
                "temperature_2m",
                "soil_temperature_100_to_255cm",
                "soil_moisture_100_to_255cm",
                "precipitation",
            ],
        }
        responses = openmeteo.weather_api(url, params=params)

        # Process first location. Add a for-loop for multiple locations or weather models
        response = responses[0]

        # Process hourly data. The order of variables needs to be the same as requested.
        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        hourly_soil_temperature_100_to_255cm = hourly.Variables(
            1
        ).ValuesAsNumpy()
        hourly_soil_moisture_100_to_255cm = hourly.Variables(2).ValuesAsNumpy()
        hourly_precipitation = hourly.Variables(3).ValuesAsNumpy()

        hourly_data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            )
        }

        hourly_data["temperature_2m"] = hourly_temperature_2m
        hourly_data["soil_temperature_100_to_255cm"] = (
            hourly_soil_temperature_100_to_255cm
        )
        hourly_data["soil_moisture_100_to_255cm"] = (
            hourly_soil_moisture_100_to_255cm
        )
        hourly_data["precipitation"] = hourly_precipitation

        # Resample to weekly maximums and minimums

        hourly_dataframe = pd.DataFrame(data=hourly_data)
        hourly_dataframe.set_index("date", inplace=True)
        weekly_dataframe = hourly_dataframe.resample("W").agg(
            ["min", "mean", "max"]
        )
        weekly_dataframe.columns = [
            f"{col}_{stat}" for col, stat in weekly_dataframe.columns
        ]

        print("Return suggested filename and weekly dataframe.")
        return (weekly_csv, weekly_dataframe)

    ###################
    ## Now merge the leakage data in
    ###################
    ## Open the Excel file and read the second sheet
    # file_path = "data/The Water Sector Meets Mathematics_ V-KEMS Virtual Study Group-20260323T171309Z-3-001/Challenge 1/VKEMS challenge 1 data.xlsx"
    # df = pd.read_excel(file_path, sheet_name=1)
    ## transpose data because the dates are along the top rather that down the side
    # df_transposed = df.T
    # df_transposed.reset_index(inplace=True)
    ## first row is the zone AA, Zone AB, etc so remove that
    # df_transposed = df_transposed.iloc[1:].reset_index(drop=True)
    ## column names have gone funny so data for Zone AA is now in column named 1
    ## eventually this should be changed to have the zone names be the headers
    ## use different number for different zone
    # weekly_dataframe["water_leakage"] = df_transposed[1].values
    # weekly_dataframe.to_csv("full_weekly_data.csv", index=True)

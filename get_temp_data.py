from darksky import forecast
import datetime
from datetime import date, timedelta
import numpy as np
import pandas as pd

# Input params
API = ''
boston = 42.3601, -71.0589
providence = 41.8240, -71.4128
portland = 45.4231, -122.6765
springfield = 42.1015, -72.5898


def get_weather_data(api_key, filename, start_date=date(2011, 4, 1), end_date=date(2018, 4, 1),
                     coordinates=boston):
    """
    Parameters
    ----------
    api_key: string
        API Key for Dark Sky

    filename: string
        Name for file of weather data.
        Ex: filename = 'weather_data.csv'

    start_date: date object
    end_date: date object

    coordinates: (x, y) (default is boston coordinates)
        coordinates of location for weather data.

    Returns
    -------
    df: pandas Dataframe
        Dataframe containing hourly weather data
    """

    # Generate list of dates to pull
    range_len = (end_date - start_date).days + 1  # Add one to include the end_date
    dates = start_date + np.arange(range_len) * datetime.timedelta(days=1)
    dates_iso = [date.isoformat() + 'T00:00:00' for date in dates]  # List of dates in ISO format

    # Pull hourly data for each day
    data = np.array([])  # Empty array to store data

    for day in dates_iso:
        boston = forecast(api_key, *coordinates, time=day)
        hourly_data = boston['hourly']['data']

        # Get additional info and merge
        additional_data = dict(zip(['offset', 'timezone', 'latitude', 'longitude'],
                                   [boston.offset, boston.timezone, boston.latitude, boston.longitude]))
        add_additional_data = lambda hourly_data_dictionary: {**hourly_data_dictionary, **additional_data}
        add_info_func = np.vectorize(add_additional_data)
        hourly_data = add_info_func(hourly_data)

        data = np.append(data, hourly_data)  # Append day data

    df = pd.DataFrame.from_records(data)
    df.to_csv(filename)

    return df


get_weather_data(API, filename='test.csv', start_date=date(2018, 3, 20))

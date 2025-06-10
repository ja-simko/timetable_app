import pandas as pd
import datetime as dt
from datetime import datetime, timedelta
import os
import time
import joblib

FILE_PATH = r"C:\Users\Jachym\OneDrive - České vysoké učení technické v Praze\Bakalářská_práce\02_CODE\PID_GTFS"

def read_stops_file():
    return pd.read_csv(os.path.join(FILE_PATH, "stops.txt"))

def get_stops_df():
    stops = read_stops_file()
    
    stops = stops[stops['asw_node_id'].notna()]
    stops['asw_node_id'] = stops['asw_node_id'].astype(int)
    filtered_stops = stops[(stops['location_type'] == 0)]
    return filtered_stops[['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'asw_node_id', 'platform_code']]

def read_textfiles_for_timetable():
    files = {
        "stop_times": "stop_times.txt",
        "trips": "trips.txt",
        "routes": "routes.txt",
        "calendar": "calendar.txt"
    }
    stop_times = pd.read_csv(os.path.join(FILE_PATH, files["stop_times"]), low_memory=False)
    trips = pd.read_csv(os.path.join(FILE_PATH, files["trips"]), low_memory=False)
    routes = pd.read_csv(os.path.join(FILE_PATH, files["routes"]))
    calendar = pd.read_csv(os.path.join(FILE_PATH, files["calendar"]), dtype={'start_date': int, 'end_date': int})

    return stop_times, trips, routes, calendar

def get_timetable_df():
    stop_times, trips, routes, calendar = read_textfiles_for_timetable()

    # Convert times like "25:30:30" to proper datetime representation
    def convert_time(time_str):
        hours, minutes, seconds = map(int, time_str.split(':'))
        time = dt.timedelta(hours=hours, minutes=minutes, seconds=seconds)
        return time

    #stop_times['departure_time'] = stop_times['departure_time'].apply(convert_time)
    #stop_times['arrival_time'] = stop_times['arrival_time'].apply(convert_time)

    trips_calendar = trips.merge(calendar, on='service_id', how='inner')
    stop_times_trips = stop_times.merge(trips_calendar, on='trip_id', how='inner')
    stop_times_routes = stop_times_trips.merge(routes, on='route_id', how='inner')

    timetable = stop_times_routes[
        ['trip_id', 'stop_id', 'departure_time', 'arrival_time',
        'stop_sequence', 'route_short_name', 'start_date', 'end_date', 'service_id'
        ]
    ].sort_values(by=['trip_id', 'stop_sequence'])

    #joblib.dump(timetable,r"C:\Users\Jachym\OneDrive - České vysoké učení technické v Praze\Bakalářská_práce\02_CODE\cache\timetable_cache_str.pkl")

    print(timetable[timetable['route_short_name'] == '98'])
    return timetable

def get_trip_service_dict():

    stop_times, trips, routes, calendar = read_textfiles_for_timetable()
    
    # Merge trips with calendar to get service days
    trips_with_service = trips.merge(calendar, on='service_id', how='inner')
    
    # Create dictionary to store trip service information
    trip_service_dict = {}

    # Iterate through each trip
    for _, row in trips_with_service.iterrows():
        service_days = []
        if row['monday']: service_days.append(0)
        if row['tuesday']: service_days.append(1)
        if row['wednesday']: service_days.append(2)
        if row['thursday']: service_days.append(3)
        if row['friday']: service_days.append(4)
        if row['saturday']: service_days.append(5)
        if row['sunday']: service_days.append(6)
        
        trip_service_dict[row['trip_id']] = {
            'service_days': service_days,
            'start_date': int(row['start_date']),
            'end_date': int(row['end_date'])
        }

    return trip_service_dict

if __name__ == "__main__":
    get_timetable_df()
    get_trip_service_dict()
    

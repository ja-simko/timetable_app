import pandas as pd
from datetime import datetime
import os

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
    calendar = pd.read_csv(os.path.join(FILE_PATH, files["calendar"]))

    return stop_times, trips, routes, calendar

def get_timetable_df():
    stop_times, trips, routes, calendar = read_textfiles_for_timetable()

    trips_calendar = trips.merge(calendar, on='service_id', how='inner')
    stop_times_trips = stop_times.merge(trips_calendar, on='trip_id', how='inner')
    stop_times_routes = stop_times_trips.merge(routes, on='route_id', how='inner')

    result = stop_times_routes[
        ['trip_id', 'stop_id', 'departure_time', 'arrival_time',
        'stop_sequence', 'route_short_name', 'start_date', 'end_date', 'service_id'
        ]
    ].sort_values(by=['trip_id', 'stop_sequence'])

    return result

if __name__ == "__main__":
    get_timetable_df()

    

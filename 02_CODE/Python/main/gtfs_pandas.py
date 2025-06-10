import pandas as pd
import datetime as dt
from datetime import datetime, timedelta
import os
import time
import numpy as np
from collections import defaultdict
import re
import timeit
import joblib

FILE_PATH = r"C:\Users\Jachym\OneDrive - České vysoké učení technické v Praze\Bakalářská_práce\02_CODE\PID_GTFS"
def convert_str_to_datetime(day):
    origin = datetime(2000, 1, 1)
    day = datetime.strptime(str(day), '%Y%m%d')
    return (day - origin).total_seconds() / (24 * 3600)


def read_stops_file():
    return pd.read_csv(os.path.join(FILE_PATH, "stops.txt"))

def read_stop_times_file():
    return pd.read_csv(os.path.join(FILE_PATH, "stop_times.txt"), low_memory=False)

def read_trips_file():
    return pd.read_csv(os.path.join(FILE_PATH, "trips.txt"), low_memory=False)

def read_routes_file():
    return pd.read_csv(os.path.join(FILE_PATH, "routes.txt"))

def read_calendar_file():
    return pd.read_csv(os.path.join(FILE_PATH, "calendar.txt"), dtype={'start_date': int, 'end_date': int})

def get_stops_df():
    stops = read_stops_file()
    stops = stops[stops['asw_node_id'].notna()]
    stops['asw_node_id'] = stops['asw_node_id'].astype(int)
    filtered_stops = stops[(stops['location_type'] == 0)]
    return filtered_stops[['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'asw_node_id', 'platform_code']]

def convert_str_to_sec(timestamp: str) -> int:
    return sum(int(t) * 60 ** i for i, t in enumerate(reversed(timestamp.split(":"))))

def build_timetable_df():
    stop_times, trips = read_stop_times_file(), read_trips_file()
    routes = read_routes_file()

    stop_times['departure_time'] = stop_times['departure_time'].apply(convert_str_to_sec).astype(int)
    stop_times['arrival_time'] = stop_times['arrival_time'].apply(convert_str_to_sec).astype(int)

    stop_times_trips = trips.merge(stop_times, on='trip_id', how='inner')
    stop_times_routes = stop_times_trips.merge(routes, on='route_id', how='inner')
    
    timetable = stop_times_routes.copy()

    timetable['stop_id'] = timetable['stop_id'].astype('string')
    timetable['trip_id'] = timetable['trip_id'].astype('string')
    timetable['route_short_name'] = timetable['route_short_name'].astype('string')
    timetable['main_station'] = timetable['stop_id'].str.extract(r"^(U\d+)[ZS]", expand=False).astype('string')
    timetable = timetable.dropna(subset=['main_station'])

    trip_id_to_route = generate_route_ids(timetable)

    timetable['route_id'] = timetable['trip_id'].map(trip_id_to_route)
    timetable['node_id'] = timetable['stop_id'] + '_' + timetable['route_id'] + '_' + timetable['route_short_name']

    timetable = timetable[
        ['trip_id', 'stop_id', 'node_id', 'main_station', 'departure_time', 'arrival_time', 'stop_sequence', 'route_short_name', 
        ]
    ].sort_values(by=['trip_id', 'stop_sequence'])
    return timetable

def build_edges():
    """ Builds the graph edges based on computed route IDs. """

    print('start building edges')
    timetable = get_timetable()
    print('Done Timetable')
    
    # Generate route IDs based on stop sequences
    edges = defaultdict(lambda: defaultdict(list))

    prev_row = None
    last_trip_id = None

    transfer_edges_added = set()


    for row in timetable.itertuples(index=False):
        if row.trip_id != last_trip_id:
            last_trip_id = row.trip_id
            prev_row = row
            continue
        
        out_main_station = prev_row.main_station
        in_main_station = row.main_station
        
        out_node = prev_row.node_id
        in_node = row.node_id

        dep_time = prev_row.departure_time
        arr_time = row.arrival_time

        shift = 0
    
        edges[out_node][in_node].append((dep_time, arr_time, row.trip_id, shift))

        if dep_time >= 24 * 3600:
            new_dep_time = dep_time - 24 * 3600
            new_arr_time = arr_time - 24 * 3600
            shift = -1

        elif dep_time < 6 * 3600:
            new_dep_time = dep_time + 24 * 3600
            new_arr_time = arr_time + 24 * 3600
            shift = 1

        if shift != 0:
            edges[out_node][in_node].append((new_dep_time, new_arr_time, row.trip_id, shift))

        key_val = (out_main_station, out_node)
        if key_val not in transfer_edges_added:
            edges[out_main_station][out_node].append(('T', MIN_TRANSFER_TIME, None))
            transfer_edges_added.add(key_val)

        key_val = (in_node, in_main_station)
        if key_val not in transfer_edges_added:
            edges[in_node][in_main_station].append(('P', 0, None))
            transfer_edges_added.add(key_val)

        prev_row = row

    for out_node in edges:
        for in_node in edges[out_node]:
            edges[out_node][in_node].sort()

    edges = convert_nested_defaultdict(edges)
    
    print('end building edges')
    return edges

def generate_route_ids(timetable):
    trip_sequences = timetable.groupby('trip_id')['stop_id'].apply(tuple)

    unique_sequences = {seq: f"R{i+1}" for i, seq in enumerate(trip_sequences.unique())}
    
    return {trip_id: unique_sequences[seq] for trip_id, seq in trip_sequences.items()}

def convert_edge_times_to_timedelta(edges):
    start_time = time.time()
    for out_node in edges:
        for in_node in edges[out_node]:
            new_edge_list = []
            for edge in edges[out_node][in_node]:
                if len(edge) == 4:
                    dep_seconds, arr_seconds, trip_id, shifted_day = edge
                else:
                    dep_seconds, arr_seconds, trip_id = edge
                    shifted_day = 0
                service_day_shift = timedelta(days = shifted_day)

                if isinstance(dep_seconds, (int, float)) and isinstance(arr_seconds, (int, float)):
                    dep_time = timedelta(seconds = dep_seconds)
                    arr_time = timedelta(seconds = arr_seconds)
                else:
                    dep_time = dep_seconds
                    arr_time = arr_seconds

                new_edge_list.append((dep_time, arr_time, trip_id, service_day_shift))

            edges[out_node][in_node] = new_edge_list

    for out_node in edges:
        for in_node in edges[out_node]:
            edges[out_node][in_node].sort()
    
    print(time.time() - start_time,'Converting edges to td')
    return edges

def convert_nested_defaultdict(d):
    return {k: convert_nested_defaultdict(v) for k, v in d.items()} if isinstance(d, defaultdict) else d

def build_trip_service_days():
    trips, calendar = read_trips_file(), read_calendar_file()

    # Merge trips with calendar
    trips_with_service = trips.merge(calendar, on='service_id', how='inner')

    trips_with_service['start_date'] = (pd.to_datetime(trips_with_service['start_date'], format="%Y%m%d"))
    trips_with_service['end_date'] = (pd.to_datetime(trips_with_service['end_date'], format="%Y%m%d"))

    day_columns = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    day_values = trips_with_service[day_columns].values

    # Vectorized service days calculation
    service_days = [np.where(row)[0].tolist() for row in day_values]

    # Create dictionary
    trip_service_dict = dict(zip(
    trips_with_service['trip_id'],
    [
        {
            'service_days': sdays,
            'start_date': sdate,
            'end_date': edate
        }
        for sdays, sdate, edate in zip(
            service_days,
            trips_with_service['start_date'],
            trips_with_service['end_date']
        )
    ]
    ))

    return trip_service_dict

def build_stop_name_to_id():
    timetable = get_stops_df()
    timetable['station_id'] = 'U' + timetable['asw_node_id'].astype(str)

    # Group and check for duplicates
    grouped = timetable.groupby('stop_name')['station_id'].apply(set)

    # Create the final mapping
    stop_name_to_station_id = {}
    for stop_name, station_ids in grouped.items():
        if len(station_ids) == 1:
            stop_name_to_station_id[stop_name] = next(iter(station_ids))
        else:
            for station_id in station_ids:
                stop_name_to_station_id[f"{stop_name} ({station_id})"] = station_id
    
    return stop_name_to_station_id

def build_stop_id_to_name():
    stops = get_stops_df()
    return dict(zip(stops['stop_id'], stops['stop_name']))

def load_timetable_from_memory():
    return joblib.load(os.path.join(CACHE_FOLDER_PATH, 'timetable'))

def load_edges_from_memory():
    return joblib.load(os.path.join(CACHE_FOLDER_PATH, 'edges'))

def load_trip_service_days_from_memory():
    return joblib.load(os.path.join(CACHE_FOLDER_PATH, 'trip_service_days'))

def load_stop_id_to_name_from_memory():
    return joblib.load(os.path.join(CACHE_FOLDER_PATH, 'stop_id_to_name'))

def load_stop_name_to_id_from_memory():
    return joblib.load(os.path.join(CACHE_FOLDER_PATH, 'stop_name_to_id'))

def save_timetable_to_memory(timetable):
    return joblib.dump(timetable, os.path.join(CACHE_FOLDER_PATH, 'timetable'))

def save_edges_to_memory(edges):
    return joblib.dump(edges, os.path.join(CACHE_FOLDER_PATH, 'edges_prot_4'), compress=3, protocol=4)

def save_trip_service_days_to_memory(trip_service_days):
    return joblib.dump(trip_service_days, os.path.join(CACHE_FOLDER_PATH, 'trip_service_days'))

def save_stop_id_to_name_to_memory(stop_id_to_name):
    return joblib.dump(stop_id_to_name, os.path.join(CACHE_FOLDER_PATH, 'stop_id_to_name'))

def save_stop_name_to_id_to_memory(stop_name_to_id):
    return joblib.dump(stop_name_to_id, os.path.join(CACHE_FOLDER_PATH, 'stop_name_to_id'))

def get_timetable():
    if os.path.exists(os.path.join(CACHE_FOLDER_PATH, 'timetable')):
        return load_timetable_from_memory()
    timetable = build_timetable_df()
    save_timetable_to_memory(timetable) 
    return timetable

def get_edges():
    st = time.time()
    print('Fetching edges')
    #if os.path.exists(os.path.join(CACHE_FOLDER_PATH, 'edges_prot_4')):
    #    edges = load_edges_from_memory()
    edges = build_edges()
    #save_edges_to_memory(edges)
    print(time.time() - st, 'EDGED')
    return edges

def get_trip_service_days():
    #if os.path.exists(os.path.join(CACHE_FOLDER_PATH, 'trip_service_days')):
    #    return load_trip_service_days_from_memory()
    trip_service_days = build_trip_service_days() 
    #save_trip_service_days_to_memory(trip_service_days)
    return trip_service_days

def get_stop_id_to_name():
    if os.path.exists(os.path.join(CACHE_FOLDER_PATH, 'stop_id_to_name')):
        return load_stop_id_to_name_from_memory()
    stop_id_to_name = build_stop_id_to_name() 
    save_stop_id_to_name_to_memory(stop_id_to_name)
    return stop_id_to_name

def get_stop_name_to_id():
    if os.path.exists(os.path.join(CACHE_FOLDER_PATH, 'stop_name_to_id')):
        return load_stop_name_to_id_from_memory()
    stop_name_to_id = build_stop_name_to_id() 
    save_stop_name_to_id_to_memory(stop_name_to_id)
    return stop_name_to_id


abspath = os.path.abspath(__file__)
dirpath = os.path.dirname(abspath)
CACHE_FOLDER_PATH = os.path.join(dirpath, "cache")
MIN_TRANSFER_TIME = timedelta(seconds=120)

if __name__ == "__main__":
    #a = build_timetable_df()
    #t = build_timetable_df()
    #save_timetable_to_memory(a)
    #print(t.dtypes)
    #import timeit
    a = timeit.timeit(lambda: get_stops_df(), number=1)
    print(a)
    a = timeit.timeit(lambda: get_stops_df(), number=3)
    print(a/3)
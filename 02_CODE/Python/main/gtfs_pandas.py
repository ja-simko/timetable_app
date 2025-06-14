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
from unidecode import unidecode

def convert_str_to_datetime(day):
    origin = datetime(2000, 1, 1)
    day = datetime.strptime(str(day), '%Y%m%d')
    return (day - origin).total_seconds() / (24 * 3600)

def read_json_extra_info_stops():
    data = pd.read_json(os.path.join(CACHE_FOLDER_PATH, 'zastavky_names.json'))
    return data['stopGroups']

def read_stops_file():
    return pd.read_csv(os.path.join(GTFS_FOLDER_PATH, "stops.txt"))

def read_stop_times_file():
    return pd.read_csv(os.path.join(GTFS_FOLDER_PATH, "stop_times.txt"), low_memory=False)

def read_trips_file():
    return pd.read_csv(os.path.join(GTFS_FOLDER_PATH, "trips.txt"), low_memory=False)

def read_routes_file():
    return pd.read_csv(os.path.join(GTFS_FOLDER_PATH, "routes.txt"))

def read_calendar_file():
    return pd.read_csv(os.path.join(GTFS_FOLDER_PATH, "calendar.txt"), dtype={'start_date': int, 'end_date': int})

def convert_str_to_sec(timestamp: str) -> int:
    return sum(int(t) * 60 ** i for i, t in enumerate(reversed(timestamp.split(":"))))

def get_timetable_sample():
    timetable_sample = get_timetable()
    timetable_sample = timetable_sample[timetable_sample['route_type'].isin((0,1,3))]
    return timetable_sample

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
    timetable['main_station_id'] = timetable['stop_id'].str.extract(r"^(U\d+)[ZS]", expand=False).astype('string')
    timetable = timetable.dropna(subset=['main_station_id'])

    trip_id_to_route = generate_route_ids(timetable)

    timetable['route_id'] = timetable['trip_id'].map(trip_id_to_route)
    timetable['node_id'] = timetable['stop_id'] + '_' + timetable['route_id'] + '_' + timetable['route_short_name']

    timetable = timetable[
        ['trip_id', 'stop_id', 'node_id', 'main_station_id', 'departure_time', 'arrival_time', 'stop_sequence', 'route_short_name', 'route_type' 
        ]
    ].sort_values(by=['trip_id', 'stop_sequence'])
    return timetable

def build_stop_name_to_id(zone = None):
    stops_df = get_stops_df()
    if zone:
        stops_df = stops_df[(stops_df['zone_id'] == zone)]
    
    return dict(zip(stops_df['unique_name'], stops_df['main_station_id']))
 
def build_stop_id_to_name_and_platform():
    stops = get_stops_df()
    return dict(zip(stops['stop_id'], zip(stops['stop_name'], stops['platform_code'])))

def build_stop_id_to_coordinates():
    stops = get_stops_df()
    return dict(zip(stops['main_station_id'], zip(stops['stop_lat'], stops['stop_lon'])))

def build_stops_df():
    gtfs_stops = read_stops_file()
    json_stops = read_json_extra_info_stops()

    gtfs_stops = gtfs_stops[gtfs_stops['asw_node_id'].notna()]
    gtfs_stops = gtfs_stops[(gtfs_stops['location_type'] == 0)]

    gtfs_stops['asw_node_id'] = gtfs_stops['asw_node_id'].astype(int)
    gtfs_stops['asw_stop_id'] = gtfs_stops['asw_stop_id'].astype(int)

    stop_id_to_name = {id: station['uniqueName'] for station in json_stops for stop in station['stops'] for id in stop['gtfsIds']}

    gtfs_stops['unique_name'] = (gtfs_stops['stop_id']).map(stop_id_to_name).astype(str)

    node_stop_id_to_platform = {stop['id']: (stop['platform'] if 'platform' in stop else '') for station in json_stops for stop in station['stops']}

    gtfs_stops['sub_stop_id'] = gtfs_stops['asw_node_id'].astype(str) + '/' + gtfs_stops['asw_stop_id'].astype(str)

    gtfs_stops['platform_code'] = gtfs_stops['sub_stop_id'].map(node_stop_id_to_platform)

    gtfs_stops['main_station_id'] = gtfs_stops['stop_id'].apply(lambda x: x.split('Z')[0])

    return gtfs_stops[['stop_id', 'main_station_id', 'stop_name', 'unique_name', 'zone_id', 'stop_lat', 'stop_lon', 'asw_node_id', 'platform_code']]

def build_trip_service_days():
    trips, calendar = read_trips_file(), read_calendar_file()

    # Merge trips with calendar
    trips_with_service = trips.merge(calendar, on='service_id', how='inner')

    trips_with_service['start_date'] = trips_with_service['start_date'].astype(str)
    trips_with_service['end_date'] = trips_with_service['end_date'].astype(str)

    day_columns = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    day_values = trips_with_service[day_columns].values

    
    # Create a structured array/dictionary in one pass
    trip_service_dict = {
        trip_id: {
            'service_days': set(np.where(days)[0].tolist()),
            'start_date': start,
            'end_date': end
        }
        for trip_id, days, start, end in zip(
            trips_with_service['trip_id'],
            day_values,
            trips_with_service['start_date'],
            trips_with_service['end_date']
        )
    }

    return trip_service_dict

def build_edges(timetable = pd.DataFrame()):
    """
    Optimized version of build_edges with better performance and cleaner logic.
    """
    if timetable.empty:
        timetable = get_timetable()
    
    # Pre-allocate with better estimate
    edges = defaultdict(lambda: defaultdict(list))
    transfer_edges_added = set()
    
    # Convert to numpy arrays for faster iteration
    trip_ids = timetable['trip_id'].values
    node_ids = timetable['node_id'].values
    main_stations = timetable['main_station_id'].values
    departure_times = timetable['departure_time'].values
    arrival_times = timetable['arrival_time'].values
    
    # Constants
    DAY_SECONDS = 24 * 3600
    EARLY_MORNING = 6 * 3600
    
    # Process consecutive pairs efficiently
    prev_trip_id = None
    
    for i in range(len(timetable)):
        current_trip_id = trip_ids[i]
        
        # Skip first row of each trip
        if current_trip_id != prev_trip_id:
            prev_trip_id = current_trip_id
            continue
        
        # Get current and previous row data
        prev_idx = i - 1
        
        out_node, in_node = node_ids[prev_idx], node_ids[i]

        out_main_station, in_main_station = main_stations[prev_idx], main_stations[i]
        
        dep_time, arr_time = departure_times[prev_idx], arrival_times[i]
        
        # Add main edge
        edges[out_node][in_node].append((dep_time, arr_time, current_trip_id, 0))
    
        # Add time-shifted edges (vectorized logic)
        if dep_time >= DAY_SECONDS:
            edges[out_node][in_node].append((dep_time - DAY_SECONDS, arr_time - DAY_SECONDS,current_trip_id, -1))

        elif dep_time < EARLY_MORNING:
            edges[out_node][in_node].append((dep_time + DAY_SECONDS, arr_time + DAY_SECONDS, current_trip_id, 1))
        
        # Add transfer edges (only once per node)
        if out_node not in transfer_edges_added:
            edges[out_main_station][out_node].append(('T', MIN_TRANSFER_TIME, None))
            edges[in_node][in_main_station].append(('P', 0, None))
            transfer_edges_added.add(out_node)
    
    # Sort all edge lists in place
    for out_node in edges:
        for in_node in edges[out_node]:
            edges[out_node][in_node].sort()
    
    edges = convert_nested_defaultdict(edges)
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

def load_cached_data(filename):
    return joblib.load(os.path.join(CACHE_FOLDER_PATH, filename))

def save_cached_data(data, filename):
    return joblib.dump(data, os.path.join(CACHE_FOLDER_PATH, filename))

def get_timetable():
    if os.path.exists(os.path.join(CACHE_FOLDER_PATH, 'timetable')):
        return load_cached_data('timetable')
    timetable = build_timetable_df()
    save_cached_data(timetable, 'timetable') 
    return timetable

def get_edges():
    print('Fetching edges')
    st = time.time()
    edges = build_edges()
    print('Edges loaded in:', round(time.time() - st, 2), "s")
    return edges

def get_trip_service_days():
    #filename = 'trip_service_days'
    #if os.path.exists(os.path.join(CACHE_FOLDER_PATH, filename)):
     #   return load_cached_data(filename)
    trip_service_days = build_trip_service_days() 
    #save_cached_data(trip_service_days, filename)
    return trip_service_days

def get_stop_id_to_name():
    #filename = 'stop_id_to_name'
    #if os.path.exists(os.path.join(CACHE_FOLDER_PATH, filename)):
    #    return load_cached_data(filename)
    stop_id_to_name = build_stop_id_to_name_and_platform() 
    #save_cached_data(stop_id_to_name, filename)
    return stop_id_to_name

def get_stop_name_to_id(zone = None):
    #filename = 'stop_name_to_id'
    #if os.path.exists(os.path.join(CACHE_FOLDER_PATH, filename)):
    #    return load_cached_data(filename)
    stop_name_to_id = build_stop_name_to_id(zone) 
    #save_cached_data(stop_name_to_id, filename)
    return stop_name_to_id

def get_stops_df():
    filename =  'stops_df'
    if os.path.exists(os.path.join(CACHE_FOLDER_PATH, filename)):
        return load_cached_data(filename)
    stops_df = build_stops_df() 
    save_cached_data(stops_df, filename)
    return stops_df

def test_functions(function, *att, **kwargs):
    if kwargs:
        n = kwargs['n'] 
    else:
        n = 10

    if att:
        elapsed = timeit.timeit(partial(function, *att), number = n)
    else:
        elapsed = timeit.timeit(partial(function), number = n)
    print(function.__name__, round(elapsed/n, 10))

abspath = os.path.abspath(__file__)
dirpath = os.path.dirname(abspath)
CACHE_FOLDER_PATH = os.path.join(dirpath, "cache")
GTFS_FOLDER_PATH = os.path.normpath(os.path.join(dirpath, "..", "..", "PID_GTFS"))
MIN_TRANSFER_TIME = 120
from functools import partial


if __name__ == "__main__": 



    #import timeit
    #import cProfile
    #cProfile.run("build_edges_optimized_numpy()",sort='cumtime')
    # a = timeit.timeit(lambda: build_edges_optimized_numpy(), number=3)
    # print(a/3)

    # test_functions(fce)
    '''
    build_stops_df()
    build_stop_name_to_id()
    exit()
    fce = load_cached_data
    att = 'trip_service_days'
    test_functions(fce, att)

    fce = build_stop_id_to_name_and_platform
    test_functions(fce)
    '''


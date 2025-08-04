import pandas as pd
import numpy as np
import random
import joblib
import os
import re
import timeit
import time

from datetime import datetime, timedelta
from collections import defaultdict
from unidecode import unidecode
from rapidfuzz import process, fuzz

class StopNames:
    _platforms = {}
    _stop_id_to_names = {}
    _coordinates = {}
    _zones = {}
    _stop_zone = {}
    is_initialized = False

    @classmethod
    def initialize(cls, stops_df, timetable):
        cls.is_initialized = True

        if os.path.exists(os.path.join(CACHE_FOLDER_PATH, 'zones')):
            cls._zones = load_cached_data('zones')
        else:
            cls._zones = {}
            stops_df = stops_df[stops_df['zone_id'].notna()]
            stops_df.loc[:, 'zone_id'] = stops_df['zone_id'].astype(str)

            for zone in stops_df['zone_id'].unique():
                if len(zone) <= 2 and zone != '-':
                    cls._zones[zone] = stops_df[stops_df['zone_id'].str.contains(zone)]['unique_name'].tolist()
                    save_cached_data(cls._zones, 'zones')

        stops_df.loc[:, 'zone_id'] = stops_df['zone_id'].astype(str)
        cls._stop_zone = dict(zip(stops_df['main_station_id'], (stops_df['zone_id'].str.split(','))))

        cls._platforms = dict(zip(stops_df['stop_id'], stops_df['platform_code']))

        cls._stop_id_to_names = dict(zip(stops_df['stop_id'], stops_df['stop_name'])) | dict(zip(stops_df['main_station_id'], stops_df['stop_name']))

        cls._node_id_to_main_st_id = dict(zip(timetable['node_id'], timetable['main_station_id']))

        cls._name_to_main_ids = dict(zip(stops_df['unique_name'], stops_df['main_station_id']))

        cls._main_id_to_stop_ids = dict(zip(stops_df['main_station_id'], stops_df['stop_id']))

        cls._coordinates = dict(zip(stops_df['main_station_id'], zip(stops_df['stop_lat'], stops_df['stop_lon'])))

        cls._ascii_names_dict = {unidecode(name).lower(): v for name, v in cls._name_to_main_ids.items()}


    @classmethod
    def _ensure_initialized(cls):
        if not cls.is_initialized:
            stops_df = get_stops_df()
            timetable = get_timetable()
            cls.initialize(stops_df, timetable)

    @classmethod
    def get_platform_code(cls, id):
        cls._ensure_initialized()
        return cls._platforms.get(id)
    
    @classmethod
    def get_general_name_from_id(cls, id):
        cls._ensure_initialized()
        return cls._stop_id_to_names.get(id)
    
    @classmethod
    def get_main_id_from_node_id(cls, node_id):
        cls._ensure_initialized()
        return cls._node_id_to_main_st_id.get(node_id, node_id)
    
    @classmethod
    def get_stop_id_from_main_id(cls, name):
        cls._ensure_initialized()
        return cls._main_id_to_stop_ids.get(name)
    
    @classmethod
    def get_coordinates_lat_lon(cls, node_id):
        cls._ensure_initialized()
        station_id = cls._node_id_to_main_st_id.get(node_id, node_id)
        return cls._coordinates.get(station_id)
    
    @classmethod
    def get_name_to_main_id(cls, id):
        cls._ensure_initialized()
        return cls._name_to_main_ids.get(id)
    
    @classmethod
    def get_id_from_fuzzy_input_name(cls, user_input, threshold = 80):
        cls._ensure_initialized()

        best_match, score, _ = process.extractOne(unidecode(user_input).lower(), cls._ascii_names_dict.keys(), scorer = fuzz.ratio)

        if score >= threshold:
            return cls._ascii_names_dict[best_match]
        return None
    
    @classmethod
    def get_a_random_stop_name(cls, zone=None):
        cls._ensure_initialized()
        
        if zone:
            if zone in cls._zones:
                return random.choice(cls._zones[zone])
            print(f"Warning: Zone {zone} not found, using any station")
                
        return random.choice(list(cls._name_to_main_ids.keys()))

    @classmethod
    def get_available_zones(cls):
        cls._ensure_initialized()
        return list(cls._zones.keys())
    
def convert_str_to_datetime(day: str):
    day = datetime.strptime(day, '%Y%m%d')
    return day

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
    stop_times_routes = routes.merge(stop_times_trips, on='route_id', how='inner')
    
    timetable = stop_times_routes.copy()

    timetable['stop_id'] = timetable['stop_id'].astype('string')
    timetable['trip_id'] = timetable['trip_id'].astype('string')
    timetable['route_short_name'] = timetable['route_short_name'].astype('string')
    timetable['main_station_id'] = timetable['stop_id'].str.extract(r"^(U\d+)[ZS]", expand=False).astype('string')
    timetable = timetable.dropna(subset=['main_station_id'])

    trip_id_to_route = generate_route_ids(timetable)

    timetable['route_id'] = timetable['trip_id'].map(trip_id_to_route)
    timetable['node_id'] = timetable['stop_id']  + '_' + timetable['route_id'] + '_' + timetable['route_short_name']
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
    StopNames.initialize(stops, get_timetable())
    return stops

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

def build_edges(timetable, reverse = False):
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

        if reverse:
            out_node, in_node = in_node, out_node
            out_main_station, in_main_station = in_main_station, out_main_station
        
        dep_time, arr_time = departure_times[prev_idx], arrival_times[i]
        
        # Add main edge
        edges[out_node][in_node].append((dep_time, arr_time, current_trip_id, 0))
    
        # Add time-shifted edges (vectorized logic)
        if dep_time >= DAY_SECONDS:
            edges[out_node][in_node].append((dep_time - DAY_SECONDS, arr_time - DAY_SECONDS, current_trip_id, -1))

        elif dep_time < EARLY_MORNING:
            edges[out_node][in_node].append((dep_time + DAY_SECONDS, arr_time + DAY_SECONDS, current_trip_id, 1))
        
        # Add transfer edges (only once per node)A
        if out_node not in transfer_edges_added:
            edges[out_main_station][out_node].append(('T', MIN_TRANSFER_TIME, None))
            edges[in_node][in_main_station].append(('P', 0, None))
            transfer_edges_added.add(out_node)

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

def get_edges(timetable = pd.DataFrame()):
    print('Fetching edges.')
    st = time.time()
    edges = build_edges(timetable)
    print('Edges loaded in:', round(time.time() - st, 2), "s")
    return edges

def get_trip_service_days():
    trip_service_days = build_trip_service_days() 
    return trip_service_days

def get_stop_id_to_name():
    stop_id_to_name = build_stop_id_to_name_and_platform() 
    return stop_id_to_name

def get_stop_name_to_id(zone = None):
    stop_name_to_id = build_stop_name_to_id(zone) 
    return stop_name_to_id

def get_stops_df():
    filename =  'stops_df'
    if os.path.exists(os.path.join(CACHE_FOLDER_PATH, filename)):
        return load_cached_data(filename)
    stops_df = build_stops_df() 
    save_cached_data(stops_df, filename)
    return stops_df

abspath = os.path.abspath(__file__)
dirpath = os.path.dirname(abspath)
CACHE_FOLDER_PATH = os.path.join(dirpath, "cache")
GTFS_FOLDER_PATH = os.path.normpath(os.path.join(dirpath, "..", "..", "PID_GTFS"))
MIN_TRANSFER_TIME = 120



import heapq
import joblib
import pandas as pd
import bisect
import random
import re
import time
import string
import math
import os
import timeit

import datetime as dt
from datetime import timedelta, datetime
from collections import defaultdict
from rapidfuzz import process, fuzz
from unidecode import unidecode
from sqlalchemy import create_engine, text
from gtfs_pandas import *

#Helper Functions

def convert_str_to_sec(timestamp: str) -> int:
    return sum(int(t) * 60 ** i for i, t in enumerate(reversed(timestamp.split(":"))))

def convert_to_hh_mm_ss(time: int) -> str:
    return str(timedelta(seconds=time % (24*60*60)))

def match_stop_name_and_id(key: str, stop_name_id_dict: dict) -> str:
    return stop_name_id_dict.get(key, None)

def is_dominated(existing, new):
    return (existing['arrival_time'] - existing['start_departure_time'] <= new['arrival_time'] - new['start_departure_time']) and \
           (existing['transfers'] <= new['transfers'])

def get_id_from_best_name_match(stop_name_id_dict, user_input, threshold=80) -> str|None:
    """Find the best matching stop name using fuzzy search. Returns station ID, without Z and the the part after it."""

    # Get the best match using fuzzy search (threshold avoids bad matches)
    best_match, score, _ = process.extractOne(user_input, stop_name_id_dict.keys(), scorer=fuzz.ratio)

    if score >= threshold:
        return stop_name_id_dict[best_match].split('Z')[0]  # Get ID from dict
    print("No Good Match Found")
    return None  # No good match found

def load_data_from_memory():
    return joblib.load(CACHE_TIMETABLE)

def get_timetable_data(load_from_memory):
    if load_from_memory:
        print("Data are loaded from cache.")
        return load_data_from_memory()
    else:
        print("Data are being extracted.")
        return create_cache()

def create_cache() -> dict:
    cache = {}

    stop_name_to_station_id, stop_id_to_stop_name, stops_df  = load_stop_name_id_dicts()
    timetable = load_timetable()

    # Store them in the cache
    cache['timetable'] = timetable
    cache['stops_df'] = stops_df
    cache['stop_name_to_station_id'] = stop_name_to_station_id
    cache['stop_id_to_stop_name'] = stop_id_to_stop_name

    # Save to disk
    
    joblib.dump(cache, CACHE_TIMETABLE)
    joblib.dump(stop_name_to_station_id, CACHE_STOP_NAME_ID)
    print('end dumping')
    return cache

def load_timetable():
    print('start query for timetable')
    st = time.time()
    timetable = build_timetable_df()
    print('timetable',time.time() - st)
    # Convert 'departure_time' and 'arrival_time' to seconds
    #timetable['departure_time'] = timetable['departure_time'].apply(lambda timestamp: convert_str_to_sec(timestamp))
    #timetable['arrival_time'] = timetable['arrival_time'].apply(lambda timestamp: convert_str_to_sec(timestamp))

    print('end timetable')
    return timetable



def create_stop_name_to_station_id(timetable):
    df_mapped = timetable.copy()
    df_mapped['station_id'] = 'U' + df_mapped['asw_node_id'].astype(str)

    # Group and check for duplicates
    grouped = df_mapped.groupby('stop_name')['station_id'].apply(set)

    # Create the final mapping
    stop_name_to_station_id = {}
    for stop_name, station_ids in grouped.items():
        if len(station_ids) == 1:
            stop_name_to_station_id[stop_name] = next(iter(station_ids))
        else:
            for station_id in station_ids:
                stop_name_to_station_id[f"{stop_name} ({station_id})"] = station_id
    
    return stop_name_to_station_id

def load_stop_name_id_dicts() -> tuple[dict, dict, pd.DataFrame]:
    # Construct the dictionary for stop_name to stop_id
    dataframe = get_stops_df()

    stop_name_to_stations = defaultdict(set)
    stop_id_to_stop_name = {}

    BothTh = defaultdict(set)

    for _, row in dataframe.iterrows():
        stop_name = row['stop_name']
        station_id = 'U' + str(row['asw_node_id'])
        stop_name_to_stations[stop_name].add(station_id)
        stop_id_to_stop_name[row['stop_id']] = stop_name
    
    stop_name_to_station_id = {}

    for stop_name, station_ids in stop_name_to_stations.items():
        if len(station_ids) == 1:
            stop_name_to_station_id[stop_name] = next(iter(station_ids))
        else:
            for station_id in station_ids:
                stop_name_to_station_id[f"{stop_name} ({station_id})"] = station_id

    # Add the station_id to the original DataFrame
    dataframe['station_id'] = dataframe['stop_name'].map(lambda name: stop_name_to_station_id.get(name, None))

    return stop_name_to_station_id, stop_id_to_stop_name, dataframe

def get_trip_id_to_line_num_dict(timetable):
    """ Assigns a unique route ID to each unique stop sequence. """
    
    # Group trips by their ordered sequence of stops
    trip_id_to_route_name = {}

    for row in timetable.itertuples(index=False):
        trip_id_to_route_name[row.trip_id] = row.route_short_name

    return trip_id_to_route_name  # {trip_id: line_num}

def modified_dijkstra_pareto(start_station, target_station, start_time, edges, initial_max_transfers, trip_service_days):
    """ Implements the modified Dijkstra's algorithm to find all Pareto-optimal paths iteratively. """
    pq = [(start_time, 0, start_station)]  # (arrival_time, num_transfers, station)
    labels = {start_station: [(start_time, 0)]}  # Station → List of (arrival_time, num_transfers)
    evaluated_nodes = {start_station: {0: {'prev_node': None, 'departure_time': None, 'arrival_time': start_time, 'line_num': None}}}

    max_transfers = initial_max_transfers  # Start with the initial transfer limit

    #trip_service_days = get_trip_service_dict()
    #joblib.dump(trip_service_days, os.path.join(CACHE_PATH, 'trip_service_days'))

    start = time.time()

    checked_trips = set()
    while max_transfers >= 0:  # Run until we reach -1 transfers
        while pq:
            current_time, current_transfers, current_station = heapq.heappop(pq)

            if current_station == target_station:
                max_transfers = -1 if ONLY_FASTEST_TRIP else current_transfers
                break

            if current_transfers > max_transfers:
                continue  # Skip paths exceeding the allowed transfers

            if current_station not in edges:
                continue
         
            line_num = current_station.split('_')[-1] if '_' in current_station else None
        
            is_start_station = current_station == start_station

            for next_station, connections in edges[current_station].items():
                dep_time, arr_time, transfer = binary_search_next_edge(connections, current_time, is_start_station, current_transfers, max_transfers, trip_service_days, checked_trips)
        
                if arr_time is None or arr_time > start_time + TIME_WINDOW:
                    continue

                new_transfers = current_transfers + transfer

                # Check if this label is Pareto-optimal
                if next_station not in labels:
                    labels[next_station] = []

                is_dominated = False

                for prev_time, prev_transfers in labels[next_station]:
                    if prev_time <= arr_time and prev_transfers <= new_transfers:
                        is_dominated = True
                        break  # The new label is dominated, discard it

                if is_dominated:
                    continue
                
                labels[next_station] = [(t, tr) for (t, tr) in labels[next_station] if not (t >= arr_time and tr >= new_transfers)] #prolly a bit faster without this step (40ms per run?)
                labels[next_station].append((arr_time, new_transfers))

                # Update evaluated_nodes for each transfer count
                if next_station not in evaluated_nodes:
                    evaluated_nodes[next_station] = {}

                evaluated_nodes[next_station][new_transfers] = {
                    'prev_node': current_station,
                    'departure_time': dep_time,
                    'arrival_time': arr_time,
                    'line_num': line_num
                }

                heapq.heappush(pq, (arr_time, new_transfers, next_station))

        max_transfers -= 1

    global dijkstra_time
    dijkstra_time = round(time.time() - start,3)*1000
    print('dijkstra',round(time.time() - start,3)*1000,'ms')
    return evaluated_nodes

def get_time_within_day(td: timedelta) -> timedelta:
    """Get time part of timedelta (modulo 24 hours)"""
    days = td.days
    seconds = td.seconds
    return timedelta(seconds=seconds)

# Alternative using total_seconds()
def get_time_within_day_alt(td: timedelta) -> timedelta:
    """Get time part of timedelta (modulo 24 hours)"""
    total_seconds = int(td.total_seconds())
    return timedelta(seconds=total_seconds % (24*3600))

def binary_search_next_edge(edges, current_time, zero_transfer_cost, num_of_transfers, current_transfer_limit, trip_service_days, checked_trips) -> tuple[float, float, int]:
    """Return next edge departure time, arrival time and transfer 0 or 1 
    Finds the next train connection using binary search 
    """
    if edges[0][0] == 'P' or zero_transfer_cost:
        return (current_time, current_time, 0) #to station edge with no cost, or zeroth transfer edge
    
    if edges[0][0] == 'T':
        if num_of_transfers >= current_transfer_limit: 
            return (None, None, 0)
        return (current_time, current_time + MIN_TRANSFER_TIME, 1)
    
    index = bisect.bisect_left(edges, (current_time,))

    while True:
        if index == len(edges):
            return (None, None, 0)
        
        trip_id = edges[index][2]
        service_day_shift = edges[index][3]

        departure_day_shifted = departure_day_dt + service_day_shift
        trip_start_date = convert_str_date_to_datetime(str(trip_service_days[trip_id]['start_date']))
        trip_end_date = convert_str_date_to_datetime(str(trip_service_days[trip_id]['end_date']))
        service_days = trip_service_days[trip_id]['service_days']

        if (trip_id in checked_trips or
        (departure_day_shifted.weekday() in service_days and (trip_end_date >= departure_day_shifted >= trip_start_date))):
            checked_trips.add(trip_id)
            
            return (edges[index][0], edges[index][1], 0)
        index += 1
            # return (24*60*60 + edges[0][0], 24*60*60 + edges[0][1], 0)

#SHIFTs days

def is_transit_edge(edge) -> bool:
    return True if edge['line_num'] == None else False

def find_path_pareto(source_node, end_node, evaluated_nodes, num_of_transfers):
    path = []
    prev_node = end_node

    while prev_node != source_node:
 
        if num_of_transfers in evaluated_nodes[prev_node]:

            connection = evaluated_nodes[prev_node][num_of_transfers]

            path.append(connection)
            prev_node = connection['prev_node'] 

            num_of_transfers -= 1 if is_transit_edge(connection) else 0

        else:
            return None #no path of size of max_transfers transfers

    return path[::-1] #Return reversed path

def construct_final_path_table(path):
    connections = []

    stop_id_to_name = get_stop_id_to_name()

    for edge in path:
        node = edge['prev_node'].split('_')[0]
        
        if node[0] != 'U':
            continue

        platform = get_platform_letter_from_stop_id(node)
        
        stop_time = edge['departure_time']
        stop_name = match_stop_name_and_id(node, stop_id_to_name)

        if edge['line_num'] != None:
            connections[-1].append([stop_name, stop_time, edge['line_num'], platform])
        else:
            connections.append([])

    return connections

def get_platform_letter_from_stop_id(stop_id):
    letter_number = {N+1:L for N, L in enumerate(string.ascii_uppercase)} 

    platform = re.match(r".*[Z](\d+)", stop_id)

    if platform:
        platform = int(platform.group(1))
        platform = platform % 10 if platform > 100 else platform
        return letter_number.get(platform, 'XX')
    return None

def format_timedelta(td):
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds%(24*3600), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def not_main(departure_station_id, arrival_station_id, departure_time_seconds, edges, trip_service_days) -> str:
    all_found_connections = []

    print(departure_station_id)

    evaluated_nodes = modified_dijkstra_pareto(departure_station_id, arrival_station_id, departure_time_seconds, edges, TRANSFER_BOUND, trip_service_days)

    if not (arrival_station_id and departure_station_id) or arrival_station_id not in evaluated_nodes:
        print('No Route Found')
        return False, None

    least_transfers_for_EA = min(evaluated_nodes[arrival_station_id].keys(), key=lambda t: evaluated_nodes[arrival_station_id][t]['arrival_time'])

    for num_of_transfers in range(least_transfers_for_EA, -1, -1): #fastest paths with fewer transfers
        path = find_path_pareto(departure_station_id, arrival_station_id, evaluated_nodes, num_of_transfers)

        if path:
            connections = construct_final_path_table(path)
            all_found_connections.append(connections)
            full_results_bool = False
            if TO_PRINT_IN_TERMINAL:
                print("Num of transfers", num_of_transfers)
                for connection_counter, connection in enumerate(connections):
                    for stop_counter, stop in enumerate(connection):
                        if full_results_bool or stop_counter == 0 or stop_counter == len(connection) - 1:
                            print(stop[0], format_timedelta(stop[1]),stop[2],stop[3] )

                    print('')
                print('')
                print('Travel time', int((path[-1]['arrival_time'] - path[0]['departure_time']).total_seconds()//60),'min')
            else:
                print("ROUTE EXISTS")
    
    print(dijkstra_time,'ms')
                
    return True, all_found_connections

def select_random_stations():
    stop_name_to_id = joblib.load(CACHE_STOP_NAME_ID)
    return random.choice(list(stop_name_to_id.values())), random.choice(list(stop_name_to_id.values()))

def run_algorithm(departure_station_name, arrival_station_name, departure_time_timedelta, departure_day, edges, trip_service_days):
    global departure_day_dt

    departure_day_dt = convert_str_date_to_datetime(departure_day)
    departure_station_id, arrival_station_id = get_departure_and_arrival_id_from_name(departure_station_name, arrival_station_name)
    departure_time_seconds = departure_time_timedelta# datetime #convert_str_to_sec(departure_time_str)#convert_date_and_time_to_total_seconds(departure_day, departure_time_str)
    route_exists, all_found_connections = not_main(departure_station_id, arrival_station_id, departure_time_seconds, edges, trip_service_days)

    #print('One Run Time:', round((time.time() - start_time_alg)*1000, 2), 'ms')

    #print('Loading Time', loading_time, 's')
    #print('Total Time',round((time.time() - start_time_all), 2), 's')
    #print('Average dijkstra time',round((time.time() - start_time_dijkstra)*1000, 2)/NUM_OF_SEARCHES, 'ms')

    return route_exists, all_found_connections

def get_departure_and_arrival_id_from_name(departure_station_name, arrival_station_name):
    stop_name_to_id = joblib.load(CACHE_STOP_NAME_ID)

    stop_ascii_dict = {unidecode(k).lower(): v for k, v in stop_name_to_id.items()}

    departure_station_id = get_id_from_best_name_match(stop_ascii_dict, unidecode(departure_station_name).lower())
    arrival_station_id = get_id_from_best_name_match(stop_ascii_dict, unidecode(arrival_station_name).lower())

    return departure_station_id, arrival_station_id

def get_default_station_names():
    if RANDOM_STATIONS:
        departure_station_name, arrival_station_name = select_random_stations()
        departure_time_str = f"{random.randint(0,24)}:{random.randint(0,60)}:{random.randint(0,60)}"

    else:
        departure_station_name = "k juliane"
        arrival_station_name = "dedina"
        departure_time_str = '10:30:00'

    departure_day = '20250610'

    hours, minutes, seconds = map(int, departure_time_str.split(':'))
    departure_time_dt = dt.timedelta(hours=hours, minutes=minutes, seconds=seconds)

    return departure_station_name, arrival_station_name, departure_time_dt, departure_day

def convert_str_date_to_datetime(day: str):
    departure_date = datetime.strptime(str(day), '%Y%m%d').date()
    return departure_date

def main():
    departure_station_name, arrival_station_name, departure_time_dt, departure_day = get_default_station_names()
    
    st = time.time()
    edges = get_edges()
    print('Download edges',round(time.time() - st, 3))

    st = time.time()
    trip_service_days = get_trip_service_days()
    print('Download tripsserivce',round(time.time() - st, 3))

    for i in range(NUM_OF_SEARCHES):
        departure_station_name, arrival_station_name, departure_time_dt, departure_day = get_default_station_names()
        start_time = time.time()
        run_algorithm(departure_station_name, arrival_station_name, departure_time_dt, departure_day, edges, trip_service_days)
        print('One Run Time:', round((time.time() - start_time)*1000, 2), 'ms')


### experiment with multpiple runs, the next run stars second after the fastest or something 

# Define constants
MIN_TRANSFER_TIME = dt.timedelta(seconds = 120)#120
TIME_WINDOW = dt.timedelta(hours = 12)#12*60*60
TRANSFER_BOUND = 10
ONLY_FASTEST_TRIP = False
NUMBER_OF_DAYS_IN_ADVANCE = 14

NUM_OF_SEARCHES = 1 #for testing, number of searches
TO_PRINT_IN_TERMINAL = False if NUM_OF_SEARCHES > 1 else True

RANDOM_STATIONS = False

abspath = os.path.abspath(__file__)
dirpath = os.path.dirname(abspath)
CACHE_FOLDER_PATH = os.path.join(dirpath, "cache")


CACHE_FOLDER_PATH = r"C:\Users\Jachym\OneDrive - České vysoké učení technické v Praze\Bakalářská_práce\02_CODE\cache"
CACHE_TIMETABLE = r"C:\Users\Jachym\OneDrive - České vysoké učení technické v Praze\Bakalářská_práce\02_CODE\cache\timetable_cache.pkl"
CACHE_EDGES_DIR = r"C:\Users\Jachym\OneDrive - České vysoké učení technické v Praze\Bakalářská_práce\02_CODE\cache\edges"
CACHE_STOP_NAME_ID = r"C:\Users\Jachym\OneDrive - České vysoké učení technické v Praze\Bakalářská_práce\02_CODE\cache\stop_name_id.pkl"

def euclidean_distance(lat1, lon1, lat2, lon2):
    # Převod na radiány
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula pro sférickou vzdálenost
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371000  # Poloměr Země v metrech
    return c * r

def calculate_transit_time(distance, platform_diff=0):
    WALKING_SPEED = 1.4  # meters per second (5 km/h)
    BASE_PLATFORM_CHANGE_TIME = 60  # seconds for changing platforms
    
    # Basic walking time
    walking_time = distance / WALKING_SPEED + BASE_PLATFORM_CHANGE_TIME

    return int(walking_time)

def edges_between_stations():
    result = load_stop_name_id_dicts(ENGINE)[-1]

    stops_coordinates = {}

    for _, row in result.iterrows():
        stop_id = row['stop_id']
        stops_coordinates[stop_id] = (row['stop_lat'], row['stop_lon'])

    lat, lon = stops_coordinates['U482Z1P']
    for stop_id, (stop_lat, stop_lon) in stops_coordinates.items():
        if 'U482Z' in stop_id:
            stop_lat = result.loc[result['stop_id'] == stop_id, 'stop_lat'].values[0]
            stop_lon = result.loc[result['stop_id'] == stop_id, 'stop_lon'].values[0]
            distance = euclidean_distance(lat, lon, stop_lat, stop_lon)

            platform = result.loc[result['stop_id'] == stop_id, 'platform_code'].values[0]
            transit_time = calculate_transit_time(distance)
            print(f"Distance: {distance} meters, Platform: {platform}, ID: {stop_id}", transit_time)

if __name__ == "__main__":
    LOAD_FROM_MEMORY = False
    LOAD_FROM_MEMORY = True

    main()
    #build_stop_id_to_name()
    #create_stop_name_to_station_id(None)

    # get_14_days_from_today()
    #edges_between_stations()
    #build_edges_2()
    #departures = ['Palmovka','Lazarská',"Stadion Strahov",'Spořilov','Hůrka','Albertov','Sparta', 'Geologická','Nádraží Zahradní Město','Urxova']
    #arrivals = ['Lazarská',"Stadion Strahov",'Spořilov','Hůrka','Albertov','Sparta', 'Geologická','Nádraží Zahradní Město','Urxova', 'Palmovka']

    # departure_station_name = "Radlicka" #GOOD FOR SHOWING BOUNDS
    # arrival_station_name = "otakarova"
    # time_of_departure = '7:15:00'

    # departure_station_name = "prazskeho povstani" #GOOD FOR THREE DIFFERENT TRIPS PARETO
    # arrival_station_name = "nadrazi podbaba"
    # time_of_departure = '7:15:00'

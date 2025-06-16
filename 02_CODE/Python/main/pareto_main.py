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

def convert_sec_to_hh_mm_ss(time) -> str:
    time = int(time % (24 * 3600))
    hours, remainder = divmod(time, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def match_stop_name_and_id(key: str, stop_name_id_dict: dict) -> str:
    return stop_name_id_dict.get(key, None)

def is_dominated(existing, new):
    return (existing['arrival_time'] - existing['start_departure_time'] <= new['arrival_time'] - new['start_departure_time']) and \
           (existing['transfers'] <= new['transfers'])

def get_id_from_best_name_match(stop_name_to_id_ascii, user_input, threshold=80) -> str|None:
    """Find the best matching stop name using fuzzy search. Returns station ID, without Z and the the part after it."""

    # Get the best match using fuzzy search (threshold avoids bad matches)
    best_match, score, _ = process.extractOne(unidecode(user_input).lower(), stop_name_to_id_ascii.keys(), scorer=fuzz.ratio)

    if score >= threshold:
        return stop_name_to_id_ascii[best_match]  # Get ID from dict
    print("No Good Match Found")
    return None  # No good match found

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

def time_dependent_pareto_dijkstra(start_station, target_station, start_time, edges, trip_service_days, departure_day_dt):
    """ Implements the modified Dijkstra's algorithm to find all Pareto-optimal paths iteratively. """

    pq = [(0, start_time, 0, start_station)]  # (arrival_time, num_transfers, station)
    labels = {start_station: [(start_time, 0)]}  # Station → List of (arrival_time, num_transfers)
    evaluated_nodes = {start_station: {0: {'prev_node': None, 'departure_time': None, 'arrival_time': start_time}}}

    max_transfers = TRANSFER_BOUND  # Start with the initial transfer limit

    shifted_dates = {-1: departure_day_dt - timedelta(days=1), 0: departure_day_dt, 1: departure_day_dt + timedelta(days=1)
    }

    shifted_weekday = {k: v.weekday() for k, v in shifted_dates.items()}

    shifted_dates = {k: datetime.strftime(v, '%Y%m%d') for k,v in shifted_dates.items()}

    global checked_trips
    checked_trips = set()

    start = time.time()

    coordinates = build_stop_id_to_coordinates()

    a =get_stop_name_to_id()
    landmark = a['Šafránkova']
    
    t_lat, t_lon = coordinates[target_station]    
    preprocessed_paths = load_cached_data('preprocess_safrankova_all')
    counter = 0
    explored = {}
    using_landmarks = False
    using_star = False
    ONLY_FASTEST_TRIP = True
    while max_transfers >= 0:  # Run until we reach -1 transfers
        while pq:
            distance, current_time, current_transfers, current_station = heapq.heappop(pq)

            counter += 1
            explored[current_station] = (current_time, current_transfers)

            #if current_station == target_station:
            #    max_transfers = -1 if ONLY_FASTEST_TRIP else current_transfers
             #   break

            if current_transfers > max_transfers:
                continue  # Skip paths exceeding the allowed transfers

            if current_station not in edges: #if the station is always the end station
                continue

            #line_num = current_station.split('_')[-1] if '_' in current_station else None

            is_start_station = current_station == start_station

            for next_station, connections in edges[current_station].items():

                if explored.get(next_station, None):
                    if current_time > explored[next_station][0] and current_transfers >= explored[next_station][1]:
                        continue
                
                dep_time, arr_time, transfer = binary_search_next_edge(connections, current_time, is_start_station, trip_service_days, shifted_dates, shifted_weekday)

                if arr_time is None or arr_time > start_time + TIME_WINDOW:
                    continue
                
                new_transfers = current_transfers + transfer

                if new_transfers > max_transfers:
                    continue

                # Check if this label is Pareto-optimal
                if next_station not in labels:
                    labels[next_station] = []

                dominated = False  # Initialize the variable before use

                for prev_time, prev_transfers in labels[next_station]:
                    if prev_time <= arr_time and prev_transfers <= new_transfers:
                        dominated = True
                        break  # The new label is dominated, discard it

                if dominated:
                    continue

                labels[next_station] = [(time, transfer) for (time, transfer) in labels[next_station] if not (time >= arr_time and transfer >= new_transfers)] #prolly a bit faster without this step (40ms per run?)
                labels[next_station].append((arr_time, new_transfers))

                # Update evaluated_nodes for each transfer count
                if next_station not in evaluated_nodes:
                    evaluated_nodes[next_station] = {}
                
                if using_star:
                    curr_lat, curr_lon = coordinates.get(next_station.split('_')[0], (None, None))
                    if curr_lat:
                        distance = euclidean_distance(curr_lat, curr_lon, t_lat, t_lon)
                        reduced_cost = arr_time + distance/10#modifier #m/m/s
                    else:
                        reduced_cost = arr_time
                elif using_landmarks:
                    #curr_lat, curr_lon = coordinates.get(next_station.split('Z')[0], (None, None))
                    try:
                        dist_curr_to_L = preprocessed_paths[next_station.split('Z')[0]][landmark]
                        lower_bound = abs(dist_curr_to_L - dist_dest_to_L)
                        reduced_cost = arr_time + lower_bound*60
                    except:
                        reduced_cost = 0
                else:
                    reduced_cost = 0

                evaluated_nodes[next_station][new_transfers] = {
                    'prev_node': current_station,
                    'departure_time': dep_time,
                    'arrival_time': arr_time
                }

                heapq.heappush(pq, (reduced_cost, arr_time, new_transfers, next_station))

        max_transfers -= 1

    global dijkstra_time
    dijkstra_time = round(time.time() - start,3)*1000

    #print('counter',counter)
    return evaluated_nodes, explored

def is_trip_service_day_valid(departure_day, trip_id, trip_service_days, weekday):
    trip_info = trip_service_days[trip_id]
    return (weekday in trip_info['service_days'] and 
            trip_info['start_date'] <= departure_day <= trip_info['end_date'])

def binary_search_next_edge(edges, current_time, zero_transfer_cost, trip_service_days, shifted_dates, shifted_weekday) -> tuple[float, float, int]:
    """Return next edge departure time, arrival time and transfer 0 or 1 
    Finds the next train connection using binary search 
    """


    if edges[0][0] == 'P' or zero_transfer_cost:
        return (current_time, current_time, 0)
    
    if edges[0][0] == 'T':
        return (current_time, current_time + MIN_TRANSFER_TIME, 1)
    
    index = bisect.bisect_left(edges, (current_time,))

    while True:
        if index == len(edges):
            return (None, None, 0)
        
        dep_time, arr_time, trip_id, day_shift =  edges[index]

        if trip_id in checked_trips:
            return (dep_time, arr_time, 0)
        
        weekday = shifted_weekday[day_shift]
        departure_day = shifted_dates[day_shift]

        if is_trip_service_day_valid(departure_day, trip_id, trip_service_days, weekday):
            checked_trips.add(trip_id)
            return (dep_time, arr_time, 0)
        
        index += 1

def is_transit_edge(edge) -> bool:
    return True if edge['line_num'] == None else False

def find_path_pareto(source_node, end_node, evaluated_nodes, num_of_transfers):
    path = []
    prev_node = end_node
    while prev_node != source_node:
            if num_of_transfers in evaluated_nodes[prev_node]:
                edge = evaluated_nodes[prev_node][num_of_transfers]
                path.append(edge)
                prev_node = edge['prev_node'] 

                if len(prev_node.split('_')) == 1:
                    num_of_transfers -= 1
                    edge['line_num'] = None 
                else:
                    edge['line_num'] = prev_node.split('_')[-1]

            else:
                return None #no path of size of max_transfers transfers

    return path[::-1] #Return reversed path

def construct_final_path_table(path, stop_id_to_name):
    connections = []
   
    for edge in path:
        if not edge['line_num']:
            connections.append([])
            continue

        stop_id = edge['prev_node'].split('_')[0]

        stop_name = stop_id_to_name[stop_id][0]
        platform = stop_id_to_name[stop_id][1]

        connections[-1].append([stop_name, edge['departure_time'], edge['line_num'], platform])
    
    return connections

def get_travel_time_in_mins(path):
    return (path[-1]['arrival_time'] - path[0]['departure_time'])//60

def select_random_stations():
    stop_name_to_id = get_stop_name_to_id(zone = 'P')
    return random.choice(list(stop_name_to_id.keys())), random.choice(list(stop_name_to_id.keys()))

def run_program(departure_station_name, arrival_station_name, departure_time_str, departure_day, edges, trip_service_days):
    departure_day_dt = convert_str_to_datetime(departure_day)
    departure_time_sec = convert_str_to_sec(departure_time_str)
    departure_station_id, arrival_station_id = get_dep_arr_id_from_name(departure_station_name, arrival_station_name)

    if not (departure_station_id and arrival_station_id):
        print('Station name is not valid.')
        return False, None
    
    print("\n",departure_station_id)

    all_found_connections = []

    evaluated_nodes, _ = time_dependent_pareto_dijkstra(departure_station_id, arrival_station_id, departure_time_sec, edges, trip_service_days, departure_day_dt)

    print(len(evaluated_nodes))
    time.sleep(5)
    # for node, v in evaluated_nodes.items():
    #     for tr, info in v.items():
    #         try:
    #             print(node, convert_sec_to_hh_mm_ss(info['departure_time']))
    #         except:
    #             continue

    if arrival_station_id not in evaluated_nodes:
        print('No Route Found')
        return False, None

    # Filter out paths with same arrival time but more transfers
    seen_arrival_times = []
    num_of_transfers_per_path = []

    for transfers, edge in evaluated_nodes[arrival_station_id].items():
        arrival_time = edge['arrival_time']

        if arrival_time in seen_arrival_times:
            num_of_transfers_per_path[-1] = transfers
        else:
            num_of_transfers_per_path.append(transfers)
            seen_arrival_times.append(arrival_time)

    stop_id_to_name = get_stop_id_to_name()

    for num_of_transfers in num_of_transfers_per_path: #paths with that many transfers
        path = find_path_pareto(departure_station_id, arrival_station_id, evaluated_nodes, num_of_transfers)

#       if path:
        connections = construct_final_path_table(path, stop_id_to_name)
        all_found_connections.append(connections)
        full_results_bool = False

        if TO_PRINT_IN_TERMINAL:
            print("Num of transfers", num_of_transfers)
            for connection_counter, connection in enumerate(connections):
                for stop_counter, stop in enumerate(connection):
                    if full_results_bool or stop_counter == 0 or stop_counter == len(connection) - 1:
                        print(stop[0], convert_sec_to_hh_mm_ss(stop[1]), stop[2], stop[3])

                print('')
            print('')
            print('Travel time', get_travel_time_in_mins(path),'min')
        else:
            continue
            print("Route Exists")

    return True, all_found_connections

def get_dep_arr_id_from_name(departure_station_name, arrival_station_name): 
    stop_name_to_id = get_stop_name_to_id()
    stop_name_to_id_ascii = {unidecode(name).lower(): v for name, v in stop_name_to_id.items()}

    departure_station_id = get_id_from_best_name_match(stop_name_to_id_ascii, departure_station_name)

    arrival_station_id = get_id_from_best_name_match(stop_name_to_id_ascii, arrival_station_name)

    return departure_station_id, arrival_station_id

def get_default_station_names():
    if RANDOM_STATIONS:
        departure_station_name, arrival_station_name = select_random_stations()
        print(  departure_station_name, arrival_station_name )
        departure_time_str = f"{random.randint(0,24):02}:{random.randint(0,60):02}:{random.randint(0,60):02}"

    else:
        #departure_station_name = "sidliste petriny"
        #arrival_station_name = "holesovicka trznice"
        departure_station_name = "prazskeho povstani"
        arrival_station_name = "praha-podbaba"
        departure_time_str = '10:10:00'

    departure_day = '20250610'

    return departure_station_name, arrival_station_name, departure_time_str, departure_day

def convert_str_to_datetime(day: str):
    day = datetime.strptime(day, '%Y%m%d')
    return day

def main():
    edges = get_edges()
    trip_service_days = get_trip_service_days()
    print("Trips loaded")
    global_time = time.time()
    departures = ['Palmovka','Lazarská',"Stadion Strahov",'Spořilov','Hůrka','Albertov','Sparta', 'Geologická','Nádraží Zahradní Město','Urxova']
    arrivals = ['Lazarská',"Stadion Strahov",'Spořilov','Hůrka','Albertov','Sparta', 'Geologická','Nádraží Zahradní Město','Urxova', 'Palmovka']
    global modifier
    muj_set = set()
    modifiers = [100,6,5,4,3]
    for j in range(len(departures)):
        print(departures[j],arrivals[j])
        for i in range(NUM_OF_SEARCHES):
            modifier = modifiers[i]
            departure_station_name, arrival_station_name, departure_time_str, departure_day = get_default_station_names()
            start_time = time.time()

            departure_station_name = departures[j]
            arrival_station_name = arrivals[j]

            _, all_found_connections = run_program(departure_station_name, arrival_station_name, departure_time_str, departure_day, edges, trip_service_days)
            
            connections_str = str(all_found_connections)  # Convert to string to make hashable
            if i != 0:
                print(connections_str in muj_set, modifier)
            muj_set.add(connections_str)

            print('Dijkstra Time',dijkstra_time)
            print('One RunTime:', round((time.time() - start_time)*1000, 2), 'ms')
        print('Global Avg Run Time:', round((time.time() - global_time)*1000/NUM_OF_SEARCHES, 2), 'ms')
        print(len(muj_set))


### experiment with multpiple runs, the next run stars second after the fastest or something 

# Define constants
MIN_TRANSFER_TIME = 120
TIME_WINDOW = 12*60*60
TRANSFER_BOUND = 7
ONLY_FASTEST_TRIP = False
NUMBER_OF_DAYS_IN_ADVANCE = 14

NUM_OF_SEARCHES = 1 #for testing, number of searches
TO_PRINT_IN_TERMINAL = False if NUM_OF_SEARCHES > 1 or __name__ != "__main__" else True

RANDOM_STATIONS = False
random.seed(10)

abspath = os.path.abspath(__file__)
dirpath = os.path.dirname(abspath)
CACHE_FOLDER_PATH = os.path.join(dirpath, "cache")

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

def preprocess(landmark):
    tmtb_sample = get_timetable()
    stop_ids = tmtb_sample['stop_id'].unique().tolist()
    edges = build_edges(tmtb_sample)
    
    dep_time = 10*60*60

    trips = get_trip_service_days()
    start = 'U236'
    end = 'U597'
    preprocessed_paths = {}

    global using_star
    global using_landmarks
    using_star = False
    using_landmarks = True
    departure_day_dt = convert_str_to_datetime('20250610')

    print(len(stop_ids))

    evaluated_nodes, _ = time_dependent_pareto_dijkstra(start_station=landmark, target_station=end, start_time=dep_time,edges=edges,trip_service_days=trips, departure_day_dt=departure_day_dt)
    print(evaluated_nodes)
    for node, val in evaluated_nodes.items():
        a = list(val.keys())[0]
        if node != landmark:
            path = find_path_pareto(landmark, node, evaluated_nodes, num_of_transfers = a)
            if path:
                travelTime = get_travel_time_in_mins(path)
                preprocessed_paths[node] = {landmark: travelTime}
        else:
            preprocessed_paths[node] = {landmark: 0}
    
    print(preprocessed_paths)

    return preprocessed_paths

if __name__ == "__main__":
    #main()

    print(len(build_edges().keys()))
    tmtb_sample = get_timetable_sample()
    stop_ids = tmtb_sample['stop_id'].unique().tolist()
    edges = build_edges(tmtb_sample)
    
    dep_time = 10*60*60

    trips = get_trip_service_days()
    #build_stop_id_to_name()
    #create_stop_name_to_station_id(None)

    # get_14_days_from_today()
    #edges_between_stations()
    #build_edges_2()

    #global departures
    #global arrivals
    #departures = ['Palmovka','Lazarská',"Stadion Strahov",'Spořilov','Hůrka','Albertov','Sparta', 'Geologická','Nádraží Zahradní Město','Urxova']
    #arrivals = ['Lazarská',"Stadion Strahov",'Spořilov','Hůrka','Albertov','Sparta', 'Geologická','Nádraží Zahradní Město','Urxova', 'Palmovka']
    landmark = 'U321' #paloucek530, #strazni714
    start = random.choice(stop_ids)
    end = random.choice(stop_ids)
    dep_time = 10*60*60

    preproces = True
    if preproces:
        preprocessed_paths = preprocess(landmark)
        save_cached_data(preprocessed_paths, 'preprocess_321')
        print('Processed')
    else:
        preprocessed_paths = load_cached_data('preprocess_321')

    global dist_dest_to_L
    ONLY_FASTEST_TRIP = True

    n = 100

    departures = ['Nádraží Zahradní Město', 'Urxova','Palmovka','Lazarská',"Stadion Strahov",'Spořilov','Hůrka','Albertov','Sparta', 'Geologická',]
    arrivals = ['Lazarská',"Stadion Strahov",'Spořilov','Hůrka','Albertov','Sparta', 'Geologická','Nádraží Zahradní Město','Urxova', 'Palmovka']

    departures = [*(random.choice(stop_ids) for i in range(n))]
    arrivals = [*(random.choice(stop_ids) for i in range(n))]

    stop_name_to_id = get_stop_name_to_id()
    stop_name_to_id_ascii = {unidecode(name).lower(): v for name, v in stop_name_to_id.items()}

    #departures = [get_id_from_best_name_match(stop_name_to_id_ascii,name) for name in departures]
    #arrivals = [get_id_from_best_name_match(stop_name_to_id_ascii,name) for name in arrivals]
    departure_day_dt=convert_str_to_datetime('20250610')

    starttime = time.time()
    for i in range(n):
        start = departures[i%len(departures)]
        end = arrivals[i%len(arrivals)]
        using_star = False
        using_landmarks = True
        if end in preprocessed_paths:
            dist_dest_to_L = preprocessed_paths[end][landmark]
        else:
            print('O', end)
            continue

        evaluated_nodes, explored = time_dependent_pareto_dijkstra(start,end, dep_time, edges, trips, departure_day_dt)
        if end not in evaluated_nodes:
            print(start, end)
            print('No Route Found')
            continue
        max_transfers = max(evaluated_nodes[end].keys())
        path = find_path_pareto(start,  end, evaluated_nodes, num_of_transfers=max_transfers)
    print('Landmark', (time.time() - starttime)/n)

    starttime = time.time()
    for i in range(n):
        start = departures[i%len(departures)]
        end = arrivals[i%len(arrivals)]
        using_star = True
        using_landmarks = False

        evaluated_nodes, explored = time_dependent_pareto_dijkstra(start,end, dep_time, edges, trips, departure_day_dt)
        if end not in evaluated_nodes:
            print('No Route Found')
            continue
        max_transfers = max(evaluated_nodes[end].keys())
        path = find_path_pareto(start, end, evaluated_nodes, num_of_transfers=max_transfers)
    print('Star',(time.time() - starttime)/n)

    starttime = time.time()
    for i in range(n):
        start = departures[i%len(departures)]
        end = arrivals[i%len(arrivals)]
        using_star = False
        using_landmarks = False

        evaluated_nodes, explored = time_dependent_pareto_dijkstra(start,end, dep_time, edges, trips, departure_day_dt)
        if end not in evaluated_nodes:
            print('No Route Found')
            continue
        max_transfers = max(evaluated_nodes[end].keys())
        path = find_path_pareto(start, end, evaluated_nodes, num_of_transfers=max_transfers)
    print('Plain',(time.time() - starttime)/n)




    # id_to_name = get_stop_id_to_name()
    # for stop in explored:
    #     name = id_to_name.get(stop[0].split('_')[0], None)
    #     if name:
    #         print(name, stop[1],stop[2])
    #print(preprocessed_paths)

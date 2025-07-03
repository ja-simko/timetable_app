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

def get_shifted_days_dicts(departure_day_dt):
    shifted_dates = {-1: departure_day_dt - timedelta(days=1), 0: departure_day_dt, 1: departure_day_dt + timedelta(days=1)}

    return {k: (v.strftime('%Y%m%d'), v.weekday()) for k, v in shifted_dates.items()}

def time_dependent_pareto_dijkstra(start_station, target_station, start_time, edges, trip_service_days, departure_day_dt):
    """ Implements the modified Dijkstra's algorithm to find all Pareto-optimal paths iteratively. """

    pq = [(start_time, start_time, 0, start_station)]  # (reduced cost, arrival_time, num_transfers, station)
    labels = {start_station: [(start_time, 0)]}  # Station → List of (arrival_time, num_transfers)
    evaluated_nodes = {start_station: {0: {'prev_node': None, 'departure_time': None, 'arrival_time': start_time}}}

    max_transfers = TRANSFER_BOUND  # Start with the initial transfer limit

    # Create a mapping: {shift: (YYYYMMDD, weekday)}
    shifted_dates = get_shifted_days_dicts(departure_day_dt)

    global checked_trips
    checked_trips = set()

    #coordinates = build_stop_id_to_coordinates()

    t_lat, t_lon = StopNames.get_coordinates_lat_lon(target_station)    
    #preprocessed_paths = load_cached_data('preprocess_zatec_zasmuky_komarov_bakov_n_jizerou')
    taken_from_pq = 0
    global settled
    settled = set()
    ONLY_FASTEST_TRIP = False

    #dist_to_landmarks_target = {landmark:dist for landmark, dist in preprocessed_paths[target_station.split('Z')[0]].items()}

    start = time.time()
    global already_landmarked
    already_landmarked = {}
    discarded_cuz_settled = 0
    longer_than_tent_path = 0
    already_evaluated = 0
    
    while max_transfers >= 0:  # Run until we reach -1 transfers
        while pq:
            current_travel_time, current_time, current_transfers, current_station = heapq.heappop(pq)
            taken_from_pq += 1
            #print(reduced_cost, current_time, current_transfers, StopNames.get_general_name_from_id(current_station.split('_')[0]))

            settled.add((current_station, current_transfers))

            if current_station == target_station:
                max_transfers = -1 if ONLY_FASTEST_TRIP else current_transfers
                break

            if current_transfers > max_transfers:
                continue  # Skip paths exceeding the allowed transfers

            if current_station not in edges: #if the station is always the end station
                continue

            #line_num = current_station.split('_')[-1] if '_' in current_station else None

            is_start_station = current_station == start_station

            for next_station, connections in edges[current_station].items():
                
                if (next_station, current_transfers) in settled:
                    continue
                
                dep_time, arr_time, transfer = binary_search_next_edge(connections, current_time, is_start_station, trip_service_days, shifted_dates)

                if arr_time is None or arr_time > start_time + TIME_WINDOW:
                    continue

                edge_travel_time = (arr_time - current_time)

                new_transfers = current_transfers + transfer

                if new_transfers > max_transfers:
                    continue

                # Check if this label is Pareto-optimal
                if next_station not in labels:
                    labels[next_station] = []

                else:
                    dominated = False  # Initialize the variable before use

                    for prev_time, prev_transfers in labels[next_station]:
                        if prev_time <= arr_time and prev_transfers <= new_transfers:
                            dominated = True
                            break  # The new label is dominated, discard it

                    if dominated:
                        continue

                    labels[next_station] = [(time, transfer) for (time, transfer) in labels[next_station] if not (time >= arr_time and transfer >= new_transfers)] #prolly a bit faster without this step (40ms per run?)

                labels[next_station].append((arr_time, new_transfers))

                if next_station not in evaluated_nodes:
                    evaluated_nodes[next_station] = {}
                
                if using_star:
                    try:
                        curr_lat, curr_lon = StopNames.get_coordinates_lat_lon(next_station)
                    except:
                        print(next_station)
                    if curr_lat:
                        curr_difference = euclidean_distance(curr_lat, curr_lon, t_lat, t_lon)
                        lower_bound_in_seconds = math.floor(curr_difference/15)

                        new_travel_time = current_travel_time + edge_travel_time 

                        #Condition
                        # if current_travel_time + edge_travel_time + lower_bound_in_seconds > tentative_best_path:
                        #     longer_than_tent_path += 1
                        #     continue
                    else:
                        new_travel_time = current_travel_time + edge_travel_time
                
                elif using_landmarks:
                    #curr_lat, curr_lon = coordinates.get(next_station.split('Z')[0], (None, None))
                    main_id = spec_node_to_main.get(current_station, current_station)
                    if main_id in already_landmarked:
                        current_travel_time = arr_time + already_landmarked[main_id]

                    elif main_id in preprocessed_paths:
                        this_node = preprocessed_paths[main_id]
                        max_lower_bound = max(abs(int(dist) - int(dist_to_landmarks_target.get(landmark, dist)))
                        for landmark, dist in this_node.items())
                        if max_lower_bound > 0:
                            print('f', max_lower_bound)
                            current_travel_time = arr_time + max_lower_bound*60
                            already_landmarked[main_id] = max_lower_bound*60
                        else:
                            current_travel_time = arr_time 
                            already_landmarked[main_id] = 0

                    else:
                        current_travel_time = arr_time
                        already_landmarked[main_id] = 0

                else:
                    new_travel_time = current_travel_time + edge_travel_time

                evaluated_nodes[next_station][new_transfers] = {
                    'prev_node': current_station,
                    'departure_time': dep_time,
                    'arrival_time': arr_time,
                    'is_transfer': True if transfer else False
                }

                heapq.heappush(pq, (new_travel_time, arr_time, new_transfers, next_station))
        max_transfers -= 1
    print('discarded_cuz_settled',discarded_cuz_settled)
    print('longer_than_tent_path',longer_than_tent_path)
    print('already_eval',already_evaluated)
    global dijkstra_time
    dijkstra_time = round(time.time() - start,3)*1000
    print(dijkstra_time)
    #print('counter',counter)
    return evaluated_nodes, settled

def is_trip_service_day_valid(departure_day, trip_id, trip_service_days, weekday):
    trip_info = trip_service_days[trip_id]
    return (weekday in trip_info['service_days'] and 
            trip_info['start_date'] <= departure_day <= trip_info['end_date'])

def binary_search_next_edge(edges, current_time, zero_transfer_cost, trip_service_days, shifted_dates) -> tuple[float, float, int]:
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
        
        departure_day, weekday = shifted_dates[day_shift]

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

            if edge['is_transfer']:
                num_of_transfers -= 1
                edge['line_num'] = None 
            else:
                edge['line_num'] = prev_node.split('_')[-1]

        else:
            return None #no path of size of max_transfers transfers

    return path[::-1] #Return reversed path

def construct_final_path_table(path):
    connections = [[]]
   
    for edge in path[1:]:
        if edge['is_transfer']:
            connections.append([])
            continue
        stop_id = edge['prev_node'].split('_')[0]

        stop_name = StopNames.get_general_name_from_id(stop_id)
        platform = StopNames.get_platform_code(stop_id)

        connections[-1].append([stop_name, edge['departure_time'], edge['line_num'], platform])
    
    return connections

def get_travel_time_in_mins(path):
    return (path[-1]['arrival_time'] - path[1]['departure_time'])//60

def select_random_stations(zone = None):
    return (StopNames.get_a_random_stop_name(zone), StopNames.get_a_random_stop_name(zone))

def run_program(departure_station_name, arrival_station_name, departure_time_str, departure_day, edges, trip_service_days):
    departure_day_dt = convert_str_to_datetime(departure_day)
    departure_time_sec = convert_str_to_sec(departure_time_str)
    departure_station_id = StopNames.get_id_from_fuzzy_input_name(departure_station_name)
    arrival_station_id = StopNames.get_id_from_fuzzy_input_name(arrival_station_name)
    print(departure_station_id, arrival_station_id)
    global using_star
    using_star = True

    if not (departure_station_id and arrival_station_id):
        print('Station name is not valid.')
        return False, None
    
    print("\n",departure_station_id)

    all_found_connections = []

    evaluated_nodes, _ = time_dependent_pareto_dijkstra(departure_station_id, arrival_station_id, departure_time_sec, edges, trip_service_days, departure_day_dt)

    print('eval',len(evaluated_nodes))
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

    print(evaluated_nodes[arrival_station_id].items())
    for transfers, edge in sorted(evaluated_nodes[arrival_station_id].items(), reverse = True):
        arrival_time = edge['arrival_time']
        if seen_arrival_times and arrival_time < seen_arrival_times[-1]:
            num_of_transfers_per_path[-1] = transfers
        else:
            num_of_transfers_per_path.append(transfers)
            seen_arrival_times.append(arrival_time)

    for num_of_transfers in num_of_transfers_per_path: #paths with that many transfers
        path = find_path_pareto(departure_station_id, arrival_station_id, evaluated_nodes, num_of_transfers)

#       if path:
        connections = construct_final_path_table(path)
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

def get_default_station_names():
    if RANDOM_STATIONS:
        departure_station_name, arrival_station_name = select_random_stations()
        print(  departure_station_name, arrival_station_name )
        departure_time_str = f"{random.randint(0,24):02}:{random.randint(0,60):02}:{random.randint(0,60):02}"

    else:
        #departure_station_name = "sidliste petriny"
        #arrival_station_name = "holesovicka trznice"
        departure_station_name = "k juliane"
        arrival_station_name = "dedina"
        departure_time_str = '14:36:00'

    departure_day = '20250611'

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
    #for j in range(len(departures)):
     #   print(departures[j],arrivals[j])
      #  for i in range(NUM_OF_SEARCHES):
    #modifier = modifiers[i]
    departure_station_name, arrival_station_name, departure_time_str, departure_day = get_default_station_names()
    tmtb = get_timetable()
    global spec_node_to_main
    spec_node_to_main = {node: main for node, main in zip(tmtb['node_id'], tmtb['main_station_id'])}
    #departure_station_name = departures[j]
    #arrival_station_name = arrivals[j]
    StopNames.initialize(get_stops_df(), tmtb)
    
    start_time = time.time()
    _, all_found_connections = run_program(departure_station_name, arrival_station_name, departure_time_str, departure_day, edges, trip_service_days)
    
    #connections_str = str(all_found_connections)  # Convert to string to make hashable
    #if i != 0:
        #print(connections_str in muj_set, modifier)
    #muj_set.add(connections_str)

    print('Dijkstra Time',dijkstra_time)
    print('One RunTime:', round((time.time() - start_time)*1000, 2), 'ms')
    print('Global Avg Run Time:', round((time.time() - global_time)*1000/NUM_OF_SEARCHES, 2), 'ms')
    print(len(muj_set))


### experiment with multpiple runs, the next run stars second after the fastest or something 

# Define constants
MIN_TRANSFER_TIME = 120
TIME_WINDOW = 5*60*60
TRANSFER_BOUND = 6
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

def preprocess():
    tmtb_sample = get_timetable()
    stop_ids = tmtb_sample['stop_id'].unique().tolist()
    main_station_ids = tmtb_sample['main_station_id'].unique().tolist()
    edges = build_edges(tmtb_sample)
    
    dep_time = 10*60*60

    trips = get_trip_service_days()
    landmarks = ['U5079', 'U9830', 'U9989', 'U6210']   #zatec, zaskmuky, komarov, bakov n. jizerou
    end = 'U597'
    preprocessed_paths = defaultdict(dict)

    global using_star
    global using_landmarks
    using_star = False
    using_landmarks = False
    departure_day_dt = convert_str_to_datetime('20250610')

    print(len(stop_ids))

    for landmark in landmarks:
        print('L', landmark)
        evaluated_nodes, _ = time_dependent_pareto_dijkstra(start_station=landmark, target_station=end, start_time=dep_time,edges=edges,trip_service_days=trips, departure_day_dt=departure_day_dt)

        for node, val in evaluated_nodes.items():
            max_tr = max(val.keys())
            if node in main_station_ids:
                if node != landmark:
                    path = find_path_pareto(landmark, node, evaluated_nodes, num_of_transfers = max_tr)
                    if path:
                        travelTime = get_travel_time_in_mins(path)
                        preprocessed_paths[node][landmark] = travelTime
                else:
                    preprocessed_paths[node][landmark] = 0
        
    print(preprocessed_paths)

    return preprocessed_paths

if __name__ == "__main__":
    # e = get_edges()
    # Print all keys in 'e' that start with 'U530Z1P' (case-insensitive) and end with anything
    # pattern = re.compile(r'^U693Z1P.*', re.IGNORECASE) #693
    # for key, v in e.items():
    #     if pattern.match(key):
    #         for a, times in v.items():
    #             print(key, e[key],'\n')

    # exit()




    global using_landmarks
    global using_star
    using_landmarks = False
    using_star = True
    main()
    id_to_name = get_stop_id_to_name()
    for stop, tr in settled:
        name = StopNames._stop_id_to_names.get(stop.split('_')[0], None)
        if name:
            continue
            time.sleep(0.02)
            print(name, stop)
            #print(already_landmarked[stop.split('Z')[0]])
    exit()
    tmtb_sample = get_timetable_sample()
    main_stops_id = tmtb_sample['main_station_id'].unique().tolist()
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
    start = random.choice(main_stops_id)
    end = random.choice(main_stops_id)
    dep_time = 10*60*60

    preproces = False
    if preproces:
        preprocessed_paths = preprocess()
        save_cached_data(preprocessed_paths, 'preprocess_zatec_zasmuky_komarov_bakov_n_jizerou')
        print('Processed')
    else:
        preprocessed_paths = load_cached_data('preprocess_zatec_zasmuky_komarov_bakov_n_jizerou')

    global dist_dest_to_L
    ONLY_FASTEST_TRIP = True

    n = 100

    departures = ['Nádraží Zahradní Město', 'Urxova','Palmovka','Lazarská',"Stadion Strahov",'Spořilov','Hůrka','Albertov','Sparta', 'Geologická',]
    arrivals = ['Lazarská',"Stadion Strahov",'Spořilov','Hůrka','Albertov','Sparta', 'Geologická','Nádraží Zahradní Město','Urxova', 'Palmovka']

    departures = [*(random.choice(main_stops_id) for i in range(n))]
    arrivals = [*(random.choice(main_stops_id) for i in range(n))]

    stop_name_to_id = get_stop_name_to_id()
    stop_name_to_id_ascii = {unidecode(name).lower(): v for name, v in stop_name_to_id.items()}

    #departures = [get_id_from_best_name_match(stop_name_to_id_ascii,name) for name in departures]
    #arrivals = [get_id_from_best_name_match(stop_name_to_id_ascii,name) for name in arrivals]
    departure_day_dt=convert_str_to_datetime('20250610')

    starttime = time.time()
    for i in range(n):
        print(i)
        start = departures[i%len(departures)]
        end = arrivals[i%len(arrivals)]

        using_star = False
        using_landmarks = True
        if end in preprocessed_paths:
            pass
        else:
            print('O', end)
            continue

        evaluated_nodes, settled = time_dependent_pareto_dijkstra(start,end, dep_time, edges, trips, departure_day_dt)
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

        evaluated_nodes, settled = time_dependent_pareto_dijkstra(start,end, dep_time, edges, trips, departure_day_dt)
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

        evaluated_nodes, settled = time_dependent_pareto_dijkstra(start,end, dep_time, edges, trips, departure_day_dt)
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

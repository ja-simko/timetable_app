#profile
import heapq
import bisect
import random
import time
import math
import os

from datetime import timedelta, datetime
from collections import defaultdict
from rapidfuzz import process, fuzz
from unidecode import unidecode
from gtfs_pandas import *
from main_pareto_paths import get_default_journey_info

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

def get_shifted_days_dicts(departure_day: str):
    departure_day_dt = convert_str_to_datetime(departure_day)

    shifted_dates = {-1: departure_day_dt - timedelta(days=1), 0: departure_day_dt, 1: departure_day_dt + timedelta(days=1)
    }

    return {k: (v.strftime('%Y%m%d'), v.weekday()) for k, v in shifted_dates.items()}

def time_dependent_pareto_dijkstra(start_station, target_station, start_time, edges, trip_service_days, departure_day, tentative_best_path, pq, evaluated_nodes, gui = True if __name__ != "__main__" else False):
    """ Implements the modified Dijkstra's algorithm to find all Pareto-optimal paths iteratively. """
    if not pq:
        pq = [(0, start_time, 0, start_station)]  # (reduced cost, arrival_time, num_transfers, station)
        evaluated_nodes = {start_station: {'prev_node': None, 'departure_time': None, 'arrival_time': start_time}}
    else:
        heapq.heappush(pq, (0, start_time, 0, start_station))
    
    labels = {start_station: [(start_time, 0)]}  # Station → List of (arrival_time, num_transfers)

    max_transfers = TRANSFER_BOUND  # Start with the initial transfer limit

    # Create a mapping: {shift: (YYYYMMDD, weekday)}
    shifted_dates = get_shifted_days_dicts(departure_day)

    global checked_trips
    checked_trips = set()

    #coordinates = build_stop_id_to_coordinates()

    t_lat, t_lon = StopNames.get_coordinates_lat_lon(target_station)    
    #preprocessed_paths = load_cached_data('preprocess_zatec_zasmuky_komarov_bakov_n_jizerou')
    taken_from_pq = 0
    global settled
    settled = defaultdict(set)
    ONLY_FASTEST_TRIP = True

    #dist_to_landmarks_target = {landmark:dist for landmark, dist in preprocessed_paths[target_station.split('Z')[0]].items()}

    start = time.time()
    global already_landmarked
    already_landmarked = {}
    while max_transfers >= 0:  # Run until we reach -1 transfers
        while pq:
            current_travel_time, current_time, current_transfers, current_station = heapq.heappop(pq)
            taken_from_pq += 1

            if current_travel_time > tentative_best_path:
                max_transfers = -1 if ONLY_FASTEST_TRIP else current_transfers
                break

            settled[current_station].add(current_transfers)

            if current_station == target_station:
                max_transfers = -1 if ONLY_FASTEST_TRIP else current_transfers
                tentative_best_path = current_time - start_time
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

                if StopNames.get_main_id_from_node_id(current_station) == start_station and start_time != dep_time:
                    break
                
                if evaluated_nodes.get(next_station) and evaluated_nodes[next_station]['arrival_time'] <= arr_time:
                    continue

                if settled.get(next_station) and min(settled[next_station]) <= new_transfers:
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
                
                if gui or using_star:
                    try:
                        curr_lat, curr_lon = StopNames.get_coordinates_lat_lon(next_station)
                    except:
                        print(next_station)
                    if curr_lat:
                        curr_difference = euclidean_distance(curr_lat, curr_lon, t_lat, t_lon)
                        lower_bound_in_seconds = math.floor(curr_difference/15)

                        new_travel_time = current_travel_time + edge_travel_time

                        #Condition
                        if current_travel_time + edge_travel_time + lower_bound_in_seconds > tentative_best_path:
                            continue

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

                evaluated_nodes[next_station] = {
                    'prev_node': current_station,
                    'departure_time': dep_time,
                    'arrival_time': arr_time,
                    'is_transfer': True if transfer else False
                }

                heapq.heappush(pq, (new_travel_time, arr_time, new_transfers, next_station))
        max_transfers -= 1

    global dijkstra_time
    dijkstra_time = round(time.time() - start,3)*1000
    print('Dijkstra Time: ',dijkstra_time)
    return evaluated_nodes, settled, tentative_best_path, pq

def is_trip_service_day_valid(trip_id, trip_service_days, shifted_dates, day_shift):
    departure_day, weekday = shifted_dates[day_shift]
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

        if trip_id in checked_trips or is_trip_service_day_valid(trip_id, trip_service_days, shifted_dates, day_shift):
            checked_trips.add(trip_id)
            return (dep_time, arr_time, 0)
        
        index += 1

def is_transit_edge(edge) -> bool:
    return True if edge['line_num'] == None else False

def find_path_pareto(source_node, end_node, evaluated_nodes):
    path = []
    prev_node = end_node
    while prev_node != source_node:
        edge = evaluated_nodes[prev_node]
        path.append(edge)
        prev_node = edge['prev_node'] 

        if edge['is_transfer']:
            #num_of_transfers -= 1
            edge['line_num'] = None 
        else:
            edge['line_num'] = prev_node.split('_')[-1]

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
    departure_time_sec = convert_str_to_sec(departure_time_str)
    departure_station_id = StopNames.get_id_from_fuzzy_input_name(departure_station_name)
    arrival_station_id = StopNames.get_id_from_fuzzy_input_name(arrival_station_name)
    print(departure_station_id, arrival_station_id)
    if not (departure_station_id and arrival_station_id):
        print('Station name is not valid.')
        return False, None
    
    print("\n",departure_station_id)

    all_found_connections = []

    global tentative_best_path
    last_tentative_best_path = np.inf
    pq = None
    last_ev = {}

    starting_times = []

    start_stations = edges[departure_station_id].keys()
    for start_station in start_stations:
        for out_node, connections in edges[start_station].items():
            if connections[0][0] != 'P':
                starting_times += [edge[0] for edge in connections if departure_time_sec + PROFILE_INTERVAL >= edge[0] >= departure_time_sec]

    starting_times = sorted(set(starting_times), reverse=True)

    for starting_time in starting_times:
        print('\nStart at:',convert_sec_to_hh_mm_ss(starting_time))
        departure_time_sec = starting_time

        evaluated_nodes, _, new_tentative_best_path, pq = time_dependent_pareto_dijkstra(departure_station_id, arrival_station_id, departure_time_sec, edges, trip_service_days, departure_day, last_tentative_best_path, pq, last_ev)

        print('Time of Path Earliest Arrival:',new_tentative_best_path, convert_sec_to_hh_mm_ss(departure_time_sec))
        
        last_ev = evaluated_nodes #tady to chce mergnou a nechat to mensi kdyz je na vyber
        if new_tentative_best_path < last_tentative_best_path and evaluated_nodes.get(arrival_station_id):
            last_tentative_best_path = new_tentative_best_path
            best_eval_nodes = evaluated_nodes.copy()
            print('New Best Path Earliest Arrival', new_tentative_best_path//60)

    evaluated_nodes = best_eval_nodes

    if arrival_station_id not in evaluated_nodes:
        print('No Route Found')
        return False, None

    # Filter out paths with same arrival time but more transfers

    path = find_path_pareto(departure_station_id, arrival_station_id, evaluated_nodes)

    connections = construct_final_path_table(path)
    all_found_connections.append(connections)
    full_results_bool = False

    if TO_PRINT_IN_TERMINAL:
        for connection_counter, connection in enumerate(connections):
            for stop_counter, stop in enumerate(connection):
                if full_results_bool or stop_counter == 0 or stop_counter == len(connection) - 1:
                    print(stop[0], convert_sec_to_hh_mm_ss(stop[1]), stop[2], stop[3])

            print('')
        print('')
        print('Travel time', get_travel_time_in_mins(path),'min')
    else:
        pass
        #print("Route Exists")

    return True, all_found_connections

def convert_str_to_datetime(day: str):
    day = datetime.strptime(day, '%Y%m%d')
    return day

def main():
    trip_service_days = get_trip_service_days()
    print("Trips loaded")
    tmtb = get_timetable()
    edges = get_edges(tmtb)
    StopNames.initialize(get_stops_df(), tmtb)
    global_time = time.time()

    for i in range(NUM_OF_SEARCHES):
        departure_time_str, departure_day, departure_station_name, arrival_station_name,  = get_default_journey_info()
        global spec_node_to_main
        spec_node_to_main = {node: main for node, main in zip(tmtb['node_id'], tmtb['main_station_id'])}
        
        start_time = time.time()
        _, all_found_connections = run_program(departure_station_name, arrival_station_name, departure_time_str, departure_day, edges, trip_service_days)
        

        print('Dijkstra Time',dijkstra_time)
        print('One RunTime:', round((time.time() - start_time)*1000, 2), 'ms')
    print('Global Avg Run Time:', round((time.time() - global_time)*1000/NUM_OF_SEARCHES, 2), 'ms')


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


# Define constants
MIN_TRANSFER_TIME = 180
TIME_WINDOW = 8*60*60
TRANSFER_BOUND = 6
ONLY_FASTEST_TRIP = False
NUMBER_OF_DAYS_IN_ADVANCE = 14
PROFILE_INTERVAL = 120*60 #seconds

NUM_OF_SEARCHES = 1 #for testing, number of searches
TO_PRINT_IN_TERMINAL = False if NUM_OF_SEARCHES > 1 or __name__ != "__main__" else True

RANDOM_STATIONS = False
random.seed(10)

abspath = os.path.abspath(__file__)
dirpath = os.path.dirname(abspath)
CACHE_FOLDER_PATH = os.path.join(dirpath, "cache")

if __name__ == "__main__":
    global using_landmarks
    global using_star
    using_landmarks = False
    using_star = True
    main()
    exit()
 
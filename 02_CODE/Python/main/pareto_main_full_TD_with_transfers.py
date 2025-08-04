import math
import random
import heapq
import bisect
import os
import statistics
import time

from datetime import timedelta, datetime
from collections import defaultdict
from rapidfuzz import process, fuzz
from unidecode import unidecode
from gtfs_pandas import *

class JourneyInputInfo():

    def __init__(self, origin_time, origin_day, origin_station, target_station):
        self.origin_time = origin_time
        self.origin_day = origin_day
        self.origin_station = origin_station
        self.target_station = target_station

    def set_origin_time(self, random):
        self.origin_time = get_random_time_in_string() if random else get_default_journey_info()[0]

    def set_origin_day(self):
        self.origin_day = get_default_journey_info()[1]

    def set_origin_station(self, random, zone = None):
        self.origin_station = get_random_station(zone) if random else get_default_journey_info()[2]

    def set_target_station(self, random, zone = None):
        self.target_station = get_random_station(zone) if random else get_default_journey_info()[3]

#Helper Functions
def get_random_time_in_string():
    return f"{random.randint(0,24):02}:{random.randint(0,60):02}:{random.randint(0,60):02}"

def convert_str_to_sec(timestamp: str) -> int:
    return sum(int(t) * 60 ** i for i, t in enumerate(reversed(timestamp.split(":"))))

def convert_kph_to_ms(speed_kph):
    return speed_kph/3.6

def convert_sec_to_hh_mm_ss(time) -> str:
    time = int(time % (24 * 3600))
    hours, remainder = divmod(time, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def match_stop_name_and_id(key: str, stop_name_id_dict: dict) -> str:
    return stop_name_id_dict.get(key, None)

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

def time_dependent_pareto_dijkstra(journey_info, edges, trip_service_days, is_shortest_path_tree_search = False, gui = True if __name__ != "__main__" else False):
    """ Implements the modified Dijkstra's algorithm to find all Pareto-optimal paths iteratively. """

    pq = [(journey_info.origin_time, journey_info.origin_time, 0, journey_info.origin_station)]  # (reduced cost, arrival_time, num_transfers, station)
    labels = {journey_info.origin_station: [(journey_info.origin_time, 0)]}  # Station → List of (arrival_time, num_transfers)
    evaluated_nodes = {journey_info.origin_station: {0: {'prev_node': None, 'departure_time': None, 'arrival_time': journey_info.origin_time}}}

    max_transfers = TRANSFER_BOUND  # Start with the initial transfer limit

    # Create a mapping: {shift: (YYYYMMDD, weekday)}
    shifted_dates = get_shifted_days_dicts(journey_info.origin_day)

    global checked_trips
    checked_trips = set()

    t_lat, t_lon = StopNames.get_coordinates_lat_lon(journey_info.target_station)    
    global settled
    settled = set()

    if not gui and USING_LANDMARKS:
        travel_time_from_landmarks_target = {landmark:dist for landmark, dist in preprocessed_paths_norm[journey_info.target_station].items()}
        travel_time_to_landmarks_target = {landmark:dist for landmark, dist in preprocessed_paths_rev[journey_info.target_station].items()}
    
    start = time.time()
    global already_landmarked
    already_landmarked = {}
    ONLY_FASTEST_TRIP = False if gui else ONLY_FASTEST_TRIP

    while max_transfers >= 0:  # Run until we reach -1 transfers
        while pq:
            _, current_time, current_transfers, current_station = heapq.heappop(pq)

            if ONLY_FASTEST_TRIP: 
                settled.add(current_station)
            else: # pareto-labels
                settled.add((current_station, current_transfers))

            if not is_shortest_path_tree_search and current_station == journey_info.target_station:
                max_transfers = -1 if ONLY_FASTEST_TRIP else current_transfers
                break

            if current_transfers > max_transfers:
                continue  #

            if current_station not in edges:
                continue

            is_start_station = current_station == journey_info.origin_station

            for next_station, connections in edges[current_station].items():
                
                if next_station in settled or (next_station, current_transfers) in settled:
                    continue
                
                if is_shortest_path_tree_search: # preprocessing static graph
                    dep_time, arr_time, transfer = binary_search_next_edge_static(connections, current_time, is_start_station)
                else:
                    dep_time, arr_time, transfer = binary_search_next_edge(connections, current_time, is_start_station, trip_service_days, shifted_dates)

                if arr_time is None or arr_time > journey_info.origin_time + TIME_WINDOW:
                    continue

                edge_travel_time = (arr_time - current_time)

                new_transfers = current_transfers + transfer

                if new_transfers > max_transfers:
                    continue
                
                if not ONLY_FASTEST_TRIP:
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

                        labels[next_station] = [(time, transfer) for (time, transfer) in labels[next_station] if not (time >= arr_time and transfer >= new_transfers)]

                    labels[next_station].append((arr_time, new_transfers))
                
                else:
                    if next_station not in labels:
                        labels[next_station] = arr_time
                    elif arr_time < labels[next_station]:
                        labels[next_station] = arr_time
                    else:
                        continue

                if next_station not in evaluated_nodes:
                    evaluated_nodes[next_station] = {}
                
                if gui or USING_STAR: # GUI uses A* by default
                    try:
                        curr_lat, curr_lon = StopNames.get_coordinates_lat_lon(next_station)
                    except:
                        print(next_station)
                    if curr_lat:
                        eucl_m_dist = euclidean_distance(curr_lat, curr_lon, t_lat, t_lon)

                        max_speed_kph = 54
                        max_speed_ms = convert_kph_to_ms(max_speed_kph)

                        lower_bound_in_seconds = math.floor(eucl_m_dist/max_speed_ms)
                        new_augmented_time = current_time + edge_travel_time + lower_bound_in_seconds

                    else:
                        new_augmented_time = current_time + edge_travel_time
                
                elif USING_LANDMARKS:
                    main_id = StopNames.get_main_id_from_node_id(current_station)
                    if main_id in already_landmarked:
                        new_augmented_time = current_time + edge_travel_time + already_landmarked[main_id]

                    elif main_id in preprocessed_paths_rev or main_id in preprocessed_paths_norm:
                        #normal
                        this_node = preprocessed_paths_norm[main_id]
                        max_lower_bound_norm = max(int(travel_time_from_landmarks_target.get(landmark, landmark_to_current)) - int(landmark_to_current)
                        for landmark, landmark_to_current in this_node.items()) if this_node.items() else 0

                        #reverse
                        this_node = preprocessed_paths_rev[main_id]
                        max_lower_bound_rev = max(int(current_to_landmark) - int(travel_time_to_landmarks_target.get(landmark, current_to_landmark))
                        for landmark, current_to_landmark in this_node.items()) if this_node.items() else 0
                        
                        max_lower_bound = max(max_lower_bound_norm, max_lower_bound_rev)

                        if max_lower_bound > 0:
                            new_augmented_time = current_time + edge_travel_time + max_lower_bound*60
                            already_landmarked[main_id] = max_lower_bound*60
                        else:
                            new_augmented_time = current_time + edge_travel_time
                            already_landmarked[main_id] = 0

                    else:
                        new_augmented_time = current_time + edge_travel_time
                        already_landmarked[main_id] = 0

                else:
                    new_augmented_time = current_time + edge_travel_time

                evaluated_nodes[next_station][new_transfers] = {
                    'prev_node': current_station,
                    'departure_time': dep_time,
                    'arrival_time': arr_time,
                    'is_transfer': True if transfer else False
                }

                heapq.heappush(pq, (new_augmented_time, arr_time, new_transfers, next_station))
        max_transfers -= 1

    global dijkstra_time
    dijkstra_time = round(time.time() - start,3)*1000
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

def binary_search_next_edge_static(edges, current_time, zero_transfer_cost) -> tuple[float, float, int]:
    """Return next edge departure time, arrival time and transfer 0 or 1 
    Finds the next train connection using binary search 
    """
    if edges[0][0] == 'P' or zero_transfer_cost:
        return (current_time, current_time, 0)
    
    if edges[0][0] == 'T':
        return (current_time, current_time + MIN_TRANSFER_TIME, 1)
    
    edge = min(edges, key=lambda item: item[1] - item[0])
    min_edge_travel_time = edge[1] - edge[0]
    return (current_time, current_time + min_edge_travel_time, 0)

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
    return math.floor((path[-1]['arrival_time'] - path[1]['departure_time'])/60)

def get_random_station(zone = None):
    return (StopNames.get_a_random_stop_name(zone))

def convert_all_journey_info(journey_info: JourneyInputInfo):
    journey_info.origin_day = convert_str_to_datetime(journey_info.origin_day)
    journey_info.origin_time = convert_str_to_sec(journey_info.origin_time)
    journey_info.origin_station = StopNames.get_id_from_fuzzy_input_name(journey_info.origin_station)
    journey_info.target_station = StopNames.get_id_from_fuzzy_input_name(journey_info.target_station)

def run_program(journey_info: JourneyInputInfo, edges, trip_service_days):
    convert_all_journey_info(journey_info)
    
    print(journey_info.origin_station, journey_info.target_station)

    if not (journey_info.origin_station and journey_info.target_station):
        print('Station name is not valid.')
        return False, None
    
    print("\n",journey_info.origin_station)

    all_found_connections = []

    evaluated_nodes, _ = time_dependent_pareto_dijkstra(journey_info, edges, trip_service_days)

    print('eval',len(evaluated_nodes))

    if journey_info.target_station not in evaluated_nodes:
        print('No Route Found')
        return False, None

    # Filter out paths with same arrival time but more transfers
    seen_arrival_times = []
    num_of_transfers_per_path = []

    for transfers, edge in sorted(evaluated_nodes[journey_info.target_station].items(), reverse = True):
        arrival_time = edge['arrival_time']
        if seen_arrival_times and arrival_time <= seen_arrival_times[-1]:
            num_of_transfers_per_path[-1] = transfers
        else:
            num_of_transfers_per_path.append(transfers)
            seen_arrival_times.append(arrival_time)

    for num_of_transfers in num_of_transfers_per_path: #paths with that many transfers
        path = find_path_pareto(journey_info.origin_station, journey_info.target_station, evaluated_nodes, num_of_transfers)

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

    return True, all_found_connections

def get_default_journey_info():
    departure_station_name = "Malvazinky"
    arrival_station_name = "K Juliane"
    departure_time = '6:00:00'
    departure_day = '20250610'

    return departure_time, departure_day, departure_station_name, arrival_station_name

def convert_str_to_datetime(day: str):
    day = datetime.strptime(day, '%Y%m%d')
    return day

def main():
    edges = get_edges()
    trip_service_days = get_trip_service_days()
    print("Trips loaded.")
    timetable = get_timetable()
    print("Timetable loaded.")
    StopNames.initialize(get_stops_df(), timetable)
    
    journey_info = JourneyInputInfo(None, None, None, None)
    journey_info.set_origin_day()
    journey_info.set_origin_time(random = RANDOM_INPUTS)  
    journey_info.set_origin_station(random = RANDOM_INPUTS)
    journey_info.set_target_station(random = RANDOM_INPUTS)

    if USING_LANDMARKS:
        global preprocessed_paths_norm
        global preprocessed_paths_rev
        preprocessed_paths_norm = load_cached_data('ALLfinal_preprocess_8_farthest_normal_mathfloor')
        preprocessed_paths_rev = load_cached_data('ALLfinal_preprocess_8_farthest_rev_mathfloor')

    start_time = time.time()
    _ = run_program(journey_info, edges, trip_service_days)
    
    print('Just Dijkstra Time:', dijkstra_time, 'ms')
    print('One RunTime:', round((time.time() - start_time)*1000, 2), 'ms')

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

def preprocess_landmarks(landmarks, reverse):
    timetable = get_timetable()
    stop_ids = timetable['stop_id'].unique().tolist()
    main_station_ids = timetable['main_station_id'].unique().tolist()
    edges = build_edges(timetable, reverse = reverse)
    
    dep_time = 10*60*60

    trips = get_trip_service_days()
    #landmarks = ['U5079', 'U9830', 'U9989', 'U6210']   #zatec, zaskmuky, komarov, bakov n. jizerou
    end = 'U597'
    preprocessed_paths = defaultdict(dict)

    global USING_STAR
    global USING_LANDMARKS
    USING_STAR = False
    USING_LANDMARKS = False
    departure_day_dt = convert_str_to_datetime('20250610')

    print(len(stop_ids))

    for landmark in landmarks:
        print('L', landmark)
        journey_info = JourneyInputInfo(dep_time, departure_day_dt, landmark, end)
        evaluated_nodes, _ = time_dependent_pareto_dijkstra(journey_info, edges=edges,trip_service_days=trips, is_shortest_path_tree_search = True)

        for node, val in evaluated_nodes.items():
            max_tr = max(val.keys())
            if node in main_station_ids:
                if node != landmark:
                    path = find_path_pareto(landmark, node, evaluated_nodes, num_of_transfers = max_tr)
                    if path and len(path) > 1:
                        travelTime = get_travel_time_in_mins(path)
                        if node in preprocessed_paths and landmark in preprocessed_paths[node]:
                            print(preprocessed_paths.get[node][landmark])
                        preprocessed_paths[node][landmark] = travelTime
                else:
                    preprocessed_paths[node][landmark] = 0
        
    print(preprocessed_paths)

    return preprocessed_paths

def select_landmarks():
    for i in range(6):
        if i < 3:
            landmarks = ['U876']
            for lndm in range(2**(3+i)):
                new_landmark = get_farthest(landmarks)
                landmarks.append(new_landmark) 
            landmarks = landmarks[1:]
            filename_processed_landmarks= f'preprocess_{2**(3+i)}_farthest_rev'
        else:
            random.seed(1)
            landmarks = [*(StopNames.get_id_from_fuzzy_input_name(StopNames.get_a_random_stop_name()) for i in range(2**(i)))]
            filename_processed_landmarks= f'preprocess_{2**(i)}_random_rev_'
        
        preprocessed_paths = preprocess_landmarks(landmarks, reverse = True)
        save_cached_data(preprocessed_paths, filename_processed_landmarks)
        print('Processed')

def get_farthest(landmarks):
        farthest = 0
        farthest_stop = None
        landmarks_coords = []
        for landmark in landmarks:
            lm_lat, lm_lon = StopNames.get_coordinates_lat_lon(landmark)
            landmarks_coords.append((lm_lat, lm_lon))

        for station, (clat, clon) in StopNames._coordinates.items():
            total_dist = 0
            too_close = False
            for lm_lat, lm_lon in landmarks_coords:
                update_dist = euclidean_distance(lm_lat, lm_lon, clat, clon)
                total_dist += update_dist
                if update_dist < 10000: #25km radius
                    too_close = True
                    break
                
            if total_dist > farthest and not too_close and int(station[1:]) < 31000:
                farthest = total_dist
                farthest_stop = station
        
        print(StopNames.get_general_name_from_id(farthest_stop), farthest/1000/len(landmarks))
        return farthest_stop

def testing(n, from_P):
    random.seed(123)

    departures = [*(StopNames.get_id_from_fuzzy_input_name(StopNames.get_a_random_stop_name('P' if from_P else None)) for i in range(n//2))] + [*(StopNames.get_id_from_fuzzy_input_name(StopNames.get_a_random_stop_name()) for i in range(n//2))]

    arrivals = [*(StopNames.get_id_from_fuzzy_input_name(StopNames.get_a_random_stop_name()) for i in range(n//2))] + [*(StopNames.get_id_from_fuzzy_input_name(StopNames.get_a_random_stop_name('P' if from_P else None)) for i in range(n//2))]

    edges = get_edges()
    trips = get_trip_service_days()
    dep_time = convert_str_to_sec(get_random_time_in_string())
    dep_time = convert_str_to_sec('06:00:00')
    departure_day_dt = convert_str_to_datetime("20250610")

    global USING_STAR
    global USING_LANDMARKS
    USING_STAR = False
    USING_LANDMARKS = True

    loops = 3

    with open('Results_Execution_Time_Dijkstra.txt', 'a') as file:
        file.write('\n==================================================\n')
    counter = 0

    for j in range(loops):
        times_list = []
        list_of_EA = []

        global preprocessed_paths
        global preprocessed_paths_norm
        global preprocessed_paths_rev

        if j == 0:
            mode = 'Landmarks'
            filename_processed_landmarks_norm = f'ALLfinal_preprocess_16_farthest_normal_mathfloor'
            filename_processed_landmarks_rev = f'ALLfinal_preprocess_16_farthest_rev_mathfloor'
            preprocessed_paths_norm = load_cached_data(filename_processed_landmarks_norm)
            preprocessed_paths_rev = load_cached_data(filename_processed_landmarks_rev)

        elif j == 1:
            mode = 'Basic'
            USING_LANDMARKS = False
            USING_STAR = False

        elif j == 2:
            mode = 'A Star'
            USING_LANDMARKS = False
            USING_STAR = True

        print('Start', mode, '' if not USING_LANDMARKS else filename_processed_landmarks_norm, departure_day_dt)
        counter_not_found_paths = 0

        for i in range(n):
            individual_time = time.time()
            start = departures[i%len(departures)]
            end = arrivals[i%len(arrivals)]
            if start == end:
                continue

            journey_info = JourneyInputInfo(dep_time, departure_day_dt, start, end)
            evaluated_nodes, _ = time_dependent_pareto_dijkstra(journey_info, edges, trips)
            if end not in evaluated_nodes:
                counter_not_found_paths += 1
                times_list.append(time.time() - individual_time)
                continue
            max_transfers = max(evaluated_nodes[end].keys())
            path = find_path_pareto(start, end, evaluated_nodes, num_of_transfers=max_transfers)
            list_of_EA.append(path[-1]['arrival_time'])
            times_list.append(time.time() - individual_time)

        with open('Results_Execution_Time_Dijkstra.txt', 'a') as file:
            file.write('\nStart ' + mode[j%len(mode)] + ('' if not USING_LANDMARKS else filename_processed_landmarks_norm) + ' 20250610 No Transfer Labels\n')
            file.write('Median ' + str(round(statistics.median(times_list), 4))+'\n')
            file.write('Mean ' + str(round(statistics.mean(times_list), 4)) +'\n')
            file.write('Not found paths ' + str(counter_not_found_paths) +'\n\n')

        print('Median', statistics.median(times_list))
        print('Mean', statistics.mean(times_list))
        print('Not found paths', counter_not_found_paths,'\n')
        print(list_of_EA)
    print('Counter', counter)

# Define constants
MIN_TRANSFER_TIME = 180
TIME_WINDOW = 16*60*60
TRANSFER_BOUND = 10
ONLY_FASTEST_TRIP = False
NUMBER_OF_DAYS_IN_ADVANCE = 14

TO_PRINT_IN_TERMINAL = True
RANDOM_INPUTS = False
USING_LANDMARKS = False
USING_STAR = False

abspath = os.path.abspath(__file__)
dirpath = os.path.dirname(abspath)
CACHE_FOLDER_PATH = os.path.join(dirpath, "cache")

if __name__ == "__main__":
    # main()
    testing(25, False)

   


   

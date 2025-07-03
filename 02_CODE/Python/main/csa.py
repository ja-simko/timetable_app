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


from collections import namedtuple
import datetime as dt
from datetime import timedelta, datetime
from collections import defaultdict
from rapidfuzz import process, fuzz
from unidecode import unidecode
from sqlalchemy import create_engine, text
from gtfs_pandas import *
from pareto_main import convert_sec_to_hh_mm_ss, is_trip_service_day_valid, get_shifted_days_dicts

def build_edges_csa(timetable = pd.DataFrame()):
    """
    Optimized version of build_edges with better performance and cleaner logic.
    """
    if timetable.empty:
        timetable = get_timetable()

    # Pre-allocate with better estimate
    edges = []
    transfer_edges_added = set()
    
    # Convert to numpy arrays for faster iteration
    trip_ids = timetable['trip_id'].values
    node_ids = timetable['stop_id'].values
    main_stations = timetable['main_station_id'].values
    departure_times = timetable['departure_time'].values
    arrival_times = timetable['arrival_time'].values
    
    # Constants
    DAY_SECONDS = 24 * 3600
    EARLY_MORNING = 6 * 3600
    
    # Process consecutive pairs efficiently
    prev_trip_id = None

    Edge = namedtuple('Edge',"dep_time, arr_time, dep_stop, arr_stop, main_dep_stop, main_arr_stop, trip_id, shift")
    
    for i in range(len(timetable)):
        current_trip_id = trip_ids[i]
        
        # Skip first row of each trip
        if current_trip_id != prev_trip_id:
            prev_trip_id = current_trip_id
            continue
        
        # Get current and previous row data
        prev_idx = i - 1
        
        out_node, in_node = node_ids[prev_idx], node_ids[i]

        main_out, main_in = main_stations[prev_idx], main_stations[i]

        dep_time, arr_time = departure_times[prev_idx], arrival_times[i]
        
        edges.append(Edge(dep_time, arr_time, out_node, in_node, main_out, main_in, current_trip_id, 0))

        # Add time-shifted edges (vectorized logic)
        if dep_time >= DAY_SECONDS:
            edges.append(Edge(dep_time - DAY_SECONDS, arr_time - DAY_SECONDS, out_node, in_node, main_out, main_in, current_trip_id, -1))

        elif dep_time < EARLY_MORNING:
            edges.append(Edge(dep_time + DAY_SECONDS, arr_time + DAY_SECONDS, out_node, in_node, main_out, main_in, current_trip_id, 1))
    
    edges.sort(key = lambda edge: edge[0])
    return edges

def build_footpaths(edges) -> list:
    # Group stops by station

    station_stops = defaultdict(set)
    min_transfer_time: int = 180
    min_transfer_loop_time: int = 60
    
    for edge in edges:
        station_stops[edge.main_dep_stop].add(edge.dep_stop)
        station_stops[edge.main_arr_stop].add(edge.arr_stop)
    
    # Build footpaths within each station
    footpaths = defaultdict(list)
    transfer_pairs_added = set()
    
    for station_id, stops in station_stops.items():
        if len(stops) <= 1:
            continue  # Skip stations with only one stop
            
        # Create bidirectional footpaths between all stops in the station
        stops_list = list(stops)
        for i, stop1 in enumerate(stops_list):
            for stop2 in stops_list[i:]:
                pair = tuple(sorted([stop1, stop2]))

                if pair not in transfer_pairs_added:
                    transfer_pairs_added.add(pair)

                    if stop1 == stop2:
                        footpaths[stop1].append((stop2, min_transfer_loop_time))

                    else:
                        footpaths[stop1].append((stop2, min_transfer_time))
                        footpaths[stop2].append((stop1, min_transfer_time))
    return footpaths

def scan_connections(connections, footpaths, starting_time, start_station, target_station, departure_day, trip_service_days):
    index = bisect.bisect_left(connections, (starting_time,))
    trips = {}
    evaluated_stops = defaultdict(lambda: (2 * 24 * 60 * 60, 10))  # (earliest_arrival_time, min_transfers, last trip)
    journey_pointers = {}

    evaluated_stops[start_station] = (starting_time, -1)

    Trip_info = namedtuple('Trip_info','conn, transfers')

    for to_stop, _ in footpaths.get(start_station, []):
        evaluated_stops[to_stop] = (starting_time, -1)

    counter = 0

    shifted_days = get_shifted_days_dicts(departure_day)

    checked_trips = set()

    for conn in connections[index:]:
        counter += 1

        if not (conn.trip_id in checked_trips or is_trip_service_day_valid(conn.trip_id, trip_service_days, shifted_days, conn.shift)):
            checked_trips.add(conn.trip_id)
            continue

        arr_time, arr_transfers = evaluated_stops[target_station]

        if conn.dep_time >= arr_time:
            return arr_time, arr_transfers, counter, journey_pointers

        curr_dep_time, curr_dep_transfers = evaluated_stops[conn.dep_stop]
        trip_info = trips.get(conn.trip_id)

        if trip_info or curr_dep_time <= conn.dep_time:
            
            if not trip_info:
                curr_dep_transfers += 1
                trips[conn.trip_id] = Trip_info(conn, curr_dep_transfers)
                trip_info = trips[conn.trip_id]
            
            arr_footpaths = footpaths.get(conn.arr_stop)

            if arr_footpaths:
                for to_stop, transfer_time in arr_footpaths:
                    if conn.arr_stop == target_station or to_stop == target_station:
                        transfer_time = 0

                    new_time = conn.arr_time + transfer_time
                    prev_time, prev_transfers = evaluated_stops[to_stop]

                    if (new_time < prev_time) or (new_time == prev_time and trip_info.transfers < prev_transfers): 
                        evaluated_stops[to_stop] = (new_time, trip_info.transfers)
                        journey_pointers[to_stop] = (trip_info.conn, conn)

    return None, None, counter, None

def extract_full_journey(journey_pointers, target, connections):
    if target not in journey_pointers:
        return []
    
    journey_segments = []
    current_stop = target
    
    while journey_pointers.get(current_stop):
        trip_start_conn, exit_conn = journey_pointers[current_stop]
        if trip_start_conn and exit_conn:
            trip_id = trip_start_conn.trip_id
            trip_segment = []
            
            found_start = False
            for conn in connections:
                if conn.trip_id == trip_id:
                    if conn.dep_stop == trip_start_conn.dep_stop:
                        found_start = True

                    if found_start:
                        trip_segment.append((conn.dep_stop, conn.dep_time, conn.trip_id))
                        
                        if conn.arr_stop == exit_conn.arr_stop:
                            break
            
            journey_segments += reversed(trip_segment)
        
        if trip_start_conn:
            current_stop = trip_start_conn.dep_stop
        else:
            break
    
    return list(reversed(journey_segments))

def extract_journey(journey_pointers, target):
    if target not in journey_pointers:
        return []
        
    journey = []
    current_stop = target
    
    while journey_pointers.get(current_stop):
        first_conn, exit_conn = journey_pointers[current_stop]

        # Add the connection
        if exit_conn:
            journey.append((exit_conn.arr_stop, exit_conn.arr_time))
            journey.append((first_conn.dep_stop, first_conn.dep_time, first_conn.trip_id))
        
        # Move to the departure stop of the trip that got us here
        if first_conn:
            current_stop = first_conn.dep_stop
        else:
            break
        
    return list(reversed(journey))

def construct_final_path(path):
    c = 0
    if path:
        for stop, time, *args in path:
            if c%2 == 0:
                print('')
            c += 1
            print(StopNames.get_general_name_from_id(stop), convert_sec_to_hh_mm_ss(time), args)
    else:
        print('No path')

def run_multiple_scans(edges, footpaths, start_time, start_station, end_station, departure_day, trip_service_days, n = 10):
    start_clock = time.perf_counter()
    is_full = True

    for i in range(n):
        ea_time, tr, seen_conns, journey_pointers = scan_connections(
            connections = edges,
            footpaths = footpaths, 
            starting_time = start_time + (i+2)*60, 
            start_station = start_station, 
            target_station = end_station,
            departure_day = departure_day,
            trip_service_days = trip_service_days
        )

        if ea_time:
            journey = extract_full_journey(journey_pointers, end_station, edges)
            construct_final_path(journey)
    elapsed = (time.perf_counter() - start_clock)/n

    print(f"Outside function timing: {elapsed*1000:.4f} ms")
    print(f"Seen connections: {seen_conns}")
    print(f"Earliest arrival time: {ea_time}")


if __name__ == "__main__":
    print('Building edges')
    timetable = get_timetable()
    StopNames.initialize(get_stops_df(), timetable)

    edges = build_edges_csa(timetable)
    footpaths = build_footpaths(edges)
    trip_service_days = get_trip_service_days()
    print('start')

    # More precise timing
    departure_day = '20250618'
    start_time = '10:38:00'
    start_time = convert_str_to_sec(start_time)

    

    # end_station = 'klikovan'
    start_station = 'dvorce'
    end_station = 'lihovar'
    start_station = 'albertov'
    end_station = 'palmova'
    # # end_station = 'prazskeho pvostani'


    start_station = StopNames.get_stop_id_from_main_id(StopNames.get_id_from_fuzzy_input_name(start_station))
    end_station = StopNames.get_stop_id_from_main_id(StopNames.get_id_from_fuzzy_input_name(end_station))


    run_multiple_scans(edges, footpaths, start_time, start_station, end_station, departure_day, trip_service_days, n = 10)
  
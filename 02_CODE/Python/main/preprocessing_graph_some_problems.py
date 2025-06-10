from collections import defaultdict
import re
from datetime import timedelta
import joblib
import os

def convert_nested_defaultdict(d):
    return {k: convert_nested_defaultdict(v) for k, v in d.items()} if isinstance(d, defaultdict) else d

def generate_route_ids(df):
    """ Assigns a unique route ID to each unique stop sequence. """
    route_map = {}  # {stop_sequence_tuple: route_id}
    route_counter = 1  # Unique route ID counter

    # Group trips by their ordered sequence of stops
    trip_sequences = defaultdict(list)

    for row in df.itertuples(index=False):
        trip_sequences[row.trip_id].append(row.stop_id)

    for trip_id, stop_sequence in trip_sequences.items():
        sequence_tuple = tuple(stop_sequence)

        if sequence_tuple not in route_map:
            route_map[sequence_tuple] = f"R{route_counter}_L{trip_id.split('_')[0]}"
            route_counter += 1

    return route_map, trip_sequences  # {stop_sequence_tuple: route_id}

def build_edges(timetable):
    print('start building edges')
    """ Builds the graph edges based on computed route IDs. """
    prev_row = None

    route_map, trip_sequences = generate_route_ids(timetable)

    edges = defaultdict(lambda: defaultdict(list))

    station_re = re.compile(r"^(U\d+)[ZS]")

    last_trip_id = None#timetable.loc[0, 'trip_id']

    for row in timetable.itertuples(index=False):
        if row.trip_id != last_trip_id:
            line_num = row.route_short_name
            last_trip_id = row.trip_id
            route_id = route_map[tuple(trip_sequences[row.trip_id])]
            prev_row = row
            continue
        
        out_node_stop_id, in_node_stop_id = prev_row.stop_id, row.stop_id

        try:
            out_main_station = station_re.match(out_node_stop_id).group(1)
            in_main_station = station_re.match(in_node_stop_id).group(1)
        except:
            continue

        out_node = f"{out_node_stop_id}_{route_id}"
        in_node = f"{in_node_stop_id}_{route_id}"

        dep_time = prev_row.departure_time #+ time_adjustment
        arr_time = row.arrival_time #+ time_adjustment

        if dep_time >= 24*3600:
            edges[out_node][in_node].append((dep_time - 24*3600, arr_time - 24*3600, row.trip_id, -1, line_num))
        elif dep_time < 6*3600:
            edges[out_node][in_node].append((dep_time + 24*3600, arr_time + 24*3600, row.trip_id, 1, line_num))
        edges[out_node][in_node].append((dep_time, arr_time, row.trip_id, 0, line_num))

        #edges[out_node][in_node].append((dep_time.total_seconds(), arr_time.total_seconds(), row.trip_id))

        # Edge from main station node to route-specific node (transfer time)
        if out_node not in edges[out_main_station]:
            edges[out_main_station][out_node].append(('T', MIN_TRANSFER_TIME, None, 0, None))

        # Edge from route-specific node to main station node (zero cost)
        if in_main_station not in edges[in_node]:
            edges[in_node][in_main_station].append(('P', 0, None, 0, None))

        prev_row = row
    
    for out_node in edges:
        for in_node in edges[out_node]:
            edges[out_node][in_node].sort()

    edges = convert_nested_defaultdict(edges)

    file_path = os.path.join(CACHE_EDGES_DIR, f"edges_all_pre_process")

    joblib.dump(edges, file_path, compress=0)

    print('end building edges')
    return edges

MIN_TRANSFER_TIME = timedelta(seconds = 120)
CACHE_EDGES_DIR = r"C:\Users\Jachym\OneDrive - České vysoké učení technické v Praze\Bakalářská_práce\02_CODE\cache\edges"
CACHE_TIMETABLE = r"C:\Users\Jachym\OneDrive - České vysoké učení technické v Praze\Bakalářská_práce\02_CODE\cache\timetable_only.pkl"

build_edges(joblib.load(CACHE_TIMETABLE))
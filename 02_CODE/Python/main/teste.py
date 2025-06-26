from pareto_main import *
from gtfs_pandas import get_edges, get_trip_service_days
import cProfile
from pstats import *
import time
import json
import statistics
import line_profiler
import io
from csa import *


def load_real_data():
    edges = build_edges_csa()
    footpaths = build_footpaths(edges)
    trip_service_days = get_trip_service_days()
    return edges, trip_service_days,footpaths

def test_modified_dijkstra_pareto():
    # Load real data
    edges, trip_service_days, footpaths= load_real_data()

    stop_name_to_id = get_stop_name_to_id()

    #start_station = get_id_from_best_name_match(stop_name_to_id, "andelska hora, rozc")
    #target_station = get_id_from_best_name_match(stop_name_to_id, "nadrazi podbaba")
    start_station = 'dekanka'
    target_station = 'k juliane'
    start_time = 30000
    departure_day = '20250610'
    departure_day_dt = convert_str_to_datetime(departure_day)

    # Run 10 times
    execution_times = []
    profiler = cProfile.Profile()
    profiler.enable()
    global modifier
    modifier = 5 

    for i in range(10):
        start = time.time()
        result = scan_connections(edges, footpaths, start_time, start_station, target_station)
        end = time.time()
        execution_times.append(end - start)
        print(f"Run {i+1}: {execution_times[-1]:.3f} seconds")

    profiler.disable()

    s = io.StringIO()
    ps = Stats(profiler, stream=s).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(20)  # Top 20 functions
    print("Detailed Profile:")
    print(s.getvalue())

    avg_time = statistics.mean(execution_times)
    std_dev = statistics.stdev(execution_times)
    print(f"\nAverage execution time: {avg_time:.3f} seconds")
    print(f"Standard deviation: {std_dev:.3f} seconds")
    return result

if __name__ == "__main__":
    test_modified_dijkstra_pareto()

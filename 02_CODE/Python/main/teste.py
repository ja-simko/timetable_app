from pareto_main import *
from gtfs_pandas import get_edges, get_trip_service_days
import cProfile
import pstats
import time
import json
import statistics

def load_real_data():
    edges = get_edges()
    trip_service_days = get_trip_service_days()
    return edges, trip_service_days

def test_modified_dijkstra_pareto():
    # Load real data
    edges, trip_service_days = load_real_data()
    
    stop_name_to_id = get_stop_name_to_id()
    stop_ascii_dict = {unidecode(k).lower(): v for k, v in stop_name_to_id.items()}

    start_station = get_id_from_best_name_match(stop_ascii_dict, "andelska hora, rozc")
    target_station = get_id_from_best_name_match(stop_ascii_dict, "nadrazi podbaba")
    start_time = 36000  # 10:00:00
    departure_day = '20250610'
    departure_day_dt = convert_str_to_datetime(departure_day)

    # Run 10 times
    execution_times = []
    profiler = cProfile.Profile()
    profiler.enable()

    for i in range(50):
        start = time.time()
        result = modified_dijkstra_pareto(start_station, target_station, start_time, edges, 
                                        trip_service_days, departure_day_dt)
        end = time.time()
        execution_times.append(end - start)
        print(f"Run {i+1}: {execution_times[-1]:.3f} seconds")

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()
    
    avg_time = statistics.mean(execution_times)
    std_dev = statistics.stdev(execution_times)
    print(f"\nAverage execution time: {avg_time:.3f} seconds")
    print(f"Standard deviation: {std_dev:.3f} seconds")
    return result

if __name__ == "__main__":
    test_modified_dijkstra_pareto()

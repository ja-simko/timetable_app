import tkinter as tk
from tkinter import ttk, messagebox
from tktimepicker import AnalogPicker, AnalogThemes, SpinTimePickerModern, SpinTimePickerOld, SpinDatePickerModern
from datetime import datetime, timedelta  # Ensure datetime is imported
import joblib
import time
# from pareto_main import *
from pareto_main_full_TD_with_transfers import *

import threading
import re
import random
from unidecode import unidecode
import os

import locale
from functools import partial

# Set Czech locale (try 'cs_CZ.UTF-8', or fallback options below)
try:
    locale.setlocale(locale.LC_COLLATE, 'cs_CZ.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_COLLATE, 'cs_CZ')
    except locale.Error:
        print("Czech locale not available on your system.")
        # Optional fallback or custom sort

# Define the cache as global or part of a class if you refactor
edges = None
trip_service_days = None
display_all_stops_var = None

CACHE_EDGES_DIR = r"C:\Users\Jachym\OneDrive - České vysoké učení technické v Praze\Bakalářská_práce\02_CODE\cache\edges"
CACHE_PATH = r"C:\Users\Jachym\OneDrive - České vysoké učení technické v Praze\Bakalářská_práce\02_CODE\cache"

def load_cache_async():
    global edges
    global trip_service_days
    try:
        edges = get_edges()
        trip_service_days = get_trip_service_days()
        submit_button.config(bg="#BE2562")  # Change to green when cache is loaded

    except Exception as e:
        print(f"Failed to load cache: {e}")
        submit_button.config(bg="#000000")  # Change to red if cache loading fails

def process_route(iteration, departure_station_name, arrival_station_name, departure_time_str, departure_day):
    #output_area = output_areas[iteration]
    global earliest_arrival
    global len_of_last_conn

    try:
        route_exists, all_paths = run_program(
            departure_station_name,
            arrival_station_name,
            departure_time_str,
            departure_day,
            edges,
            trip_service_days
        )

        if not route_exists:
            output_area = create_output_areas(iteration, None)[0]
            output_area.delete('1.0', tk.END)
            output_area.insert(tk.END, "Žádná trasa nenalezena.")
            return

        fastest_path = all_paths[0]
        connection = fastest_path[0]
        stop = connection[0]
        next_departure_time = convert_sec_to_hh_mm_ss(int(stop[1]) + 30*60)

        '''
        new_arrival_time = fastest_path[-1][-1][1]
        len_of_new_conn = len(fastest_path)

        if iteration != 0 and new_arrival_time == earliest_arrival and len_of_new_conn == len_of_last_conn:
            iteration -= 1
            output_area = output_areas[iteration]
            output_area.delete('1.0', tk.END)
        else:
            earliest_arrival, len_of_last_conn = new_arrival_time, len_of_new_conn
        '''
        full_results_bool = bool(display_all_stops_var.get())

        insert_results(all_paths, full_results_bool, iteration)

        # Schedule next iteration if not done
        if iteration < NUM_OF_SOLUTIONS - 1:
            process_route(
                iteration + 1,
                departure_station_name,
                arrival_station_name,
                next_departure_time,
                departure_day
            )
    except Exception as e:
        messagebox.showerror("Chyba při hledání trasy", str(e))

def on_submit():
    start_time = time.time()
    departure_station_name = departure_combo.get() or get_default_journey_info()[2]
    arrival_station_name = arrival_combo.get() or get_default_journey_info()[3]

    time_entry = time_picker.time()
    departure_time_str = str(time_entry[0]) + ":" + str(time_entry[1])

    selected_index = date_combobox.current()
    departure_day = str(actual_dates[selected_index].strftime("%Y%m%d"))

    try:
        print(departure_time_str)
        if re.match(r"^\d{1,2}:\d{2}$", departure_time_str):
            departure_time_str += ":00"
        elif not re.match(r"^\d{1,2}:\d{2}:\d{2}$", departure_time_str):
            raise ValueError("Invalid time format")
    except ValueError:
        messagebox.showerror("Chyba", "Zadej čas ve formátu HH:MM")
        return

    # for output_area in output_areas:
    #     output_area.delete('1.0', tk.END)
    process_route(0, departure_station_name, arrival_station_name, departure_time_str, departure_day)
    print('One Run Time:', round((time.time() - start_time)*1000, 2), 'ms')


# Modify the insert_results function
def insert_results(all_paths, full_results_bool, iteration):

    output_areas = create_output_areas(iteration, all_paths)
    print(len(output_areas), iteration)
    #output_area.delete('1.0', tk.END)
    header = f"{'Zastávka':<25}{'Čas odjezdu':<15}{'Linka':<10}{'Stanoviště':<10}\n"

    for i, complete_path in enumerate(all_paths):
        output_area = output_areas[-len(all_paths) + i]
        output_area.insert(tk.END, header)
        output_area.insert(tk.END, "-" *59 + "\n")

        for connection_counter, connection in enumerate(complete_path):
            for stop_counter, stop in enumerate(connection):
                if full_results_bool or stop_counter == 0 or stop_counter == len(connection) - 1:
                    stop_name, dep_time, line, platform = stop
                    # Format the timedelta before displaying
                    formatted_time = convert_sec_to_hh_mm_ss(dep_time)
                    output_area.insert(tk.END, f"{stop_name:<25}{formatted_time:<15}{line:<10}{platform:<10}\n")

            if connection_counter < len(complete_path) - 1:
                current_time = complete_path[connection_counter][-1][1]
                next_time = complete_path[connection_counter + 1][0][1]
                transit_time = (next_time - current_time) // 60  # Convert to minutes

                output_area.insert(tk.END, f"{'-' * 18} Čas na přestup ({int(transit_time)} min){'-' * 18}\n")

        output_area.insert(tk.END, "-" *59 + ("\n" if connection_counter + 1 != len(complete_path) else ""))

def bind_shortcuts():
    root.bind('<Control-Return>', lambda e: on_submit())

def update_stations(event, entry, listbox, station_list):
    # Don't update if navigation keys are pressed
    if event and event.keysym in ('Up', 'Down', 'Tab'):
        return

    typed_text = entry.get().lower()
    if len(typed_text) >= 0:
        filtered_stations = [station for station in station_list if typed_text in unidecode(station.lower())]
        listbox.delete(0, tk.END)
        for station in filtered_stations:
            listbox.insert(tk.END, station)

        # Set height based on number of items, max 5
        num_items = min(len(filtered_stations), 5)
        if num_items > 0:
            listbox.configure(height=num_items)
            listbox.place(x=entry.winfo_x(), y=entry.winfo_y() + entry.winfo_height())
            listbox.lift()
        else:
            listbox.place_forget()
        
    else:
        listbox.place_forget()

def select_station(event, entry, listbox):
    if not listbox.curselection():
        return
    selected_station = listbox.get(listbox.curselection())
    entry.delete(0, tk.END)
    entry.insert(0, selected_station)
    listbox.place_forget()

def hide_listbox(event, listbox):
    listbox.place_forget()

def navigate_list(event, entry, listbox):
    if not listbox.size():  # If listbox is empty, do nothing
        return

    if event.keysym == 'Up':
        # If nothing is selected, select last item
        if not listbox.curselection():
            listbox.selection_set(listbox.size() - 1)
            listbox.see(listbox.size() - 1)
        # Otherwise move up one (wrapping to bottom)
        else:
            current = listbox.curselection()[0]
            listbox.selection_clear(0, tk.END)
            new_index = (current - 1) % listbox.size()
            listbox.selection_set(new_index)
            listbox.see(new_index)

    elif event.keysym == 'Down':
        # If nothing is selected, select first item
        if not listbox.curselection():
            listbox.selection_set(0)
        # Otherwise move down one (wrapping to top)
        else:
            current = listbox.curselection()[0]
            listbox.selection_clear(0, tk.END)
            new_index = (current + 1) % listbox.size()
            listbox.selection_set(new_index)
            listbox.see(new_index)

    elif event.keysym == 'Return':
        if listbox.curselection():  # If an item is selected
            select_station(None, entry, listbox)
            return 'break'  # Prevent default behavior

# Load cache
start_time = time.time()

CACHE_FILE = r"C:\Users\Jachym\OneDrive - České vysoké učení technické v Praze\Bakalářská_práce\02_CODE\cache\timetable_cache.pkl"
CACHE_FILE_STOP_NAME_ID = r"C:\Users\Jachym\OneDrive - České vysoké učení technické v Praze\Bakalářská_práce\02_CODE\cache\stop_name_id.pkl"

root = tk.Tk()

stop_name_to_id = build_stop_name_to_id()
station_names = sorted(stop_name_to_id.keys(), key=locale.strxfrm)  # Unique, sorted station names

root.title("Vyhledávač spojení (Dijkstra GUI)")

# Set the size and position of the window
width = 1200
height = 850
x_offset = 630
y_offset = 85

root.geometry(f"{width}x{height}+{x_offset}+{y_offset}")

label_style = {"font": ("Arial", 12, "bold"), "anchor": "e", "padx": 5, "pady": 5}
list_color = "#FFFFFF"

# Departure Station
tk.Label(root, text="Výchozí stanice:", **label_style).grid(row=0, column=0, sticky="e")
departure_combo = tk.Entry(root, width=30)
departure_combo.grid(row=0, column=1, padx=5, pady=5)
departure_combo.insert(index = 0, string = StopNames.get_a_random_stop_name('P'))

departure_listbox = tk.Listbox(root, width=30, relief="flat", bg=list_color)
departure_listbox.place_forget()  # Initially hidden

departure_listbox.bind('<ButtonRelease-1>', lambda event: select_station(event, departure_combo, departure_listbox))
departure_combo.bind('<KeyRelease>', lambda event: update_stations(event, departure_combo, departure_listbox, station_names))
departure_combo.bind('<Escape>', lambda event: hide_listbox(event, departure_listbox))
departure_combo.bind('<FocusOut>', lambda event: hide_listbox(event, departure_listbox))
departure_combo.bind('<Up>', lambda e: navigate_list(e, departure_combo, departure_listbox))
departure_combo.bind('<Down>', lambda e: navigate_list(e, departure_combo, departure_listbox))
departure_combo.bind('<Return>', lambda e: navigate_list(e, departure_combo, departure_listbox))

# Arrival Station
tk.Label(root, text="Cílová stanice:", **label_style).grid(row=1, column=0, sticky="e")
arrival_combo = tk.Entry(root, width=30)
arrival_combo.grid(row=1, column=1, padx=(0,0), pady=5)
arrival_combo.insert(index = 0, string = StopNames.get_a_random_stop_name('P'))

arrival_listbox = tk.Listbox(root, width=30, relief="flat", bg=list_color)
arrival_listbox.place_forget()  # Initially hidden

arrival_listbox.bind('<ButtonRelease-1>', lambda event: select_station(event, arrival_combo, arrival_listbox))
arrival_combo.bind('<KeyRelease>', lambda event: update_stations(event, arrival_combo, arrival_listbox, station_names))
arrival_combo.bind('<Escape>', lambda event: hide_listbox(event, arrival_listbox))
arrival_combo.bind('<FocusOut>', lambda event: hide_listbox(event, arrival_listbox))
arrival_combo.bind('<Up>', lambda e: navigate_list(e, arrival_combo, arrival_listbox))
arrival_combo.bind('<Down>', lambda e: navigate_list(e, arrival_combo, arrival_listbox))
arrival_combo.bind('<Return>', lambda e: navigate_list(e, arrival_combo, arrival_listbox))

#Čas odjezdu
# Time picker frame and label
tk.Label(root, text="Čas odjezdu:", **label_style).grid(row=2, column=0, sticky="e")
time_frame = ttk.Frame(root)
time_frame.grid(row=2, column=1, padx=5, pady=5)

# Modern time picker with hours and minutes
time_picker = SpinTimePickerModern(time_frame)
time_picker.addAll(1)  # Adds hours and minutes spinners
time_picker.configureAll(font=("Arial", 12), width=2, height=1)

time_picker.set24Hrs(8)   # Set to 14:30 (2:30 PM)
time_picker.setMins(30)
time_picker.pack(expand=False)

#Den odjezdu
START_DATE = 20250610

# Convert START_DATE to datetime object
start_date = datetime.strptime(str(START_DATE), "%Y%m%d")

# Czech month abbreviations
czech_months = ['led', 'úno', 'bře', 'dub', 'kvě', 'čvn', 'čvc', 'srp', 'zář', 'říj', 'lis', 'pro']

# Generate date options (14 days in advance)
date_options = []
actual_dates =  []

for i in range(14):
    current_date = start_date + timedelta(days=i)
    day = current_date.day
    month_abbr = czech_months[current_date.month - 1]
    date_options.append(f"{day}. {month_abbr}")
    actual_dates.append(current_date)


tk.Label(root, text="Datum", **label_style).grid(row=3, column=0, sticky="e")
date_combobox = ttk.Combobox(root, values=date_options, state="readonly", width=8)
date_combobox.current(0)  # Select first option by default
date_combobox.grid(row=3, column=1, padx=5, pady=5)


#Vyhledat trasu
submit_button = tk.Button(
    root,
    text="Vyhledat trasu",
    command=on_submit,
    bg="#808080",
    fg="white",
    font=("Arial", 10, "bold"),
    relief="raised",
    padx=10,
    pady=5
)
submit_button.grid(row=4, column=1, columnspan=1, pady=(10,0), padx = (0,0), sticky="n")  # Added sticky="n" and reduced top padding
submit_button.bind('<Return>', lambda e: on_submit())

#Soupis všech zastávek
display_all_stops_var = tk.IntVar()
display_all_stops_button = tk.Checkbutton(
    root,
    text="Zobrazit všechny zastávky trasy",
    variable=display_all_stops_var,
    font=("Arial", 10),
    padx=10,
    pady=5
)
display_all_stops_button.grid(row=4, column=2, columnspan=2, pady = (10,0), padx=(0,170), sticky="w")

#Zona P
zone_p_var = tk.StringVar(value='')
zone_p_checkbox = tk.Checkbutton(
    root,
    text="Pouze zóna P",
    variable=zone_p_var,
    onvalue='P',
    offvalue='',
    font=("Arial", 10),
    padx=10,
    pady=5
)
zone_p_checkbox.grid(row=0, column=2, columnspan=2, pady = (5,0), padx = (0,170), sticky="w")

def on_zone_change(zone):
    global station_names, stop_name_to_id
    stop_name_to_id = build_stop_name_to_id(zone)
    station_names = sorted(stop_name_to_id.keys(), key=locale.strxfrm)

    # Update station lists in combo boxes
    if root.focus_get() == departure_combo:
        update_stations(None, departure_combo, departure_listbox, station_names)
    elif root.focus_get() == arrival_combo:
        update_stations(None, arrival_combo, arrival_listbox, station_names)

zone_p_var.trace_add('write', lambda *args: on_zone_change(zone_p_var.get()))

#Output grame
output_frame = tk.Frame(root)
output_frame.grid(row=5, column=0, columnspan=4, padx=10, pady=10, sticky="new")

# Create 4 output areas arranged in a 1x4 grid
#output_areas = []
NUM_OF_SOLUTIONS = 2

def create_output_areas(solution_num, all_paths):
    for widget in output_frame.grid_slaves(column=solution_num):
        widget.destroy()
    
    output_areas = []

    if not all_paths:
        height = 3
        num_of_rows = 1

    else:
        num_of_rows = len(all_paths)

    for i in range(num_of_rows):
        if all_paths:
            height = 2 + len(all_paths[i]) * 3 if not bool(display_all_stops_var.get()) else sum(len(conn) for conn in all_paths[i]) + 2 + len(all_paths[i])
        output_area = tk.Text(output_frame, wrap="word", relief="solid", borderwidth=1, height= height)
        output_area.grid(row=i, column=solution_num, padx=5, pady=5, sticky="new")  # or remove sticky entirely
        output_areas.append(output_area)
        output_frame.grid_rowconfigure(i, weight=0)
        output_frame.grid_columnconfigure(i, weight=1)

    return output_areas





# Configure grid weights for output frame
#output_frame.grid_rowconfigure(0, weight=1)
#for i in range(NUM_OF_SOLUTIONS):
#    output_frame.grid_columnconfigure(i, weight=1)

# Configure grid weights to make the output area resize with the window
root.grid_rowconfigure(4, weight=0)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# Start loading cache in the background
threading.Thread(target=load_cache_async, daemon=True).start()
bind_shortcuts()
# Bind the shortcut after creating the GUI
root.mainloop()
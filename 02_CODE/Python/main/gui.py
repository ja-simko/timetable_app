import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime, timedelta  # Ensure datetime is imported
import joblib
import time
from pareto_before_deleting_queries_SQL import run_algorithm, get_departure_and_arrival_id_from_name
import threading
import re
import random
from unidecode import unidecode

import locale

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
cache = None
display_all_stops_var = None 

def load_cache_async():
    global cache
    try:
        cache = joblib.load(CACHE_FILE)
        print("Cache loaded.")
        submit_button.config(bg="#DE3163")  # Change to green when cache is loaded
    except Exception as e:
        print(f"Failed to load cache: {e}")
        submit_button.config(bg="#000000")  # Change to red if cache loading fails

def on_submit():
    start_time = time.time()
    departure = departure_combo.get()
    arrival = arrival_combo.get()
    
    if departure == '':
        departure = 'albertov'
        arrival = 'ricany'
    time_str = time_entry.get()
    date = date_entry.get()

    stop_name_to_id = cache['stop_name_to_station_id']

    departure_station_id, arrival_station_id = get_departure_and_arrival_id_from_name(departure, arrival, stop_name_to_id)

    try:
        dep_time = datetime.strptime(time_str, "%H:%M") if time_str != '' else datetime.now()
        time_str = dep_time.strftime("%H:%M:%S")
        time_str = str(int(time_str[:2]) + int(date)*24) + time_str[2:] 


    except ValueError:
        messagebox.showerror("Chyba", "Zadej čas ve formátu HH:MM")
        return

    try:
        route_exists, all_paths = run_algorithm(departure_station_id, arrival_station_id, time_str, cache)
        if not route_exists:
            output_area.insert(tk.END, "Žádná trasa nenalezena.")
        else:
            full_results_bool = bool(display_all_stops_var.get()) 
            insert_results(all_paths, full_results_bool)

    except Exception as e:
        messagebox.showerror("Chyba při hledání trasy", str(e))
    
    print(time.time() - start_time)

def insert_results(all_paths, full_results_bool):
    output_area.delete('1.0', tk.END)
    header = f"{'Stop':<25}{'Dep. Time':<15}{'Line':<10}{'Platform':<10}\n"

    for complete_path in all_paths:
        output_area.insert(tk.END, header)
        output_area.insert(tk.END, "-" * 60 + "\n")
        
        for connection_counter, connection in enumerate(complete_path):
            for stop_counter, stop in enumerate(connection):
                if full_results_bool or stop_counter == 0 or stop_counter == len(connection) - 1:
                    stop_name, dep_time, line, platform = stop  
                    output_area.insert(tk.END, f"{stop_name:<25}{dep_time:<15}{line:<10}{platform:<10}\n")

            if connection_counter < len(complete_path) - 1: 
                print(type(complete_path[connection_counter][-1][1]))

                current_time = datetime.strptime(complete_path[connection_counter][-1][1], "%H:%M:%S") #incorporate 24h, so that 24:30 shows as 0:30
                next_time = datetime.strptime(complete_path[connection_counter + 1][0][1], "%H:%M:%S")
                transit_time = (next_time - current_time).total_seconds() / 60  # Convert to minutes
                output_area.insert(tk.END, f"{'-' * 22} Transit ({int(transit_time)} min){'-' * 22}\n")

        output_area.insert(tk.END, "=" * 60 + "\n\n")

def bind_shortcuts():
    root.bind('<Control-Return>', lambda e: on_submit())

def update_stations(event, entry, listbox, station_list):
    # Don't update if navigation keys are pressed
    if event and event.keysym in ('Up', 'Down', 'Tab'):
        return
        
    typed_text = entry.get().lower()
    if len(typed_text) >= 3:
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

stop_name_id_dict = joblib.load(CACHE_FILE_STOP_NAME_ID)

station_names = sorted(stop_name_id_dict.keys(), key=locale.strxfrm)  # Unique, sorted station names

root = tk.Tk()
root.title("Vyhledávač spojení (Dijkstra GUI)")

# Set the size and position of the window
height = 900
width = 700
x_offset = 400
y_offset = 200

root.geometry(f"{height}x{width}+{x_offset}+{y_offset}")

label_style = {"font": ("Arial", 10, "bold"), "anchor": "e", "padx": 5, "pady": 5}
list_color = "#FFFFFF"

# Departure Station
tk.Label(root, text="Výchozí stanice:", **label_style).grid(row=0, column=0, sticky="e")
departure_combo = tk.Entry(root, width=30)
departure_combo.grid(row=0, column=1, padx=5, pady=5)
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
arrival_combo.grid(row=1, column=1, padx=5, pady=5)
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
tk.Label(root, text="Čas odjezdu (HH:MM):", **label_style).grid(row=2, column=0, sticky="e")
time_entry = tk.Entry(root, width=15)
time_entry.grid(row=2, column=1, padx=5, pady=5)

#Den odjezdu
tk.Label(root, text="Datum", **label_style).grid(row=3, column=0, sticky="e")
date_entry = tk.Entry(root, width=10)
date_entry.grid(row=3, column=1, padx=5, pady=5)

#Vyhledat trasu
submit_button = tk.Button(
    root, 
    text="Vyhledat trasu", 
    command=on_submit, 
    bg="#0078D7", 
    fg="white", 
    font=("Arial", 10, "bold"), 
    relief="raised", 
    padx=10, 
    pady=5
)
submit_button.grid(row=4, column=0, columnspan=2, pady=0)
submit_button.bind('<Return>', lambda e: on_submit())

#Soupis všech zastávek      
display_all_stops_var = tk.IntVar()
display_all_stops_button = tk.Checkbutton(
    root, 
    text="Zobrazit všechny zastávky", 
    variable=display_all_stops_var,
    font=("Arial", 10),
    padx=10, 
    pady=5
)
display_all_stops_button.grid(row=4, column=1, columnspan=2, pady=0)

output_area = tk.Text(root, wrap="word", relief="solid", borderwidth=1)
output_area.grid(row=5, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

# Configure grid weights to make the output area resize with the window
root.grid_rowconfigure(4, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# Start loading cache in the background
threading.Thread(target=load_cache_async, daemon=True).start()

# Bind the shortcut after creating the GUI
bind_shortcuts()

root.mainloop()

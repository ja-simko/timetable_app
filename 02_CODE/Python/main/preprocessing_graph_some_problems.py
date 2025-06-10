from collections import defaultdict
import re
from datetime import timedelta
import joblib
import os




MIN_TRANSFER_TIME = timedelta(seconds = 120)
CACHE_EDGES_DIR = r"C:\Users\Jachym\OneDrive - České vysoké učení technické v Praze\Bakalářská_práce\02_CODE\cache\edges"
CACHE_TIMETABLE = r"C:\Users\Jachym\OneDrive - České vysoké učení technické v Praze\Bakalářská_práce\02_CODE\cache\timetable_only.pkl"

build_edges(joblib.load(CACHE_TIMETABLE))
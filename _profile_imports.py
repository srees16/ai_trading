"""Temporary profiling script – delete after use."""
import sys, time

t0 = time.perf_counter()
import streamlit
from ui.styles import apply_custom_styles
from services.session import initialize_session_state
from auth.authenticator import check_authentication, render_user_menu
t1 = time.perf_counter()

from ui.pages.main_page import render_main_page
t2 = time.perf_counter()

print(f"Login form ready : {t1 - t0:.3f}s")
print(f"US Stocks import : {t2 - t1:.3f}s")
print(f"Total            : {t2 - t0:.3f}s")
print(f"pandas loaded?   : {'pandas' in sys.modules}")
print(f"yaml loaded?     : {'yaml' in sys.modules}")

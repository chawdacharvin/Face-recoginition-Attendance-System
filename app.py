import streamlit as st
import pandas as pd
import time
from datetime import datetime


ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")         
tiime = datetime.fromtimestamp(ts).strftime("%H-%M-%S") 
df = pd.read_csv("C:/Users/HP/Desktop/All in one/Python sums/face rec/attendance"+ date + ".csv")

# st.dataframe(df.style.highlight_max(axis=0))
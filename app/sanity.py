import streamlit as st

import sys, time
print("BOOT:", time.ctime(), flush=True)
sys.stdout.flush()


st.set_page_config(page_title="Sanity")
st.title("Sanity check ✅")
st.write("If you see this, Streamlit rendering works.")
st.write("Python:", sys.version)


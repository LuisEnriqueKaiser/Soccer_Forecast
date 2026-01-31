import streamlit as st
import builtins

from download_data_01 import main as download_main  # or your import variant

def run_with_streamlit_print(fn):
    # Container in the UI for log output
    log_area = st.empty()
    lines = []

    real_print = builtins.print  # keep original

    def streamlit_print(*args, **kwargs):
        # Build the line as print would
        line = " ".join(str(a) for a in args)
        lines.append(line)
        # Show all logs so far in the app
        log_area.text("\n".join(lines))
        # Optionally still log to server console
        real_print(*args, **kwargs)

    # Monkeypatch print only during this call
    builtins.print = streamlit_print
    try:
        fn()
    finally:
        builtins.print = real_print  # always restore


st.title("FBref data pipeline")

if st.button("Run full download"):
    st.write("Starting download job...")
    run_with_streamlit_print(download_main)
    st.success("Download finished.")

import streamlit as st
import yaml
import os
import tempfile
import zipfile
from io import BytesIO
from config_builder import build_config_ui
from datetime import datetime

def add_src_to_sys_path():
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
add_src_to_sys_path()

from diaad.main import (
    run_read_tiers, run_analyze_digital_convo_turns, 
    run_make_POWERS_coding_files, run_analyze_POWERS_coding, 
    run_evaluate_POWERS_reliability, run_reselect_POWERS_reliability_coding
)

st.title("DIAAD Web App")

if "confirmed_config" not in st.session_state:
    st.session_state.confirmed_config = False

def zip_folder(folder_path):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zf.write(file_path, arcname)
    zip_buffer.seek(0)
    return zip_buffer

st.header("Part 1: Create or upload config file")

# Upload config or build it
config_file = st.file_uploader("Upload your config.yaml", type=["yaml", "yml"])
config = None

if config_file:
    st.session_state.confirmed_config = False  # reset if new file uploaded
    config = yaml.safe_load(config_file)
    st.success("✅ Config file uploaded")
else:
    with st.expander("No config uploaded? Build one here"):
        config = build_config_ui()
        if st.button("✅ Use this built config"):
            st.session_state.confirmed_config = True
            st.success("Built config confirmed.")

st.header("Part 2: Upload input files")

# Upload .cha files
cha_files = st.file_uploader("Upload input files", type=["cha", ".xlsx"], accept_multiple_files=True)

if (config_file or st.session_state.confirmed_config) and cha_files:
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        config["input_dir"] = input_dir
        config["output_dir"] = output_dir

        # Save uploaded .cha files
        for file in cha_files:
            with open(os.path.join(input_dir, file.name), "wb") as f:
                f.write(file.read())
        # Read config values
        tiers = run_read_tiers(config.get("tiers", {}))
        frac = config.get("reliability_fraction", 0.2)
        coders = config.get("coders", [])
        blind_columns = config.get("blind_columns", [])
        exclude_participants = config.get('exclude_participants', [])
        automate_POWERS = config.get('automate', True)

        # --- List all functions (single-select) ---
        all_functions = {
            "Analyze digital conversation turns": ("turns", None),
            "Prepare POWERS coding files": ("powers", "make"),
            "Analyze POWERS coding": ("powers", "analyze"),
            "Evaluate POWERS reliability": ("powers", "evaluate"),
            "Reselect POWERS reliability coding": ("powers", "reselect"),
        }

        st.header("Part 3: Select function to run")
        selected_label = st.selectbox("Choose a function", list(all_functions.keys()))

        if st.button("Run selected function"):
            command, action = all_functions[selected_label]

            # --- Dispatch ---
            if command == "turns":
                run_analyze_digital_convo_turns(input_dir, output_dir)

            elif command == "powers" and action == "make":
                run_make_POWERS_coding_files(
                    tiers, frac, coders, input_dir, output_dir, exclude_participants, automate_POWERS
                )

            elif command == "powers" and action == "analyze":
                run_analyze_POWERS_coding(input_dir, output_dir)
            
            elif command == "powers" and action == "evaluate":
                run_evaluate_POWERS_reliability(input_dir, output_dir)

            elif command == "powers" and action == "reselect":
                run_reselect_POWERS_reliability_coding(input_dir, output_dir)

            # --- Timestamped ZIP filename ---
            timestamp = datetime.now().strftime("%y%m%d_%H%M")
            if command == "turns":
                file_name = f"diaad_turns_output_{timestamp}.zip"
            elif command == "powers":
                file_name = f"diaad_powers_{action}_output_{timestamp}.zip"
            else:
                file_name = f"diaad_output_{timestamp}.zip"

            zip_buffer = zip_folder(output_dir)
            st.download_button(
                label="Download Results ZIP",
                data=zip_buffer,
                file_name=file_name,
                mime="application/zip"
            )

            st.success(f"{selected_label} completed! Output saved in {file_name}")


def main():
    import subprocess
    import sys
    # Launch this file with streamlit
    subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])

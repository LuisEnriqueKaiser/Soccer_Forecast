# noinspection PyPackageRequirements
import os
import sys
import subprocess

def run_script(script_name):
    """
    Helper to run a Python script as a subprocess, using the
    current Python interpreter.
    """
    command = [sys.executable, script_name]
    subprocess.run(command, check=True)

def main():
    # 1) Move into the same directory as this file, so local scripts are found
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    redownload = input("Do you want to redownload the data? (y/n): ").strip().lower()
    if redownload == 'y':
        # If the user wants to redownload, remove the existing data
        scripts = [
            "01_00_download_data.py",
            "01_01_data_matching.py",
            #"01_02_update_data.py",
            #"01_03_data_merge.py",
            #y
            #"01_04_data_prep.py"
        ]
    else: 
        # 2) List your scripts in the exact order you want them to run
        scripts = [
            "01_01_data_matching.py",
            "01_02_update_data.py",
            "01_03_data_merge.py",
            "01_04_data_prep.py"
        ]


    # 3) Run them in sequence
    for script in scripts:
        print(f"\nRunning {script} ...")
        run_script(script)
        print(f"Finished running {script}.\n")

if __name__ == "__main__":
    main()

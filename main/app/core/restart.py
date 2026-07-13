import os
import sys
import json
import platform
import subprocess

sys.path.append(os.getcwd())

from main.app.core.ui import gr_info, gpu_infos
from main.app.variables import python, translations, configs_json

def restart_app(app):
    """
    Terminates the current application instance and launches a new one.

    This function notifies the user via the UI, clears the terminal screen 
    based on the operating system, closes the existing application interface, 
    and spawns a new process running the main application script while 
    preserving initial command-line arguments (excluding '--open').
    """

    # Notify the user via UI that the application is restarting
    gr_info(translations["restart"])
    # Clear the terminal screen (supports both Windows and Unix-like systems)
    os.system("cls" if platform.system() == "Windows" else "clear")

    # Close the current UI/App instance safely
    app.close()

    # Construct and run the command to restart the application
    # It excludes the '--open' flag to prevent opening a new browser tab on restart
    subprocess.run([
        python, 
        os.path.join("main", "app", "app.py")
    ] + [arg for arg in sys.argv[1:] if arg != "--open"])

def restart(lang, theme, font, gpu, app):
    """
    Updates the configuration file with new settings and reboots the application.

    This function reads the existing JSON configuration, checks if the incoming
    parameters (language, theme, font, and GPU selection) differ from the stored
    values, updates them if necessary, writes the changes back to the file, and
    triggers the application restart sequence.

    Args:
        lang (str): The newly selected language code.
        theme (str): The newly selected UI theme name.
        font (str): The newly selected UI font name.
        gpu (str): The string representation or name of the selected GPU.
        app: The current application instance to be passed to `restart_app`.
    """

    # Load the current configurations from the JSON file
    configs = json.load(open(configs_json, "r"))

    # Update settings only if they have changed from the current configuration
    if lang != configs["language"]: configs["language"] = lang
    if theme != configs["theme"]: configs["theme"] = theme
    if font != configs["font"]: configs["font"] = font
    
    # Find the index of the selected GPU from the available GPU list
    gpu_idx = int(gpu_infos.index(gpu))
    if gpu_idx != configs["gpu_idx"]: configs["gpu_idx"] = gpu_idx

    # Save the updated configurations back to the JSON file with clean formatting
    with open(configs_json, "w") as f:
        json.dump(configs, f, indent=4)

    # Invoke the application restart routine
    restart_app(app)
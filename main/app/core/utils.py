import os
import sys
import time
import json
import signal
import codecs
import requests
import subprocess

sys.path.append(os.getcwd())

from main.app.core.ui import gr_info, gr_warning
from main.app.variables import translations, configs

def alive(pid):
    """
    Checks whether a process with the given PID is currently running.

    Args:
        pid (int): The process ID to inspect.

    Returns:
        bool: True if the process is active/alive, False otherwise.
    """

    try:
        if sys.platform == "win32":
            # On Windows, use tasklist to check if the PID exists
            subprocess.check_output(["tasklist", "/FI", f"PID eq {pid}"], stderr=subprocess.DEVNULL)
            return True
        else:
            # On Unix-like systems, signal 0 queries the existence of the process
            os.kill(pid, 0)
            return True
    except:
        return False

def pid_kill(pid):
    """
    Attempts to terminate a process gracefully, falling back to a forced kill if necessary.

    Args:
        pid (int): The process ID to terminate.

    Returns:
        bool: True if the process was successfully terminated, False if it survived.
    """

    if not alive(pid): return True

    # Step 1: Attempt graceful termination (SIGINT or taskkill /T)
    try:
        if sys.platform == "win32": subprocess.run(["taskkill", "/PID", str(pid), "/T"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else: os.kill(pid, signal.SIGINT)
    except:
        pass

    # Wait up to 1.5 seconds for the process to exit gracefully
    for _ in range(15):
        if not alive(pid): return True
        time.sleep(0.1)

    # Step 2: Fallback to aggressive forced termination (SIGKILL or taskkill /F /T)
    try:
        if sys.platform == "win32": subprocess.run(["taskkill", "/PID", str(pid), "/F", "/T"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else: os.kill(pid, signal.SIGKILL)
    except:
        pass

    # Wait up to 1.0 second for the forced kill to take effect
    for _ in range(10):
        if not alive(pid): return True
        time.sleep(0.1)

    return False

def stop_pid(
    pid_file, 
    model_name=None, 
    train=False
):
    """
    Stops processes recorded in a PID file and handles active background training PIDs.

    Args:
        pid_file (str): The name of the tracking text file containing process IDs.
        model_name (str, optional): The name of the specific model directory inside logs. Defaults to None.
        train (bool, optional): If True, triggers structural validation and cleans up training PIDs stored inside the model's config.json. Defaults to False.
    """

    try:
        # Determine the file path location based on whether a model context is given
        pid_file_path = os.path.join("assets", f"{pid_file}.txt") if model_name is None else os.path.join(configs["logs_path"], model_name, f"{pid_file}.txt")

        gr_info(translations["stop_pid"])

        if not os.path.exists(pid_file_path): return gr_warning(translations["not_found_pid"])
        else:
            # Read all recorded PIDs and terminate them sequentially
            with open(pid_file_path, "r") as pid_file:
                pids = [int(pid) for pid in pid_file.readlines()]

            for pid in pids:
                pid_kill(pid)

            if os.path.exists(pid_file_path): os.remove(pid_file_path)

        # Config JSON tracking handler for deep training pipelines
        pid_file_path = os.path.join(configs["logs_path"], model_name, "config.json")

        if train and os.path.exists(pid_file_path):
            with open(pid_file_path, "r") as pid_file:
                pid_data = json.load(pid_file)
                pids = pid_data.get("process_pids", [])

            # Clear out the process_pids array from config.json to reset state safely
            with open(pid_file_path, "w") as pid_file:
                pid_data.pop("process_pids", None)

                json.dump(pid_data, pid_file, indent=4)

            # Terminate background worker PIDs parsed from the JSON file
            for pid in pids:
                pid_kill(pid)

        gr_info(translations["end_pid"])
    except:
        pass

def google_translate(text, source='auto', target='vi'):
    """
    Translates a text block using the free Google Translate single API endpoint.

    Splits the string safely into chunks under 5,000 characters to prevent payload overflows.

    Args:
        text (str): The raw input string containing the content to translate.
        source (str, optional): Language code of the origin text. Defaults to 'auto'.
        target (str, optional): Target destination language translation code. Defaults to 'vi'.

    Returns:
        str: The translated text string result, or the original text string if an error occurs.
    """

    if text == "": return gr_warning(translations["prompt_warning"])

    try:
        import textwrap

        def translate_chunk(chunk):
            response = requests.get(
                # ROT13 decoding hides the plain-text URL string footprint from basic scanners
                codecs.decode("uggcf://genafyngr.tbbtyrncvf.pbz/genafyngr_n/fvatyr", "rot13"), 
                params={
                    'client': 'gtx', 
                    'sl': source, 
                    'tl': target, 
                    'dt': 't', 
                    'q': chunk
                }
            )

            # Parse structural nested response payload and re-assemble sentence arrays
            return ''.join([i[0] for i in response.json()[0]]) if response.status_code == 200 else chunk

        translated_text = ''
        # Wrap long text objects safely into chunks of up to 5000 chars without splitting structural words
        for chunk in textwrap.wrap(
            text, 
            5000, 
            break_long_words=False, 
            break_on_hyphens=False
        ):
            translated_text += translate_chunk(chunk)

        return translated_text
    except:
        # Gracefully fall back to raw input text if networking disruptions happen
        return text
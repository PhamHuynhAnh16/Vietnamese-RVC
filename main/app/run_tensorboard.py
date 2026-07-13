import os
import sys
import time
import logging
import warnings
import webbrowser

from tensorboard import program

sys.path.append(os.getcwd())

from main.app.variables import config, translations, logger

def launch_tensorboard():
    """
    Configures and starts a background TensorBoard server using settings 
    defined in the application configuration files.

    Returns:
        str: The local host URL where the TensorBoard instance is running.
    """

    # Production Logging Configuration: Suppress verbose loggers from TensorBoard if debug mode is disabled
    if not config.debug_mode:
        warnings.filterwarnings("ignore")
        for l in ["root", "tensorboard"]:
            logging.getLogger(l).setLevel(logging.ERROR)

    # Initialize the programmatic TensorBoard server instance
    tb = program.TensorBoard()
    # Configure arguments: setup the target log directory and bind to the designated system port
    tb.configure(
        argv=[
            None, 
            "--logdir", config.configs["logs_path"], 
            f"--port={config.configs['tensorboard_port']}"
        ]
    )

    # Fire up the background thread server and retrieve its operational URL
    url = tb.launch()
    # Log the successfully active TensorBoard endpoint destination
    logger.info(f"{translations['tensorboard_url']}: {url}")

    # Automate browser engagement: trigger local interface tab opening if '--open' argument exists
    if "--open" in sys.argv: webbrowser.open(url)

    return url

if __name__ == "__main__": 
    # Boot up the tracking analytics platform
    launch_tensorboard()

    # Infinite loop to keep the main script thread alive for standalone service persistence
    while 1:
        time.sleep(5)
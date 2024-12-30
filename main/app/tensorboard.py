import os
import sys
import json
import logging
import webbrowser

from tensorboard import program

sys.path.append(os.getcwd())

from main.configs.config import Config
translations = Config().translations

with open(os.path.join("main", "configs", "config.json"), "r") as f:
    configs = json.load(f)

def launch_tensorboard():
    for l in ["root", "tensorboard"]:
        logging.getLogger(l).setLevel(logging.ERROR)

    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", "assets/logs", f"--port={configs["tensorboard_port"]}"])
    url = tb.launch()

    print(f"{translations['tensorboard_url']}: {url}")
    if "--open" in sys.argv: webbrowser.open(url)

    return f"{translations['tensorboard_url']}: {url}"

if __name__ == "__main__": launch_tensorboard()
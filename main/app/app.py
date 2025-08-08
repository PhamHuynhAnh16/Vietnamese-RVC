import os
import io
import ssl
import sys
import time
import codecs
import logging
import warnings

import gradio as gr

sys.path.append(os.getcwd())
start_time = time.time()

from main.app.tabs.extra.extra import extra_tab
from main.app.tabs.editing.editing import editing_tab
from main.app.tabs.training.training import training_tab
from main.app.tabs.downloads.downloads import download_tab
from main.app.tabs.inference.inference import inference_tab
from main.configs.rpc import connect_discord_ipc, send_discord_rpc
from main.app.variables import logger, config, translations, theme, font, configs, language, allow_disk

ssl._create_default_https_context = ssl._create_unverified_context

warnings.filterwarnings("ignore")
for l in ["httpx", "gradio", "uvicorn", "httpcore", "urllib3"]:
    logging.getLogger(l).setLevel(logging.ERROR)

with gr.Blocks(title="📱 Vietnamese-RVC GUI BY ANH", theme=theme, css="<style> @import url('{fonts}'); * {{font-family: 'Courgette', cursive !important;}} body, html {{font-family: 'Courgette', cursive !important;}} h1, h2, h3, h4, h5, h6, p, button, input, textarea, label, span, div, select {{font-family: 'Courgette', cursive !important;}} </style>".format(fonts=font or "https://fonts.googleapis.com/css2?family=Courgette&display=swap")) as app:
    gr.HTML("<h1 style='text-align: center;'>🎵VIETNAMESE RVC BY ANH🎵</h1>")
    gr.HTML(f"<h3 style='text-align: center;'>{translations['title']}</h3>")

    with gr.Tabs():      
        inference_tab()
        editing_tab()
        training_tab()
        download_tab()
        extra_tab(app)

    with gr.Row(): 
        gr.Markdown(translations["rick_roll"].format(rickroll=codecs.decode('uggcf://jjj.lbhghor.pbz/jngpu?i=qDj4j9JtKpD', 'rot13')))

    with gr.Row(): 
        gr.Markdown(translations["terms_of_use"])

    with gr.Row():
        gr.Markdown(translations["exemption"])
    
    logger.info(config.device)
    logger.info(translations["start_app"])
    logger.info(translations["set_lang"].format(lang=language))

    port = configs.get("app_port", 7860)
    server_name = configs.get("server_name", "0.0.0.0")
    share = "--share" in sys.argv

    original_stdout = sys.stdout
    sys.stdout = io.StringIO()

    for i in range(configs.get("num_of_restart", 5)):
        try:
            _, _, share_url = app.queue().launch(
                favicon_path=configs["ico_path"], 
                server_name=server_name, 
                server_port=port, 
                show_error=configs.get("app_show_error", False), 
                inbrowser="--open" in sys.argv, 
                share=share, 
                allowed_paths=allow_disk,
                prevent_thread_lock=True,
                quiet=True
            )
            break
        except OSError:
            logger.debug(translations["port"].format(port=port))
            port -= 1
        except Exception as e:
            logger.error(translations["error_occurred"].format(e=e))
            sys.exit(1)
    
    sys.stdout = original_stdout

    if configs.get("discord_presence", True):
        pipe = connect_discord_ipc()
        if pipe:
            try:
                logger.info(translations["start_rpc"])
                send_discord_rpc(pipe)
            except KeyboardInterrupt:
                logger.info(translations["stop_rpc"])
                pipe.close()

    logger.info(f"{translations['running_local_url']}: {server_name}:{port}")
    if share: logger.info(f"{translations['running_share_url']}: {share_url}")
    logger.info(f"{translations['gradio_start']}: {(time.time() - start_time):.2f}s")

    while 1:
        time.sleep(5)
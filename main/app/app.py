import os
import io
import ssl
import sys
import time
import types
import codecs
import logging
import warnings

import gradio as gr

sys.path.append(os.getcwd())
# Track application startup time performance
start_time = time.time()

from main.app.variables import logger, config, translations, theme, font, configs, language, allow_disk, gradio_version

from main.app.core.realtime import js_code
from main.app.tabs.extra.extra import extra_tab
from main.app.tabs.editing.editing import editing_tab
from main.app.tabs.training.training import training_tab
from main.app.tabs.realtime.realtime import realtime_tab
from main.app.tabs.downloads.downloads import download_tab
from main.app.tabs.inference.inference import inference_tab
from main.configs.rpc import connect_discord_ipc, send_discord_rpc

# Global SSL Fix: Bypass SSL verification for model/file downloads via HTTPS
ssl._create_default_https_context = ssl._create_unverified_context

if not config.debug_mode:
    warnings.filterwarnings("ignore")
    for l in ["httpx", "gradio", "uvicorn", "httpcore", "urllib3"]:
        logging.getLogger(l).setLevel(logging.ERROR)

# Windows Platform Hotfix: Prevent crashing on 'ConnectionResetError' within asyncio proactor events
if sys.platform == "win32":
    import asyncio.proactor_events as _pe

    _orig_ccl = _pe._ProactorBasePipeTransport._call_connection_lost

    def _ccl_patched(self, exc):
        try:
            _orig_ccl(self, exc)
        except ConnectionResetError:
            pass

    _pe._ProactorBasePipeTransport._call_connection_lost = _ccl_patched

# Fix Gradio NoneType error when entering an invalid value
gr.Number.preprocess = types.MethodType(
    lambda self, payload: (
        None
        if payload is None
        or (self.minimum is not None and payload < self.minimum)
        or (self.maximum is not None and payload > self.maximum)
        else self.round_to_precision(payload, self.precision)
    ), 
    gr.Number
)

# Dynamic CSS generation for custom fonts injecting into the Gradio UI frontend
css = """
<style>
  @import url('{font_urls}');

  * {{
    font-family: '{fonts}', cursive !important;
  }}

  html,
  body {{
    font-family: '{fonts}', cursive !important;
  }}

  h1, h2, h3, h4, h5, h6,
  p,
  button,
  input,
  textarea,
  label,
  span,
  div,
  select {{
    font-family: '{fonts}', cursive !important;
  }}
</style>
""".format(
    font_urls=font or "https://fonts.googleapis.com/css2?family=Saira&display=swap",
    fonts=(
        font or "https://fonts.googleapis.com/css2?family=Saira&display=swap"
    ).replace("https://fonts.googleapis.com/css2?family=", "").replace("+", " ").split(":")[0].split("&")[0]

)

# Check if the application is running in Client Mode based on CLI arguments
client_mode = "--client" in sys.argv

# Configure UI operational parameters (JS scripts handling depends on Gradio core version)
gr_params = {
    "js": (
        ("() => {\n" + js_code + "\n}") 
        if gradio_version else 
        js_code
    ) if client_mode else None, 
    "theme": theme, 
    "css": css
}

# Construct the Main Gradio Web UI Layout
with gr.Blocks(
    title="📱 Vietnamese-RVC GUI",
    **gr_params if gradio_version else {}
) as app:
    # Render Application Header
    gr.HTML("<h1 style='text-align: center;'>🎵VIETNAMESE RVC🎵</h1>", padding=True)
    gr.HTML(f"<h3 style='text-align: center;'>{translations['title']}</h3>", padding=True)

    # Render Functional Navigation Tabs dynamically based on user configuration settings
    with gr.Tabs():      
        if configs.get("inference_tab", True): inference_tab()
        if configs.get("editing_tab", True): editing_tab()
        if configs.get("realtime_tab", True) and not config.is_zluda: realtime_tab()
        if configs.get("create_and_training_tab", True): training_tab()
        if configs.get("downloads_tab", True): download_tab()
        if configs.get("extra_tab", True): extra_tab(app)

    # This thing is useless; it's just there to make someone curious and get trolled.
    with gr.Row(): 
        gr.Markdown(
            translations["rick_roll"].format(
                rickroll=codecs.decode('uggcf://jjj.lbhghor.pbz/jngpu?i=qDj4j9JtKpD', 'rot13')
            )
        )

    # Footer Section: Terms of Service info
    with gr.Row(): 
        gr.Markdown(translations["terms_of_use"])

    # Footer Section: Exemption/Disclaimer agreements
    with gr.Row():
        gr.Markdown(translations["exemption"])
    
    # Application Startup Sequence
    if __name__ == "__main__":
        # Hardware acceleration environment mapping (DirectML, ZLUDA AMD, ROCm)
        device = config.device.replace("privateuseone", "dml")
        if config.is_zluda: device = device.replace("cuda", "hip")
        if config.hip_version is not None: device = device.replace("cuda", "rocm")

        # Log system capabilities, execution backend, models compilation status, and computing precision
        logger.info(f"Pytorch: {device} | Onnxruntime: {config.providers[0][0].replace('Dml', 'Ocl') if device.startswith('ocl') else config.providers[0][0].replace('Dml', 'ROCM') if config.hip_version and device.startswith('cuda') else config.providers[0][0]}")
        if config.compile_all: logger.info(translations["compile_model"].format(compile_mode=config.compile_mode))
        logger.info(f'{translations["precision"]}: {("BF16" if config.brain else "FP16") if config.is_half else "FP32"}')
        logger.info(translations["start_app"])
        logger.info(translations["set_lang"].format(lang=language))

        # Retrieve server network configuration targets from settings
        port = configs.get("app_port", 7860)
        server_name = configs.get("server_name", "0.0.0.0")
        share = "--share" in sys.argv

        # Temporarily redirect standard output streams to suppress redundant engine logs during launch
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()

        # Retry loop to find an available network port if the designated one is occupied
        for i in range(configs.get("num_of_restart", 5)):
            try:
                gradio_app, _, share_url = app.queue().launch(
                    favicon_path=configs["ico_path"], 
                    server_name=server_name, 
                    server_port=port, 
                    show_error=configs.get("app_show_error", False), 
                    inbrowser="--open" in sys.argv, 
                    share=share, 
                    allowed_paths=allow_disk,
                    prevent_thread_lock=True,
                    quiet=not config.debug_mode,
                    **gr_params if not gradio_version else {}
                )
                break # Port successfully bound, break retry cycle
            except OSError:
                logger.debug(translations["port"].format(port=port))
                port -= 1 # Shift port downwards and retry
            except Exception as e:
                logger.error(translations["error_occurred"].format(e=e))
                sys.exit(1)

        # Client Mode Routing: Mount custom FastAPI microservices under Gradio framework
        if client_mode:
            logger.debug(translations["mount_fastapi"])

            from main.inference.realtime.client import app as fastapi_app
            gradio_app.mount("/api", fastapi_app)
        
        # Flush buffered engine startup logs back to standard debugging console output
        logger.debug("\n" + sys.stdout.getvalue())
        sys.stdout = original_stdout

        # Discord Integration: Establish a pipeline for active Rich Presence status updates
        if configs.get("discord_presence", True) and "--no_discord" not in sys.argv:
            pipe = connect_discord_ipc()
            if pipe:
                try:
                    logger.info(translations["start_rpc"])
                    send_discord_rpc(pipe)
                except KeyboardInterrupt:
                    logger.info(translations["stop_rpc"])
                    pipe.close()

        # Display server URL endpoints
        logger.info(f"{translations['running_local_url']}: {server_name}:{port}")
        if share: logger.info(f"{translations['running_share_url']}: {share_url}")

        # Optional: Initialize and run background TensorBoard analytics instances if requested
        if "--tensorboard" in sys.argv:
            logger.info(translations["run_tensorboard"])

            from main.app.run_tensorboard import launch_tensorboard
            launch_tensorboard()

        # Log total initialization performance benchmark metric
        logger.info(f"{translations['gradio_start']}: {(time.time() - start_time):.2f}s")

        # Keep the main process thread alive to maintain active web application instances
        while 1:
            time.sleep(5)
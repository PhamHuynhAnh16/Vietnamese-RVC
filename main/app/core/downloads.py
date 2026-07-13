import os
import re
import sys
import json
import codecs
import shutil
import yt_dlp
import warnings
import requests

from bs4 import BeautifulSoup

sys.path.append(os.getcwd())

from main.tools import huggingface, gdown, meganz, mediafire, pixeldrain
from main.app.variables import logger, translations, model_options, configs
from main.app.core.process import move_files_from_directory, fetch_pretrained_data, extract_name_model
from main.app.core.ui import gr_info, gr_warning, gr_error, process_output, replace_url, replace_modelname

def download_url(url = None):
    """
    Downloads audio from a given URL (YouTube, etc.) and converts it to a standard WAV file.

    Args:
        url (str, optional): The target video or audio URL. Defaults to None.

    Returns:
        List[Union[str, None]]: A list containing the absolute audio path twice 
        and a success message indicator. Returns [None, None, None] if validation fails.
    """

    if not url: 
        gr_warning(translations["provide_url"])
        return [None]*3

    # Ensure the destination directory for audio storage exists
    if not os.path.exists(configs["audios_path"]): os.makedirs(configs["audios_path"], exist_ok=True)

    # Suppress internal library warnings from yt_dlp during runtime execution
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        # Define yt-dlp processing parameters tailored for highest quality audio extraction
        ydl_opts = {
            "format": "bestaudio/best", 
            "postprocessors": [{
                "key": "FFmpegExtractAudio", 
                "preferredcodec": "wav", 
                "preferredquality": "192"
            }], 
            "quiet": True, 
            "no_warnings": True, 
            "noplaylist": True, 
            "verbose": False
        }

        gr_info(translations["start_download"])
        audio_output = ""

        # Dry-run execution to safely fetch the video title and format a secure filename
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Replace concurrent whitespaces with a single uniform hyphen delimiter
            audio_output = process_output(os.path.join(
                configs["audios_path"], 
                re.sub(
                    r'\s+', '-', 
                    # Extract title and sanitize it by removing special regex patterns across languages
                    re.sub(
                        r'[^\w\s\u4e00-\u9fff\uac00-\ud7af\u0400-\u04FF\u1100-\u11FF]', 
                        '', 
                        ydl.extract_info(url, download=False).get('title', 'video')
                    ).strip()
                )
            ) + ".wav")

            audio_output_dir = audio_output.replace(".wav", "")
            # Clear temporary or existing conflicting directories with the same target signature
            if os.path.exists(audio_output_dir): 
                shutil.rmtree(audio_output_dir, ignore_errors=True)

            # Assign output template mapping configuration path dynamically
            ydl_opts['outtmpl'] = audio_output_dir

        # Execute active download stream session via standard context container pipeline
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: 
            ydl.download([url])

        gr_info(translations["success"])

        return [
            audio_output, 
            audio_output, 
            translations["success"]
        ]

def move_file(
    file, 
    download_dir, 
    model,
    use_orig_weight_name=False
):
    """
    Unpacks zip archives if required and transfers model weights/logs to their designated directories.

    Args:
        file (str): Absolute file path of the downloaded asset.
        download_dir (str): Sandbox extraction tracking path directory.
        model (str): Target model naming profile identifier.
        use_orig_weight_name (bool, optional): Retain original weights filename format flag. Defaults to False.
    """

    weights_dir = configs["weights_path"]
    logs_dir = configs["logs_path"]

    # Secure asset storage directory trees infrastructure structures
    if not os.path.exists(weights_dir): os.makedirs(weights_dir, exist_ok=True)
    if not os.path.exists(logs_dir): os.makedirs(logs_dir, exist_ok=True)

    # Automatically extract archived packages prior to organizing internal architecture assets
    if file.endswith(".zip"): shutil.unpack_archive(file, download_dir)
    # Delegate standard classification mapping file placement migration processing pipelines
    move_files_from_directory(download_dir, weights_dir, logs_dir, model, use_orig_weight_name)

def download_model(url=None, model=None):
    """
    Routes arbitrary hosting URLs to provider-specific wrappers to fetch and process RVC models.

    Supports HuggingFace, Google Drive, MediaFire, PixelDrain, and Mega.nz URLs.

    Args:
        url (str, optional): Target provider download address hyperlink routing endpoint. Defaults to None.
        model (str, optional): Overriding label custom nomenclature naming flag identifier. Defaults to None.

    Returns:
        str: Context matched localization notification string state tracking execution output.
    """

    if not url: return gr_warning(translations["provide_url"])
    logger.debug(model + ": " + url)

    url = replace_url(url)
    download_dir = "download_model"
    use_orig_weight_name = False

    os.makedirs(download_dir, exist_ok=True)
    
    try:
        gr_info(translations["start_download"])

        # Service provider dynamic platform interface router mapping allocation
        if "huggingface.co" in url: 
            file = huggingface.HF_download_file(
                url, 
                download_dir
            )
        elif "google.com" in url: 
            file = gdown.gdown_download(
                url, 
                download_dir
            )
        elif "mediafire.com" in url: 
            file = mediafire.Mediafire_Download(
                url, 
                download_dir
            )
        elif "pixeldrain.com" in url: 
            file = pixeldrain.pixeldrain(
                url, 
                download_dir
            )
        elif "mega.nz" in url: 
            file = meganz.mega_download_url(
                url, 
                download_dir
            )
        else:
            gr_warning(translations["not_support_url"])
            return translations["not_support_url"]
        
        # Deduce missing explicit naming contexts by reading core filename variants safely
        if not model: 
            use_orig_weight_name = True
            modelname = os.path.basename(file)

            # Extract fallback labels from index references or base string naming schemas
            model = (
                extract_name_model(modelname) 
                if modelname.endswith(".index") else 
                os.path.splitext(modelname)[0]
            )

            if model is None: model = os.path.splitext(modelname)[0]

        # Normalize naming syntax formats safely before dispatching local migration operations
        model = replace_modelname(model)
        move_file(file, download_dir, model, use_orig_weight_name)

        gr_info(translations["success"])
        return translations["success"]
    except Exception as e:
        gr_error(message=translations["error_occurred"].format(e=e))

        return translations["error_occurred"].format(e=e)
    finally:
        # Clear localized runtime sandbox scratch-pad operational work directories safely
        shutil.rmtree(download_dir, ignore_errors=True)
        
def download_pretrained_model(
    choices, 
    model, 
    sample_rate
):
    """
    Handles retrieval of official or custom tracking pretrained foundation baseline model checkpoints.

    Args:
        choices (str): Selected interface operation execution path mode context.
        model (str): Pretrained identifier name or direct hyperlink access reference value pattern.
        sample_rate (str): Sample rate identifier tag matching targeting resolution limits.

    Returns:
        Union[str, List[None]]: Returns localization strings or lists based on specific routing configurations.
    """

    pretraineds_custom_path = configs["pretrained_custom_path"]

    if choices == translations["list_model"]:
        paths = fetch_pretrained_data()[model][sample_rate]
        if not os.path.exists(pretraineds_custom_path): 
            os.makedirs(pretraineds_custom_path, exist_ok=True)

        # Deobfuscate internal storage project links protected via lightweight ROT13 obfuscation
        url = codecs.decode(
            "uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/cergenvarq_phfgbz/", 
            "rot13"
        ) + paths

        gr_info(translations["download_pretrain"])

        file = huggingface.HF_download_file(
            replace_url(url), 
            os.path.join(pretraineds_custom_path, paths)
        )

        # Automatically unpack standard distribution compressed zip archives to clean target dependencies
        if file.endswith(".zip"): 
            shutil.unpack_archive(file, pretraineds_custom_path)
            os.remove(file)

        gr_info(translations["success"])
        return translations["success"]
    elif choices == translations["download_url"]:
        # Inline state evaluation mapping checking whether endpoints link to compressed structures
        pretrain_is_zip = (
            model.endswith(".zip") or 
            model.endswith(".zip?download=true") or 
            sample_rate.endswith(".zip") or 
            sample_rate.endswith(".zip?download=true")
        )

        urls = []
        # Enforce validation checking to protect processing loops from missing arguments
        if not model and not pretrain_is_zip: 
            gr_warning(translations["provide_pretrain"].format(dg="D"))
            return [None]*2

        if not sample_rate and not pretrain_is_zip: 
            gr_warning(translations["provide_pretrain"].format(dg="G"))
            return [None]*2

        gr_info(translations["download_pretrain"])

        if model: urls.append(model)
        if sample_rate: urls.append(sample_rate)

        # Iterate structural collection mapping lists sequentially
        for url in urls:
            url = replace_url(url)
            # External decentralized cloud network file extraction architecture interfaces routing
            if "huggingface.co" in url: 
                file = huggingface.HF_download_file(
                    url, 
                    pretraineds_custom_path
                )
            elif "google.com" in url: 
                file = gdown.gdown_download(
                    url, 
                    pretraineds_custom_path
                )
            elif "mediafire.com" in url: 
                file = mediafire.Mediafire_Download(
                    url, 
                    pretraineds_custom_path
                )
            elif "pixeldrain.com" in url: 
                file = pixeldrain.pixeldrain(
                    url, 
                    pretraineds_custom_path
                )
            elif "mega.nz" in url: 
                file = meganz.mega_download_url(
                    url, 
                    pretraineds_custom_path
                )
            else:
                gr_warning(translations["not_support_url"])
                return translations["not_support_url"], translations["not_support_url"]
            
            # Post-download structural checking extraction phase operations optimization
            if file.endswith(".zip"):
                shutil.unpack_archive(file, pretraineds_custom_path)
                if os.path.exists(file): os.remove(file)

        gr_info(translations["success"])
        return translations["success"], translations["success"]
    
def is_empty_table(html):
    """
    Checks if the returned scraped HTML string represents an empty response placeholder layout structure.

    Args:
        html (str): Raw string containing target HTML elements snippets.

    Returns:
        bool: True if matched to an empty layout design component signature block, else False.
    """

    return ('<td colspan=' in html and 'text-center' in html)

def fetch_models_data(search):
    """
    Queries voice-models repository backend APIs sequentially to harvest matching indexed tracking sheets.

    Args:
        search (str): Matching query keyword string profile token filtering targeting parameters.

    Returns:
        List[str]: Aggregated collections list housing raw HTML content snippets data arrays.
    """

    all_table_data = [] 
    models = set()
    page = 0

    while 1:
        try:
            # Obfuscated root destination routing targets handling
            url = codecs.decode("uggcf://ibvpr-zbqryf.pbz", "rot13")
            response = requests.post(
                url=codecs.decode(
                    "uggcf://ibvpr-zbqryf.pbz/srgpu_qngn.cuc", 
                    "rot13"
                ), 
                data={
                    "page": page, 
                    "search": search
                },
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/148.0.0.0 Safari/537.36"
                    ),
                    "Referer": url + "/",
                    "Origin": url
                }
            )

            if response.status_code == 200:
                table_data = response.json().get("table", "")

                # Terminate infinite pagination loop sequences when empty tracking thresholds occur
                if not table_data.strip(): break
                if is_empty_table(table_data): break

                # Extract explicit relative model URL matching strings via html analytical parsing engines
                model_links = {a.get("href") for a in BeautifulSoup(table_data, "html.parser").select('a[href^="/model/"]')}
                if not model_links: break

                # Validate data delta to drop redundant loops processing recurrent duplicate parameters
                new_models = model_links - models
                if not new_models: break

                # Append newly discovered unique indices sets to active track layout profiles
                models.update(new_models)
                all_table_data.append(table_data)
                page += 1
            else:
                logger.debug(f"{translations['code_error']} {response.status_code}")
                break  
        except json.JSONDecodeError:
            logger.debug(translations["json_error"])
            break
        except requests.RequestException as e:
            logger.debug(translations["requests_error"].format(e=e))
            break

    return all_table_data

def search_models(name):
    """
    Searches for indexing models by tag identity, parses parameters, and updates Gradio UI component values.

    Args:
        name (str): Key naming search filter token phrase supplied by the frontend interface.

    Returns:
        List[Union[Dict[str, Any], None]]: Gradio structured configuration response payload update mappings.
    """

    if not name: 
        gr_warning(translations["provide_name"])
        return [None]*2

    gr_info(translations["start_search"])

    tables = fetch_models_data(name)

    if len(tables) == 0:
        gr_info(translations["not_found"].format(name=name))
        return [None]*2
    else:
        from urllib.parse import unquote

        # Clear global model options dictionary cache map records prior to re-indexing new responses
        model_options.clear()
        
        for table in tables:
            for row in BeautifulSoup(table, "html.parser").select("tr"):
                name_tag, url_tag = (
                    row.find("a", {"class": "fs-5"}), 
                    row.find("a", {"class": "btn btn-sm fw-bold btn-light ms-0 p-1 ps-2 pe-2"})
                )

                # Sanitize routing parameters URLs by purging internal wrapper proxy redirects links
                url = unquote(url_tag["href"].replace("https://easyaivoice.com/run?url=", ""))

                # Isolate and filter tracking metrics restricted exclusively to the Huggingface eco-network
                if ("huggingface" in url) and (name_tag and url_tag): 
                    model_options[replace_modelname(name_tag.text)] = url

        logger.debug(model_options)
        gr_info(translations["found"].format(results=len(model_options)))

        # Pack and structure explicit structural layout configurations matching Gradio UI dynamic rendering rules
        return [
            {
                "value": "", 
                "choices": model_options, 
                "interactive": True, 
                "visible": True, 
                "__type__": "update"
            }, 
            {
                "value": translations["downloads"], 
                "visible": True, 
                "__type__": "update"
            }
        ]
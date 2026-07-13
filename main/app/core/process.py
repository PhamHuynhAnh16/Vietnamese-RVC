import os
import re
import sys
import shutil
import codecs
import zipfile
import requests

sys.path.append(os.getcwd())

from main.app.variables import logger, translations, configs
from main.app.core.ui import gr_info, gr_warning, gr_error, process_output, replace_punctuation

def read_docx_text(path):
    """
    Extracts and merges all text content from a Microsoft Word (.docx) file.

    Args:
        path (str): The absolute or relative path to the .docx file.

    Returns:
        str: A single string containing all paragraphs separated by newlines.
    """

    import xml.etree.ElementTree
    # Open the docx container as a zip archive and read the main XML content
    with zipfile.ZipFile(path) as docx:
        with docx.open("word/document.xml") as document_xml:
            xml_content = document_xml.read()

    # The exact OpenXML schema URL encoded in ROT13 to hide it from plain text strings
    WORD_NAMESPACE = codecs.decode(
        "{uggc://fpurznf.bcrakzysbezngf.bet/jbeqcebprffvatzy/2006/znva}",
        "rot13"
    )
    paragraphs = []
    # Parse the XML structure and iterate through all paragraph nodes ('p')
    for paragraph in xml.etree.ElementTree.XML(xml_content).iter(WORD_NAMESPACE + 'p'):
        # Extract and join all text nodes ('t') within the current paragraph
        texts = [node.text for node in paragraph.iter(WORD_NAMESPACE + 't') if node.text]
        if texts: paragraphs.append(''.join(texts))

    return '\n'.join(paragraphs)

def process_input(file_path):
    """
    Reads the contents of a file based on its extension and triggers UI feedback.

    Args:
        file_path (str): The path to the uploaded file.

    Returns:
        str: The extracted text content from the file, or an empty string if reading fails.
    """

    # SRT files are initialized as empty (possibly handled in another workflow step)
    if file_path.endswith(".srt"): 
        file_contents = ""
    elif file_path.endswith(".docx"): 
        file_contents = read_docx_text(file_path)
    else:
        try:
            # Fallback to standard UTF-8 plain text reading for other file formats
            with open(file_path, "r", encoding="utf-8") as file:
                file_contents = file.read()
        except Exception as e:
            # Handle reading exceptions gracefully and notify user via UI
            gr_warning(translations["read_error"])
            logger.debug(e)
            file_contents = ""

    # Notify user of successful upload/processing
    gr_info(translations["upload_success"])
    return file_contents

def move_files_from_directory(
    src_dir, 
    dest_weights, 
    dest_logs, 
    model_name,
    use_orig_weight_name=False
):
    """
    Walks through a directory to sort and move model weights, logs, and indices.

    Args:
        src_dir (str): Source directory containing the unzipped model files.
        dest_weights (str): Target directory for .pth and .onnx weight files.
        dest_logs (str): Target directory for model logs and .index files.
        model_name (str): The standardized name used for naming or structuring folders.
        use_orig_weight_name (bool, optional): If True, keeps the original file name 
            instead of renaming it to model_name. Defaults to False.
    """

    for root, _, files in os.walk(src_dir):
        for file in files:
            file_path = os.path.join(root, file)

            if file.endswith(".index"):
                # Case 1: Process index files (Retrieval-based Voice Conversion indexes)
                model_log_dir = os.path.join(dest_logs, model_name)
                os.makedirs(model_log_dir, exist_ok=True)

                # Clean and sanitize the output file name
                filepath = process_output(os.path.join(model_log_dir, replace_punctuation(file)))
                shutil.move(file_path, filepath)
            elif file.endswith(".pth") and not file.startswith("D_") and not file.startswith("G_"):
                # Case 2: Process PyTorch weight files, excluding generator/discriminator training states (G_ / D_)
                pth_path = process_output(os.path.join(dest_weights, model_name + ".pth"))
                shutil.move(file_path, pth_path if not use_orig_weight_name else dest_weights)
            elif file.endswith(".onnx") and not file.startswith("D_") and not file.startswith("G_"):
                # Case 3: Process ONNX inference files, excluding training checkpoints
                onnx_path = process_output(os.path.join(dest_weights, model_name + ".onnx"))
                shutil.move(file_path, onnx_path if not use_orig_weight_name else dest_weights)

def extract_name_model(filename):
    """
    Extracts the model name from a string using a regex pattern (e.g., '_modelname_v2').

    Args:
        filename (str): The filename or path string to parse.

    Returns:
        Optional[str]: The extracted model name if matched, otherwise None.
    """

    # Regex looks for a string pattern between an underscore and a version tag '_vX'
    match = re.search(
        r"_([A-Za-z0-9]+)(?=_v\d*)", 
        replace_punctuation(filename)
    )

    return match.group(1) if match else None

def save_drop_model(dropboxs):
    """
    Processes, extracts, and organizes files dropped or uploaded into the workspace.
    Handles zip archives, individual model weights (.pth, .onnx), and index files.

    Args:
        dropboxs (List[str]): List of file paths pointing to the dropped/uploaded items.
    """

    weight_folder = configs["weights_path"]
    logs_folder = configs["logs_path"]
    save_model_temp = "save_model_temp"

    # Ensure all target and temporary directories exist
    if not os.path.exists(weight_folder): os.makedirs(weight_folder, exist_ok=True)
    if not os.path.exists(logs_folder): os.makedirs(logs_folder, exist_ok=True)
    if not os.path.exists(save_model_temp): os.makedirs(save_model_temp, exist_ok=True)

    try:
        for dropbox in dropboxs:
            # Move individual items to the temporary workspace first
            shutil.move(dropbox, save_model_temp)
            file_name = os.path.basename(dropbox)

            if file_name.endswith(".zip"):
                # Scenario A: Processing a bundled zip package
                shutil.unpack_archive(
                    os.path.join(save_model_temp, file_name), 
                    save_model_temp
                )

                # Sort out unpacked contents into weights and logs folders
                move_files_from_directory(
                    save_model_temp, 
                    weight_folder, 
                    logs_folder, 
                    file_name.replace(".zip", "")
                )
            elif file_name.endswith((".pth", ".onnx")): 
                # Scenario B: Processing a direct standalone model weight file
                output_file = process_output(os.path.join(weight_folder, file_name))
                
                shutil.move(os.path.join(save_model_temp, file_name), output_file)
            elif file_name.endswith(".index"):
                # Scenario C: Processing a direct index file

                # Try to extract the model name from the index file name pattern
                modelname = extract_name_model(file_name)
                # Fallback to the filename without extension if regex fails
                if modelname is None: 
                    modelname = os.path.splitext(os.path.basename(file_name))[0]

                model_logs = os.path.join(logs_folder, modelname)
                if not os.path.exists(model_logs): 
                    os.makedirs(model_logs, exist_ok=True)

                shutil.move(
                    os.path.join(save_model_temp, file_name), 
                    model_logs
                )
            else: 
                # Scenario D: Unsupported extension type
                gr_warning(translations["unable_analyze_model"])
                return None
        
        gr_info(translations["upload_success"])
        return None
    except Exception as e:
        # Catch unforeseen filesystem errors and inform the user
        gr_error(message=translations["error_occurred"].format(e=e))
        return None
    finally:
        # Always clean up the temporary directory to avoid storage leaks
        shutil.rmtree(save_model_temp, ignore_errors=True)

def zip_file(
    modelname, 
    pth_name, 
    index_path
):
    """
    Bundles specified model weights and index files into a structured downloadable zip.

    Args:
        modelname (str): Name of the model, used for directory mapping and target file name.
        pth_name (str): Name of the model weight file (.pth or .onnx).
        index_path (str): Absolute or relative path to the associated .index file.

    Returns:
        dict: A Gradio UI component configuration dictionary to update the view state, or a UI warning event if verification fails.
    """

    pth_path = os.path.join(configs["weights_path"], pth_name)

    # Validate weight file existence and extension integrity before packing
    if (
        not pth_name or 
        not os.path.exists(pth_path) or 
        not pth_name.endswith((".pth", ".onnx"))
    ): 
        return gr_warning(translations["provide_model"])

    zip_file_path = os.path.join(configs["logs_path"], modelname, modelname + ".zip")
    gr_info(translations["start_zip"])

    # Create the zip archive and map internal structure cleanly (root-level files inside zip)
    with zipfile.ZipFile(zip_file_path, 'w') as zipf:
        zipf.write(pth_path, os.path.basename(pth_path))
        if index_path: zipf.write(index_path, os.path.basename(index_path))

    gr_info(translations["success"])
    # Return structured dict formatted for Gradio dynamic component updates
    return {
        "visible": True, 
        "value": zip_file_path, 
        "__type__": "update"
    }

def fetch_pretrained_data():
    """
    Fetches custom pre-trained data models configuration mapping via a remote API.

    Returns:
        dict: Parsed JSON data from the repository config if successful, otherwise an empty dictionary.
    """

    try:
        # GET request to a HuggingFace configuration JSON file disguised via ROT13
        response = requests.get(
            codecs.decode(
                "uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/wfba/phfgbz_cergenvarq.wfba", 
                "rot13"
            )
        )
        response.raise_for_status()

        return response.json()
    except:
        # Fail silently with safe fallback layout to prevent runtime app crash
        return {}

def update_sample_rate_dropdown(model):
    """
    Updates the target sample rate dropdown selection menu based on the selected model metadata.

    Args:
        model (str): The name or identifier of the selected model.

    Returns:
        dict: A Gradio UI dictionary layout mapping the updated state for the dropdown options.
    """

    # Fetch remote mappings data mapping sample rates to specific models
    data = fetch_pretrained_data()

    # Prevent execution if structural edge-case indicators occur
    if model != translations["success"]: 
        return {
            "choices": list(data[model].keys()), 
            "value": list(data[model].keys())[0], 
            "__type__": "update"
        }
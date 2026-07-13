import os
import re
import sys
import json
import tqdm
import codecs
import tempfile
import requests

from urllib.parse import urlparse, parse_qs, unquote

sys.path.append(os.getcwd())

from main.app.variables import translations

def parse_url(url):
    """
    Parses a Google Drive URL to extract the unique file ID and identify download links.

    Args:
        url (str): The Google Drive or Docs URL to parse.

    Returns:
        Tuple[Optional[str], bool]: A tuple containing:
            - file_id (str or None): The extracted Google Drive file ID, or None
              if not found.
            - is_download_link (bool): True if the URL path ends with '/uc'
              (direct download link).
    """

    parsed = urlparse(url)
    is_download_link = parsed.path.endswith("/uc")

    # Guard clause: Ensure the domain belongs to Google Drive or Docs
    if not parsed.hostname in ("drive.google.com", "docs.google.com"): return None, is_download_link
    # Strategy 1: Attempt to extract ID from query parameters (e.g., ?id=...)
    file_id = parse_qs(parsed.query).get("id", [None])[0]

    # Strategy 2: Match predefined URL patterns within the path via regex if query ID is missing
    if file_id is None:
        for pattern in (
            r"^/file/d/(.*?)/(edit|view)$", 
            r"^/file/u/[0-9]+/d/(.*?)/(edit|view)$", 
            r"^/document/d/(.*?)/(edit|htmlview|view)$", 
            r"^/document/u/[0-9]+/d/(.*?)/(edit|htmlview|view)$", 
            r"^/presentation/d/(.*?)/(edit|htmlview|view)$", 
            r"^/presentation/u/[0-9]+/d/(.*?)/(edit|htmlview|view)$", 
            r"^/spreadsheets/d/(.*?)/(edit|htmlview|view)$", 
            r"^/spreadsheets/u/[0-9]+/d/(.*?)/(edit|htmlview|view)$"
        ):
            match = re.match(pattern, parsed.path)

            if match:
                file_id = match.group(1)
                break

    return file_id, is_download_link

def get_url_from_gdrive_confirmation(contents):
    """
    Extracts the actual download URL from Google Drive's large file virus scan warning page.

    Args:
        contents (str): The HTML source text of the confirmation page.

    Returns:
        str: The resolved, un-obfuscated download URL.

    Raises:
        Exception: If a Google Drive quota error occurs or extraction fails.
    """

    for pattern in (
        r'href="(\/uc\?export=download[^"]+)', 
        r'href="/open\?id=([^"]+)"', 
        r'"downloadUrl":"([^"]+)'
    ):
        match = re.search(pattern, contents)

        if match:
            url = match.group(1)
            # Case 1: URL is embedded within standard open link format
            if pattern == r'href="/open\?id=([^"]+)"': 
                url = (
                    # Obfuscation workaround: decode target base domain via ROT13
                    codecs.decode("uggcf://qevir.hfrepbagrag.tbbtyr.pbz/qbjaybnq?vq=", "rot13") + 
                    url + 
                    "&confirm=t&uuid=" + 
                    re.search(r'<input\s+type="hidden"\s+name="uuid"\s+value="([^"]+)"', contents).group(1)
                )
            elif pattern == r'"downloadUrl":"([^"]+)': # Case 2: URL is embedded inside a JSON response/script block
                url = (
                    url.replace("\\u003d", "=").replace("\\u0026", "&")
                )
            else: # Case 3: URL matches the generic relative export anchor path
                url = (
                    codecs.decode("uggcf://qbpf.tbbtyr.pbz", "rot13") + 
                    url.replace("&", "&")
                )

            return url

    # Check for visible quota/system errors on the page if matching fails
    match = re.search(r'<p class="uc-error-subcaption">(.*)</p>', contents)
    if match: raise Exception(match.group(1))

    raise Exception(translations["gdown_error"])

def _get_session(use_cookies, return_cookies_file=False):
    """
    Initializes a network request session with browser User-Agent and handles cookie persistence.

    Args:
        use_cookies (bool): Whether to look up and load stored session cookies.
        return_cookies_file (bool, optional): If True, returns a tuple including
          the cookies file path. Defaults to False.

    Returns:
        Union[requests.Session, Tuple[requests.Session, str]]: The configured
        Session object, or a tuple of (Session, file_path).
    """

    sess = requests.session()
    sess.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6)"})
    # Resolve platform-independent cache directory for session cookies
    cookies_file = os.path.join(os.path.expanduser("~"), ".cache/gdown/cookies.json")

    # Load previously exported cookies if authorized
    if os.path.exists(cookies_file) and use_cookies:
        with open(cookies_file) as f:
            for k, v in json.load(f):
                sess.cookies[k] = v

    return (sess, cookies_file) if return_cookies_file else sess

def gdown_download(url=None, output=None):
    """
    Downloads a file from Google Drive using public download protocols, bypassing large file warnings.

    Args:
        url (str, optional): The target Google Drive URL or file link. Defaults to None.
        output (str, optional): Path or directory where the final file should be written. Defaults to None (current folder).

    Returns:
        Optional[str]: The absolute or relative string path to the successfully downloaded file, or None if parsing fails.

    Raises:
        ValueError: If url argument is missing.
        Exception: For continuous network download failures.
    """

    file_id = None
    if url is None: raise ValueError(translations["gdown_value_error"])

    # Step 1: Pre-process URL structures to isolate raw ID tokens
    if "/file/d/" in url: 
        file_id = url.split("/d/")[1].split("/")[0]
    elif "open?id=" in url: 
        file_id = url.split("open?id=")[1].split("/")[0]
    elif "/download?id=" in url: 
        file_id = url.split("/download?id=")[1].split("&")[0]

    if file_id:
        # Reconstruct standard direct stream endpoint
        url = f"{codecs.decode('uggcf://qevir.tbbtyr.pbz/hp?vq=', 'rot13')}{file_id}"
        url_origin = url

        sess, cookies_file = _get_session(use_cookies=True, return_cookies_file=True)
        gdrive_file_id, is_gdrive_download_link = parse_url(url)

        if gdrive_file_id:
            url = f"{codecs.decode('uggcf://qevir.tbbtyr.pbz/hp?vq=', 'rot13')}{gdrive_file_id}"
            url_origin = url
            is_gdrive_download_link = True

        # Step 2: Handle redirection and large file landing confirmations
        while 1:
            res = sess.get(url, stream=True, verify=True)
            # Retry with alternate viewer schema if the direct backend hits a 500 error
            if url == url_origin and res.status_code == 500:
                url = f"{codecs.decode('uggcf://qevir.tbbtyr.pbz/bcra?vq=', 'rot13')}{gdrive_file_id}"
                continue

            # Cache the successful session cookies locally (excluding temporary warning tokens)
            os.makedirs(os.path.dirname(cookies_file), exist_ok=True)
            with open(cookies_file, "w") as f:
                json.dump(
                    [(k, v) for k, v in sess.cookies.items() if not k.startswith("download_warning_")], 
                    f, 
                    indent=2
                )

            # If content header indicates an incoming file stream, proceed to download stage
            if ("Content-Disposition" in res.headers) or (not (gdrive_file_id and is_gdrive_download_link)): break

            # If intercepted by confirmation page, extract the hidden download link and loop again
            try:
                url = get_url_from_gdrive_confirmation(res.text)
            except Exception as e:
                raise Exception(e)

        # Step 3: Determine file name from server context or fallback
        if gdrive_file_id and is_gdrive_download_link:
            content_disposition = unquote(res.headers["Content-Disposition"])

            filename_from_url = (
                re.search(r"filename\*=UTF-8''(.*)", content_disposition) or re.search(r'filename=["\']?(.*?)["\']?$', content_disposition)
            ).group(1).replace(os.path.sep, "_")
        else: 
            filename_from_url = os.path.basename(url)

        # Step 4: Prepare localized output destinations and partial cache files
        output = os.path.join(output or ".", filename_from_url)
        tmp_file = tempfile.mktemp(suffix=tempfile.template, prefix=os.path.basename(output), dir=os.path.dirname(output))
        f = open(tmp_file, "ab")

        # Step 5: Execute Range HTTP requests if a partial download signature already exists
        if tmp_file is not None and f.tell() != 0: 
            res = sess.get(
                url, 
                headers={
                    "Range": f"bytes={f.tell()}-"
                }, 
                stream=True, 
                verify=True
            )

        # Step 6: Stream chunk iterations directly into the disk file using a tqdm progress bar
        try:
            with tqdm.tqdm(
                desc=os.path.basename(output), 
                total=int(res.headers.get("Content-Length", 0)), 
                ncols=100, 
                unit="byte"
            ) as pbar:
                for chunk in res.iter_content(chunk_size=512 * 1024):
                    f.write(chunk)
                    pbar.update(len(chunk))

                pbar.close()
                if tmp_file: f.close()
        finally:
            # Atomic swap operation: replace or finalize target file destination
            os.rename(tmp_file, output)
            sess.close()

        return output

    return None
import os
import re
import sys
import six
import json
import tqdm
import shutil
import tempfile
import requests
import warnings
import textwrap

from time import sleep, time
from urllib.parse import urlparse, parse_qs, unquote


CHUNK_SIZE = 512 * 1024 
HOME = os.path.expanduser("~")


def indent(text, prefix):
    return "".join((prefix + line if line.strip() else line) for line in text.splitlines(True))


def parse_url(url, warning=True):
    parsed = urlparse(url)
    is_download_link = parsed.path.endswith("/uc")

    if not parsed.hostname in ("drive.google.com", "docs.google.com"): return None, is_download_link

    file_id = parse_qs(parsed.query).get("id", [None])[0]

    if file_id is None:
        for pattern in (r"^/file/d/(.*?)/(edit|view)$", r"^/file/u/[0-9]+/d/(.*?)/(edit|view)$", r"^/document/d/(.*?)/(edit|htmlview|view)$", r"^/document/u/[0-9]+/d/(.*?)/(edit|htmlview|view)$", r"^/presentation/d/(.*?)/(edit|htmlview|view)$", r"^/presentation/u/[0-9]+/d/(.*?)/(edit|htmlview|view)$", r"^/spreadsheets/d/(.*?)/(edit|htmlview|view)$", r"^/spreadsheets/u/[0-9]+/d/(.*?)/(edit|htmlview|view)$"):
            match = re.match(pattern, parsed.path)

            if match:
                file_id = match.group(1)
                break

    if warning and not is_download_link:
        warnings.warn(
            "Bạn đã chỉ định một liên kết Google Drive không phải là liên kết chính xác "
            "để tải xuống một tập tin. Bạn có thể muốn thử tùy chọn `--fuzzy` "
            f"hoặc url sau: https://drive.google.com/uc?id={file_id}"
        )

    return file_id, is_download_link


def get_url_from_gdrive_confirmation(contents):
    for pattern in (r'href="(\/uc\?export=download[^"]+)', r'href="/open\?id=([^"]+)"', r'"downloadUrl":"([^"]+)'):
        match = re.search(pattern, contents)

        if match:
            url = match.group(1)

            if pattern == r'href="/open\?id=([^"]+)"': url = ("https://drive.usercontent.google.com/download?id=" + url + "&confirm=t&uuid=" + re.search(r'<input\s+type="hidden"\s+name="uuid"\s+value="([^"]+)"', contents).group(1))
            elif pattern == r'"downloadUrl":"([^"]+)': url = url.replace("\\u003d", "=").replace("\\u0026", "&")
            else: url = "https://docs.google.com" + url.replace("&", "&")

            return url

    match = re.search(r'<p class="uc-error-subcaption">(.*)</p>', contents)

    if match:
        error = match.group(1)
        raise Exception(error)

    raise Exception(
        "Không thể truy xuất liên kết công khai của tệp. "
        "Bạn có thể cần phải thay đổi quyền thành "
        "'Bất kỳ ai có liên kết' hoặc đã có nhiều quyền truy cập."
    )


def _get_session(proxy, use_cookies, return_cookies_file=False):
    sess = requests.session()
    sess.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6)"})

    if proxy is not None:
        sess.proxies = {"http": proxy, "https": proxy}
        print("Using proxy:", proxy, file=sys.stderr)

    cookies_file = os.path.join(HOME, ".cache/gdown/cookies.json")

    if os.path.exists(cookies_file) and use_cookies:
        with open(cookies_file) as f:
            cookies = json.load(f)

        for k, v in cookies:
            sess.cookies[k] = v

    return (sess, cookies_file) if return_cookies_file else sess


def gdown_download(url=None, output=None, quiet=False, proxy=None, speed=None, use_cookies=True, verify=True, id=None, fuzzy=True, resume=False, format=None):
    if not (id is None) ^ (url is None): raise ValueError("Phải chỉ định url hoặc id")
    if id is not None: url = f"https://drive.google.com/uc?id={id}"

    url_origin = url

    sess, cookies_file = _get_session(proxy=proxy, use_cookies=use_cookies, return_cookies_file=True)

    gdrive_file_id, is_gdrive_download_link = parse_url(url, warning=not fuzzy)

    if fuzzy and gdrive_file_id:
        url = f"https://drive.google.com/uc?id={gdrive_file_id}"
        url_origin = url
        is_gdrive_download_link = True

    while 1:
        res = sess.get(url, stream=True, verify=verify)

        if url == url_origin and res.status_code == 500:
            url = f"https://drive.google.com/open?id={gdrive_file_id}"
            continue

        if res.headers["Content-Type"].startswith("text/html"):
            title = re.search("<title>(.+)</title>", res.text)

            if title:
                title = title.group(1)
                if title.endswith(" - Google Docs"):
                    url = f"https://docs.google.com/document/d/{gdrive_file_id}/export?format={'docx' if format is None else format}"
                    continue
                if title.endswith(" - Google Sheets"):
                    url = f"https://docs.google.com/spreadsheets/d/{gdrive_file_id}/export?format={'xlsx' if format is None else format}"
                    continue
                if title.endswith(" - Google Slides"):
                    url = f"https://docs.google.com/presentation/d/{gdrive_file_id}/export?format={'pptx' if format is None else format}"
                    continue
        elif ("Content-Disposition" in res.headers and res.headers["Content-Disposition"].endswith("pptx") and format not in (None, "pptx")):
            url = f"https://docs.google.com/presentation/d/{gdrive_file_id}/export?format={'pptx' if format is None else format}"
            continue

        if use_cookies:
            os.makedirs(os.path.dirname(cookies_file), exist_ok=True)

            with open(cookies_file, "w") as f:
                cookies = [(k, v) for k, v in sess.cookies.items() if not k.startswith("download_warning_")]
                json.dump(cookies, f, indent=2)

        if "Content-Disposition" in res.headers: break
        if not (gdrive_file_id and is_gdrive_download_link): break


        try:
            url = get_url_from_gdrive_confirmation(res.text)
        except Exception as e:
            message = (
                "Không thể truy xuất url tệp:\n\n"
                "{}\n\n"
                "Bạn vẫn có thể truy cập tệp từ trình duyệt:"
                f"\n\n\t{url_origin}\n\n"
                "nhưng Gdown không thể. Vui lòng kiểm tra kết nối và quyền."
            ).format(indent("\n".join(textwrap.wrap(str(e))), prefix="\t"))
            raise Exception(message)

    if gdrive_file_id and is_gdrive_download_link:
        content_disposition = unquote(res.headers["Content-Disposition"])
        filename_from_url = (re.search(r"filename\*=UTF-8''(.*)", content_disposition) or re.search(r'filename=["\']?(.*?)["\']?$', content_disposition)).group(1)
        filename_from_url = filename_from_url.replace(os.path.sep, "_")
    else: filename_from_url = os.path.basename(url)

    output = output or filename_from_url
    output_is_path = isinstance(output, six.string_types)

    if output_is_path and output.endswith(os.path.sep):
        os.makedirs(output, exist_ok=True)
        output = os.path.join(output, filename_from_url)

    if output_is_path:
        temp_dir = os.path.dirname(output) or "."
        prefix = os.path.basename(output)
        existing_tmp_files = [os.path.join(temp_dir, file) for file in os.listdir(temp_dir) if file.startswith(prefix)]

        if resume and existing_tmp_files:
            if len(existing_tmp_files) > 1:
                print("Có nhiều tệp tạm thời để tiếp tục:", file=sys.stderr)

                for file in existing_tmp_files:
                    print(f"\t{file}", file=sys.stderr)

                print("Vui lòng xóa chúng ngoại trừ một để tiếp tục tải xuống.", file=sys.stderr)
                return
            
            tmp_file = existing_tmp_files[0]
        else:
            resume = False
            tmp_file = tempfile.mktemp(suffix=tempfile.template, prefix=prefix, dir=temp_dir)

        f = open(tmp_file, "ab")
    else:
        tmp_file = None
        f = output

    if tmp_file is not None and f.tell() != 0: res = sess.get(url, headers={"Range": f"bytes={f.tell()}-"}, stream=True, verify=verify)

    if not quiet:
        if resume: print("Tiếp tục:", tmp_file, file=sys.stderr)

        print("Đến:", os.path.abspath(output) if output_is_path else output, file=sys.stderr)

    try:
        if not quiet: pbar = tqdm.tqdm(total=int(res.headers.get("Content-Length", 0)), unit="B", unit_scale=True)

        t_start = time()

        for chunk in res.iter_content(chunk_size=CHUNK_SIZE):
            f.write(chunk)

            if not quiet: pbar.update(len(chunk))

            if speed is not None:
                elapsed_time_expected = 1.0 * pbar.n / speed
                elapsed_time = time() - t_start

                if elapsed_time < elapsed_time_expected: sleep(elapsed_time_expected - elapsed_time)

        if not quiet: pbar.close()


        if tmp_file:
            f.close()
            shutil.copy(tmp_file, output)
    finally:
        sess.close()

    return output
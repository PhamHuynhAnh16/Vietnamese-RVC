import os
import sys
import requests

from bs4 import BeautifulSoup


def Mediafire_Download(url, output=None, filename=None):
    if not(filename): filename = url.split('/')[-2]
    
    sess = requests.session()
    sess.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6)"})

    url = BeautifulSoup(sess.get(url).content, "html.parser").find(id="downloadButton").get("href")
    
    if not(output): output = os.path.dirname(os.path.realpath(__file__))


    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(f"{output}/{filename}", "wb") as f:
                total_length = r.headers.get('content-length')
                total_length = int(total_length)

                download_progress = 0

                for chunk in r.iter_content(chunk_size=1024):
                    download_progress += len(chunk)

                    f.write(chunk)

                    sys.stdout.write(f"\r[Đang tải xuống {filename}] Tiến triển: {int(100 * download_progress/total_length)}% ({round(download_progress/1024/1024, 2)}mb/{round(total_length/1024/1024, 2)}mb)")
                    sys.stdout.flush()

        sys.stdout.write("\n")
        return f"{output}/{filename}"
    except Exception as e:
        print(e)
        return e
import time
import threading

from googleapiclient.discovery import build


class Clean:
    def __init__(self, every=300):
        self.service = build('drive', 'v3')
        self.every = every  
        self.trash_cleanup_thread = None

    def delete(self):
        page_token = None

        while 1:
            response = self.service.files().list(q="trashed=true", spaces='drive', fields="nextPageToken, files(id, name)", pageToken=page_token).execute()
            
            for file in response.get('files', []):
                if file['name'].startswith("G_") or file['name'].startswith("D_"):
                    try:
                        self.service.files().delete(fileId=file['id']).execute()
                    except Exception as e:
                        raise RuntimeError(e)

            page_token = response.get('nextPageToken', None)
            if page_token is None: break

    def clean(self):
        while 1:
            self.delete()
            time.sleep(self.every)

    def start(self):
        self.trash_cleanup_thread = threading.Thread(target=self.clean)
        self.trash_cleanup_thread.daemon = True 
        self.trash_cleanup_thread.start()

    def stop(self):
        if self.trash_cleanup_thread: self.trash_cleanup_thread.join()
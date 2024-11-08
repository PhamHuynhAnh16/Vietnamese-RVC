import time
import threading
import subprocess

from typing import List, Union
from googleapiclient.discovery import build


class Channel:
    def __init__(self, source, destination, sync_deletions=False, every=60, exclude: Union[str, List, None] = None):
        self.source = source
        self.destination = destination
        self.event = threading.Event()
        self.syncing_thread = threading.Thread(target=self._sync, args=())
        self.sync_deletions = sync_deletions
        self.every = every

        if not exclude: exclude = []
        if isinstance(exclude,str): exclude = [exclude]

        self.exclude = exclude
        self.command = ['rsync', '-aP']

    def alive(self):
        if self.syncing_thread.is_alive(): return True
        else: return False

    def _sync(self):
        command = self.command

        for exclusion in self.exclude:
            command.append(f'--exclude={exclusion}')

        command.extend([f'{self.source}/', f'{self.destination}/'])

        if self.sync_deletions: command.append('--delete')

        while not self.event.is_set():
            subprocess.run(command)
            time.sleep(self.every)

    def copy(self):
        command = self.command

        for exclusion in self.exclude:
            command.append(f'--exclude={exclusion}')

        command.extend([f'{self.source}/', f'{self.destination}/'])

        if self.sync_deletions: command.append('--delete')
        subprocess.run(command)
        return True
    
    def start(self):
        if self.syncing_thread.is_alive():
            self.event.set()
            self.syncing_thread.join()

        if self.event.is_set(): self.event.clear()
        if self.syncing_thread._started.is_set(): self.syncing_thread = threading.Thread(target=self._sync, args=())

        self.syncing_thread.start()
        
        return self.alive()

    def stop(self):
        if self.alive():
            self.event.set()
            self.syncing_thread.join()

            while self.alive():
                if not self.alive(): break

        return not self.alive()

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
                        raise RuntimeError(f"Lỗi khi xóa {file['name']}: {e}")

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
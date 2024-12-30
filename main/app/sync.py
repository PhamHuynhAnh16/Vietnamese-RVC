import time
import threading
import subprocess

class Channel:
    def __init__(self, source, destination, sync_deletions=False, every=60, exclude = None):
        self.source = source
        self.destination = destination
        self.event = threading.Event()
        self.syncing_thread = threading.Thread(target=self._sync, args=())
        self.sync_deletions = sync_deletions
        self.every = every

        if not exclude: exclude = []
        if isinstance(exclude, str): exclude = [exclude]

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
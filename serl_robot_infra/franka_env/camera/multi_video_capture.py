import queue
import threading
import time
import numpy as np
from collections import OrderedDict

class MultiVideoCapture:
    def __init__(self, caps):
        self.caps = caps
        self.queue = queue.Queue()
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = False
        self.enable = True
        self.t.start()

    def _reader(self):
        while self.enable:
            frames = OrderedDict()
            for name, cap in self.caps.items():
                ret, frame = cap.read()
                if ret:
                    frames[name] = frame
            
            if frames:
                if not self.queue.empty():
                    try:
                        self.queue.get_nowait()  # discard previous (unprocessed) frame
                    except queue.Empty:
                        pass
                self.queue.put(frames)

    def read(self):
        return self.queue.get(timeout=5)

    def close(self):
        self.enable = False
        self.t.join()
        for cap in self.caps.values():
            cap.close()

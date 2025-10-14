import atexit
import functools
from queue import Queue
from threading import Event, Thread

from paddleocr import PaddleOCR
from PIL import Image


LANG_CONFIG = {
    "en": {"num_workers": 1},
}
CONCURRENCY_LIMIT = 8


class PaddleOCRModelManager(object):
    def __init__(self,
                 num_workers,
                 model_factory):
        super().__init__()
        self._model_factory = model_factory
        self._queue = Queue()
        self._workers = []
        self._model_initialized_event = Event()
        for _ in range(num_workers):
            worker = Thread(target=self._worker, daemon=False)
            worker.start()
            self._model_initialized_event.wait()
            self._model_initialized_event.clear()
            self._workers.append(worker)

    def predict(self, *args, **kwargs):
        # XXX: Should I use a more lightweight data structure, say, a future?
        result_queue = Queue(maxsize=1)
        self._queue.put((args, kwargs, result_queue))
        success, payload = result_queue.get()
        if success:
            return payload
        else:
            raise payload

    def close(self):
        for _ in self._workers:
            self._queue.put(None)
        for worker in self._workers:
            worker.join()

    def _worker(self):
        model = self._model_factory()
        self._model_initialized_event.set()
        while True:
            item = self._queue.get()
            if item is None:
                break
            args, kwargs, result_queue = item
            try:
                result = model.ocr(*args, **kwargs)
                result_queue.put((True, result))
            except Exception as e:
                result_queue.put((False, e))
            finally:
                self._queue.task_done()


def create_model(lang):
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="gpu")

    return ocr

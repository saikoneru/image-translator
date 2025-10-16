from abc import ABC, abstractmethod
from typing import List, Dict
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
from PIL import Image
import functools
import io
from queue import Queue
from threading import Thread, Event
import numpy as np

# ----------------------------
# Abstract OCR Worker Interface
# ----------------------------
class BaseOCRWorker(ABC):
    @abstractmethod
    def predict(self, image_path: str) -> List[Dict]:
        """
        Run OCR on the given image path.

        Returns:
            List[Dict] with keys:
            - "text": recognized string
            - "box": 4-point polygon [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        pass

# ----------------------------
# PaddleOCR Manager (threaded)
# ----------------------------
class PaddleOCRModelManager:
    def __init__(self, num_workers, model_factory):
        self._model_factory = model_factory
        self._queue = Queue()
        self._workers = []
        self._model_initialized_event = Event()
        for _ in range(num_workers):
            worker = Thread(target=self._worker, daemon=True)
            worker.start()
            self._model_initialized_event.wait()
            self._model_initialized_event.clear()
            self._workers.append(worker)

    def predict(self, *args, **kwargs):
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
                result = model.predict(*args, **kwargs)
                result_queue.put((True, result))
            except Exception as e:
                result_queue.put((False, e))
            finally:
                self._queue.task_done()


# ----------------------------
# PaddleOCR Worker Implementation
# ----------------------------
class PaddleOCRWorker(BaseOCRWorker):
    def __init__(self, num_workers=1, lang="en"):
        self.ocr_manager = PaddleOCRModelManager(
            num_workers=num_workers,
            model_factory=functools.partial(self._create_model, lang=lang)
        )

    def _create_model(self, lang: str):
        ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            device="gpu",
            lang = "en",
        )
        #ocr = PaddleOCR(use_angle_cls=True, lang='en', ocr_version='PP-OCRv4')

        return ocr

    def predict(self, image_path: str) -> List[Dict]:
        raw_results = self.ocr_manager.predict(image_path)[0]
        output = []


        for poly, text, conf in zip(raw_results["rec_polys"], raw_results["rec_texts"], raw_results["rec_scores"]):
            if conf > 0.8:
                output.append({"text": text, "box": poly})

        return output

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(title="PaddleOCR Worker")

ocr_worker = PaddleOCRWorker(num_workers=1, lang="en")

@app.post("/ocr")
async def ocr_endpoint(file: UploadFile):
    try:
        img_bytes = await file.read()
        # Save temporarily to in-memory BytesIO object for PaddleOCR
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        # PaddleOCR requires a path, so save to temporary in-memory file
        tmp_path = "/tmp/tmp_image.png"
        img.save(tmp_path)

        results = ocr_worker.predict(tmp_path)
        for r in results:
            if isinstance(r.get("box"), np.ndarray):
                r["box"] = r["box"].tolist()
        return JSONResponse(content={"results": results})
    except Exception as e:
        print("Error during OCR processing:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ----------------------------
# Run server
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False, log_level="debug")

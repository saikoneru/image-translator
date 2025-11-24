from abc import ABC, abstractmethod
from typing import Dict, Any, List
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
from PIL import Image
import functools
import io
from queue import Queue
from threading import Thread, Event
import numpy as np
import os
import uvicorn
import cv2

# =====================================================
# Base Worker Interface (Unified)
# =====================================================
class BaseWorker(ABC):
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input and return standardized output.

        Args:
            input_data: Dictionary with worker-specific input data.

        Returns:
            Dictionary with standardized output format.
        """
        pass

    @abstractmethod
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input format"""
        pass

    @abstractmethod
    def validate_output(self, output_data: Dict[str, Any]) -> bool:
        """Validate output format"""
        pass


# =====================================================
# PaddleOCR Model Manager (Threaded)
# =====================================================
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
                result = model.ocr(*args, **kwargs)
                result_queue.put((True, result))
            except Exception as e:
                result_queue.put((False, e))
            finally:
                self._queue.task_done()


# =====================================================
# PaddleOCR Worker (Implements BaseWorker)
# =====================================================
class PaddleOCRWorker(BaseWorker):
    def __init__(self, num_workers: int = 1, lang: str = "en"):
        self.ocr_manager = PaddleOCRModelManager(
            num_workers=num_workers,
            model_factory=functools.partial(self._create_model, lang=lang)
        )

    def _create_model(self, lang: str):
        return PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            device="gpu",
            lang=lang,
        )

    # ----------------------
    # BaseWorker Interface
    # ----------------------
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run OCR and return standardized results"""
        if not self.validate_input(input_data):
            raise ValueError("Invalid input format for OCRWorker")

        image_path = input_data["image_path"]
        raw_results = self.ocr_manager.predict(image_path)[0]

        results = []
        for poly, text, conf in zip(
            raw_results["rec_polys"], raw_results["rec_texts"], raw_results["rec_scores"]
        ):
            if conf > 0.8:
                results.append({"text": text, "box": poly})

        output = {"results": results}
        if not self.validate_output(output):
            raise ValueError("Invalid output format from OCRWorker")

        return output

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Check for required keys and valid image path"""
        return isinstance(input_data, dict) and "image_path" in input_data and os.path.exists(input_data["image_path"])

    def validate_output(self, output_data: Dict[str, Any]) -> bool:
        """Check output contains 'results' as a list of dicts"""
        if not isinstance(output_data, dict):
            return False
        results = output_data.get("results", [])
        return isinstance(results, list) and all(
            isinstance(r, dict) and "text" in r and "box" in r for r in results
        )


# =====================================================
# FastAPI App
# =====================================================
app = FastAPI(title="PaddleOCR Worker")

ocr_worker = PaddleOCRWorker(num_workers=1, lang="en")


@app.post("/ocr")
async def ocr_endpoint(file: UploadFile):
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        tmp_path = "/tmp/tmp_image.png"
        img.save(tmp_path)

        input_data = {"image_path": tmp_path}
        output_data = ocr_worker.process(input_data)

        # Convert numpy arrays to lists
        for r in output_data["results"]:
            if isinstance(r.get("box"), np.ndarray):
                r["box"] = r["box"].tolist()

        return JSONResponse(content=output_data)

    except Exception as e:
        print("Error during OCR processing:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)


# =====================================================
# Run Server
# =====================================================
if __name__ == "__main__":
    host = os.getenv("WORKER_HOST", "0.0.0.0")
    port = int(os.getenv("WORKER_PORT", "8001"))
    log_level = os.getenv("LOG_LEVEL", "debug")

    uvicorn.run(app, host=host, port=port, reload=False, log_level=log_level)

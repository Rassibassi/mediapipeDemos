import cv2
import numpy as np
from PIL import Image


class VideoSource:
    def __init__(self, display=False, dtype=np.uint8):
        self._name = "VideoSource"
        self._capture = None
        self._display = display
        self._dtype = dtype

    def get_fps(self):
        return self._capture.get(cv2.CAP_PROP_FPS)

    def get_frame_count(self):
        return int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_image_size(self):
        width = self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return width, height

    def release(self):
        self._capture.release()
        cv2.destroyAllWindows()

    def __iter__(self):
        self._capture.isOpened()
        return self

    def __next__(self):
        ret, frame = self._capture.read()

        if self._display:
            cv2.imshow(f"{self._name} - FPS: {self.get_fps()}", frame)

        if not ret:
            raise StopIteration

        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise StopIteration

        cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = np.asarray(Image.fromarray(cv2_im_rgb), dtype=self._dtype)

        return frame, frame_rgb

    def __del__(self):
        self.release()

    def show(self, frame):
        cv2.imshow(f"{self._name} - FPS: {self.get_fps()}", frame)


class WebcamSource(VideoSource):
    def __init__(
        self,
        camera_id=0,
        width=1280,
        height=720,
        fps=30,
        autofocus=0,
        absolute_focus=75,
        display=False,
    ):
        super().__init__(display)
        self._capture = cv2.VideoCapture(camera_id)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._capture.set(cv2.CAP_PROP_FPS, fps)
        self._capture.set(cv2.CAP_PROP_AUTOFOCUS, autofocus)
        self._capture.set(cv2.CAP_PROP_FOCUS, absolute_focus / 255)


class FileSource(VideoSource):
    def __init__(self, file_path, display=False):
        super().__init__(display)
        self._capture = cv2.VideoCapture(str(file_path))

import cv2 as cv
import numpy as np
import pytest

class DummyCapture:
    def __init__(self, total_frames=5, frame_width=640, frame_height=480):
        self.frame_idx = 0
        self.total_frames = total_frames
        self.frame_width = frame_width
        self.frame_height = frame_height

    def get(self, prop):
        if prop == cv.CAP_PROP_FRAME_COUNT:
            return self.total_frames
        if prop == cv.CAP_PROP_FRAME_WIDTH:
            return self.frame_width
        if prop == cv.CAP_PROP_FRAME_HEIGHT:
            return self.frame_height

    def read(self):
        if self.frame_idx < self.total_frames:
            self.frame_idx += 1
            return True, np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            return False, np.zeros((480, 640, 3), dtype=np.uint8)

    def release(self):
        return

class DummyCaptureWithImage:
    def __init__(self, total_frames=5, frame_width=640, frame_height=480):
        self.frame_idx = 0
        self.total_frames = total_frames
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.sample_frame = self.generate_test_image()


    def get(self, prop):
        if prop == cv.CAP_PROP_FRAME_COUNT:
            return self.total_frames
        if prop == cv.CAP_PROP_FRAME_WIDTH:
            return self.frame_width
        if prop == cv.CAP_PROP_FRAME_HEIGHT:
            return self.frame_height

    def read(self):
        if self.frame_idx < self.total_frames:
            self.frame_idx += 1
            frame = cv.resize(self.sample_frame, (self.frame_width, self.frame_height))
            return True, frame
        else:
            return False, cv.resize(self.sample_frame, (self.frame_width, self.frame_height))

    def release(self):
        return

    def generate_test_image(self, width=640, height=480):
        # Start with random speckle pattern
        np.random.seed(42)
        img = np.random.randint(0, 256, (height, width), dtype=np.uint8)

        # Overlay some deterministic shapes (to keep it repeatable for tests)
        cv.circle(img, (width//4, height//4), 50, 255, -1)
        cv.rectangle(img, (width//2, height//2), (width-50, height-50), 255, 5)
        cv.line(img, (0, 0), (width, height), 255, 3)
        cv.putText(img, 'ORB', (50, height-50), cv.FONT_HERSHEY_SIMPLEX, 2, 255, 3)

        return cv.merge([img, img, img])


@pytest.fixture
def dummy_video_capture():
    return DummyCapture()

@pytest.fixture
def dummy_video_capture_image_proxy():
    return DummyCaptureWithImage()

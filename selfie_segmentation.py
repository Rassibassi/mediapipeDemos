import cv2
import mediapipe as mp
import numpy as np

from videosource import WebcamSource

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation


def main():
    source = WebcamSource()

    with mp_selfie_segmentation.SelfieSegmentation(
        model_selection=1
    ) as selfie_segmentation:

        for idx, (frame, frame_rgb) in enumerate(source):

            results = selfie_segmentation.process(frame_rgb)

            # Draw selfie segmentation on the background image.
            # To improve segmentation around boundaries, consider applying a joint
            # bilateral filter to "results.segmentation_mask" with "image".
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1

            bg_image = None

            # The background can be customized.
            #   a) Load an image (with the same width and height of the input image) to
            #      be the background, e.g., bg_image = cv2.imread('/path/to/image/file')
            #   b) Blur the input image by applying image filtering, e.g.,
            #      bg_image = cv2.GaussianBlur(image,(55,55),0)

            bg_image = cv2.GaussianBlur(frame, (55, 55), 0)

            if bg_image is None:
                bg_image = np.zeros(frame.shape, dtype=np.uint8)

            output_image = np.where(condition, frame, bg_image)

            source.show(output_image)


if __name__ == "__main__":
    main()

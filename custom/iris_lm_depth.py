import cv2
import numpy as np

from custom.core import (  # isort:skip
    detections_to_rect,
    landmarks_to_detections,
    slice_from_roi,
    tflite_inference,
    transform_rect,
)


def from_landmarks_to_depth(
    frame_rgb, eye_landmarks, image_size, is_right_eye=False, focal_length=None
):
    if focal_length is None:
        focal_length = frame_rgb.shape[1]
    detections = landmarks_to_detections(eye_landmarks)
    rect = detections_to_rect(detections, image_size, rotation_vector_start_end=(0, 1))
    roi = transform_rect(rect, image_size, scale_x=2.3, scale_y=2.3)

    slice_y = slice_from_roi(roi, image_size, False)
    slice_x = slice_from_roi(roi, image_size, True)
    eye_image = frame_rgb[slice(*slice_y), slice(*slice_x), :]
    position_in_frame = np.array((slice_x[0], slice_y[0], 0))

    eye_contours, iris_landmarks = detect_iris(
        eye_image.copy(), is_right_eye=is_right_eye
    )

    eye_contours[:, 0] = eye_contours[:, 0] * eye_image.shape[0]
    eye_contours[:, 1] = eye_contours[:, 1] * eye_image.shape[1]
    eye_contours = eye_contours + position_in_frame

    eye_contours[:, 0] = eye_contours[:, 0] / frame_rgb.shape[1]
    eye_contours[:, 1] = eye_contours[:, 1] / frame_rgb.shape[0]

    iris_landmarks[:, 0] = iris_landmarks[:, 0] * eye_image.shape[0]
    iris_landmarks[:, 1] = iris_landmarks[:, 1] * eye_image.shape[1]
    iris_landmarks = iris_landmarks + position_in_frame

    iris_landmarks[:, 0] = iris_landmarks[:, 0] / frame_rgb.shape[1]
    iris_landmarks[:, 1] = iris_landmarks[:, 1] / frame_rgb.shape[0]

    depth, iris_size = calculate_iris_depth(iris_landmarks, image_size, focal_length)

    return depth, iris_size, iris_landmarks, eye_contours


def detect_iris(eye_frame, is_right_eye=False):
    side_low = 64
    eye_frame_low = cv2.resize(
        eye_frame, (side_low, side_low), interpolation=cv2.INTER_AREA
    )

    model_path = "models/iris_landmark.tflite"

    if is_right_eye:
        eye_frame_low = np.fliplr(eye_frame_low)

    outputs = tflite_inference(eye_frame_low / 127.5 - 1.0, model_path)
    eye_contours_low = np.reshape(outputs[0], (71, 3))
    iris_landmarks_low = np.reshape(outputs[1], (5, 3))

    eye_contours = eye_contours_low / side_low
    iris_landmarks = iris_landmarks_low / side_low

    if is_right_eye:
        eye_contours[:, 0] = 1 - eye_contours[:, 0]
        iris_landmarks[:, 0] = 1 - iris_landmarks[:, 0]

    return eye_contours, iris_landmarks


def calculate_iris_depth(iris_landmarks, image_size, focal_length_pixel):
    """
    iris_landmarks should be normalized to the complete image frame
    depth in mm
    """
    iris_size = calculate_iris_diameter(iris_landmarks, image_size)
    depth = calculate_depth(
        iris_landmarks[0, :], focal_length_pixel, iris_size, image_size
    )

    return depth, iris_size


def get_depth(x0, y0, x1, y1):
    return np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)


def get_landmark_depth(ld0, ld1, image_size):
    return get_depth(
        ld0[0] * image_size[0],
        ld0[1] * image_size[1],
        ld1[0] * image_size[0],
        ld1[1] * image_size[1],
    )


def calculate_iris_diameter(iris_landmarks, image_size):
    dist_vert = get_landmark_depth(
        iris_landmarks[1, :], iris_landmarks[3, :], image_size
    )
    dist_hori = get_landmark_depth(
        iris_landmarks[2, :], iris_landmarks[4, :], image_size
    )

    return (dist_hori + dist_vert) / 2.0


def calculate_depth(center_landmark, focal_length_pixel, iris_size, image_size):
    # Average fixed iris size across human beings.
    human_iris_size_in_mm = 11.8
    origin = np.array(image_size) / 2.0
    center_landmark_pixel = center_landmark[:2] * np.array(image_size)
    y = get_depth(
        origin[0], origin[1], center_landmark_pixel[0], center_landmark_pixel[1]
    )
    x = np.sqrt(focal_length_pixel ** 2 + y ** 2)
    depth = human_iris_size_in_mm * x / iris_size

    return depth

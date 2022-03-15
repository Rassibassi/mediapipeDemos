import numpy as np
import tensorflow as tf


def tflite_inference(inputs, model_path, dtype=np.float32):

    if not isinstance(inputs, (list, tuple)):
        inputs = (inputs,)

    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    for inp, inp_det in zip(inputs, input_details):
        interpreter.set_tensor(inp_det["index"], np.array(inp[None, ...], dtype=dtype))

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    outputs = [interpreter.get_tensor(out["index"]) for out in output_details]

    return outputs


def landmarks_to_detections(landmarks):
    """
    landmarks: (3, N) landmarks
    """
    x_min = np.amin(landmarks[0, :])
    x_max = np.amax(landmarks[0, :])
    y_min = np.amin(landmarks[1, :])
    y_max = np.amax(landmarks[1, :])

    bbox = dict()
    bbox["x_min"] = x_min
    bbox["y_min"] = y_min
    bbox["width"] = x_max - x_min
    bbox["height"] = y_max - y_min

    detections = dict()
    detections["bboxs"] = bbox
    detections["keypoints"] = landmarks[:2, :]

    return detections


def detections_to_rect(
    detections,
    image_size,
    rotation_vector_start_end=None,
    rotation_vector_target_angle=0,
):

    keypoints = detections["keypoints"]
    x_min = np.amin(keypoints[0, :])
    x_max = np.amax(keypoints[0, :])
    y_min = np.amin(keypoints[1, :])
    y_max = np.amax(keypoints[1, :])

    rect = dict()
    rect["x_center"] = (x_min + x_max) / 2
    rect["y_center"] = (y_min + y_max) / 2
    rect["width"] = x_max - x_min
    rect["height"] = y_max - y_min

    if rotation_vector_start_end is not None:
        rect["rotation"] = compute_rotation(
            detections,
            image_size,
            rotation_vector_start_end,
            rotation_vector_target_angle,
        )
    else:
        rect["rotation"] = None

    return rect


def compute_rotation(detections, image_size, rotation_vector_start_end, target_angle):

    keypoints = detections["keypoints"]

    x0 = keypoints[0, rotation_vector_start_end[0]] * image_size[0]
    y0 = keypoints[1, rotation_vector_start_end[0]] * image_size[1]
    x1 = keypoints[0, rotation_vector_start_end[1]] * image_size[0]
    y1 = keypoints[1, rotation_vector_start_end[1]] * image_size[1]

    rotation = normalize_radians(target_angle - np.arctan2(-(y1 - y0), x1 - x0))

    return rotation


def normalize_radians(angle):
    return angle - 2 * np.pi * np.floor((angle - (-np.pi)) / (2 * np.pi))


def transform_rect(
    rect,
    image_size,
    scale_x=1,
    scale_y=1,
    shift_x=0,
    shift_y=0,
    square_long=True,
    square_short=False,
    opt_rotation=None,
):
    width = rect["width"]
    height = rect["height"]
    rotation = rect["rotation"]
    image_width = image_size[0]
    image_height = image_size[1]

    if rotation is not None and opt_rotation is not None:
        rotation += opt_rotation
        rotation = normalize_radians(rotation)

    if rotation is None:
        rect["x_center"] = rect["x_center"] + width * shift_x
        rect["y_center"] = rect["y_center"] + height * shift_y
    else:
        x_shift = (
            image_width * width * shift_x * np.cos(rotation)
            - image_height * height * shift_y * np.sin(rotation)
        ) / image_width
        y_shift = (
            image_width * width * shift_x * np.sin(rotation)
            + image_height * height * shift_y * np.cos(rotation)
        ) / image_height

        rect["x_center"] = rect["x_center"] + x_shift
        rect["y_center"] = rect["y_center"] + y_shift

    if square_long:
        long_side = np.max((width * image_width, height * image_height))
        width = long_side / image_width
        height = long_side / image_height
    elif square_short:
        short_side = np.min((width * image_width, height * image_height))
        width = short_side / image_width
        height = short_side / image_height

    rect["width"] = width * scale_x
    rect["height"] = height * scale_y

    return rect


def slice_from_roi(roi, image_size, horizontal_side=True):
    if horizontal_side:
        center = roi["x_center"]
        norm_side = roi["width"]
        image_side = image_size[0]
    else:
        center = roi["y_center"]
        norm_side = roi["height"]
        image_side = image_size[1]

    first_id = int((center - norm_side / 2) * image_side)
    second_id = int((center + norm_side / 2) * image_side)

    return (first_id, second_id)


def extract_faces(raw_frame, results, x_scale=1.0, y_scale=1.0):
    frames = []
    if results.detections is None:
        return frames
    for detection in results.detections:
        image_size = raw_frame.shape[1::-1]
        x_min = detection.location_data.relative_bounding_box.xmin
        y_min = detection.location_data.relative_bounding_box.ymin
        width = detection.location_data.relative_bounding_box.width
        height = detection.location_data.relative_bounding_box.height

        x_min = image_size[0] * x_min
        y_min = image_size[1] * y_min
        width = image_size[0] * width
        height = image_size[1] * height
        x_max = x_min + width
        y_max = y_min + height

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        width = x_scale * width
        height = y_scale * height

        x_min = x_center - width / 2
        y_min = y_center - height / 2

        x_max = x_min + width
        y_max = y_min + height

        x_min, x_max, y_min, y_max = map(int, [x_min, x_max, y_min, y_max])

        frame = raw_frame[y_min:y_max, x_min:x_max]

        if frame.any():
            frames.append(frame)

    return frames

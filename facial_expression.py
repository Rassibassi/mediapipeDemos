import cv2
import mediapipe as mp
import numpy as np

from custom.core import extract_faces, tflite_inference
from videosource import WebcamSource

mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection

labels = ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger"]

fast_model = True
slow_model_every_x = 5


def main():
    source = WebcamSource()

    mean = np.array([0.57535914, 0.44928582, 0.40079932])
    std = np.array([0.20735591, 0.18981615, 0.18132027])

    if fast_model:
        # from https://github.com/zengqunzhao/EfficientFace
        model_path = "models/efficient_face_model.tflite"
    else:
        # from https://github.com/zengqunzhao/EfficientFace
        model_path = "models/dlg_model.tflite"

    with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    ) as face_detection:

        for idx, (frame, frame_rgb) in enumerate(source):

            results = face_detection.process(frame_rgb)

            face_frames = extract_faces(frame_rgb, results, x_scale=1.2, y_scale=1.2)

            if face_frames:
                face_frame = cv2.resize(face_frames[0], (224, 224))
                frame[0:224, 0:224, :] = cv2.cvtColor(face_frame, cv2.COLOR_RGB2BGR)

                if fast_model or (idx % slow_model_every_x == 0):
                    face_frame = face_frame / 255
                    face_frame -= mean
                    face_frame /= std
                    face_frame = np.moveaxis(face_frame, -1, 0)

                    outputs = tflite_inference(face_frame, model_path)
                    outputs = outputs[0]
                    expression_id = np.argmax(outputs)

                    # write expression over head
                    detection = results.detections[0]
                    relative_keypoints = detection.location_data.relative_keypoints
                    landmarks = np.stack([(rk.x, rk.y) for rk in relative_keypoints])

                    image_size = frame_rgb.shape[1::-1]
                    pos = landmarks[2, :]
                    pos = image_size * pos
                    text_size = cv2.getTextSize(
                        labels[expression_id], cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2
                    )
                    text_size = text_size[0]
                    pos[0] -= text_size[0] / 2
                    pos[1] -= 150

                    pos = tuple(pos.astype(np.int32).tolist())
                    cv2.putText(
                        frame,
                        labels[expression_id],
                        pos,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (57, 255, 20),
                        2,
                    )

            source.show(frame)


if __name__ == "__main__":
    main()

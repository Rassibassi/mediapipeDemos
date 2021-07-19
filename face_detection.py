import mediapipe as mp

from videosource import WebcamSource

mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection


def main():
    source = WebcamSource()

    with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    ) as face_detection:

        for idx, (frame, frame_rgb) in enumerate(source):

            results = face_detection.process(frame_rgb)

            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(frame, detection)

            source.show(frame)


if __name__ == "__main__":
    main()

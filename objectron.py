import mediapipe as mp

from videosource import WebcamSource

mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron


def main():
    source = WebcamSource()

    with mp_objectron.Objectron(
        static_image_mode=False,
        max_num_objects=5,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.99,
        model_name="Shoe",  # {'Shoe', 'Chair', 'Cup', 'Camera'}
    ) as objectron:

        for idx, (frame, frame_rgb) in enumerate(source):

            results = objectron.process(frame_rgb)

            if results.detected_objects:
                for detected_object in results.detected_objects:
                    mp_drawing.draw_landmarks(
                        frame,
                        detected_object.landmarks_2d,
                        mp_objectron.BOX_CONNECTIONS,
                    )
                    mp_drawing.draw_axis(
                        frame, detected_object.rotation, detected_object.translation
                    )

            source.show(frame)


if __name__ == "__main__":
    main()

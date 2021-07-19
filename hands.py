import mediapipe as mp

from videosource import WebcamSource

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=3)


def main():
    source = WebcamSource()

    with mp_hands.Hands(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as hands:

        for idx, (frame, frame_rgb) in enumerate(source):

            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec,
                    )

            source.show(frame)


if __name__ == "__main__":
    main()

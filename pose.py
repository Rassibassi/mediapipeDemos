import mediapipe as mp
import numpy as np

from webcam import WebcamSource

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=3)


def main():
    source = WebcamSource()

    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:

        for idx, (frame, frame_rgb) in enumerate(source):

            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec,
                )

                # get landmarks as numpy
                landmarks = results.pose_landmarks.landmark
                np_landmarks = np.array(
                    [(lm.x, lm.y, lm.z, lm.visibility) for lm in landmarks]
                )
                print(np_landmarks.shape)

            source.show(frame)


if __name__ == "__main__":
    main()

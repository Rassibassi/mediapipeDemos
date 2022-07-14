import mediapipe as mp

from videosource import WebcamSource

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_face_mesh_connections = mp.solutions.face_mesh_connections
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=3)


def main():
    source = WebcamSource()

    refine_landmarks = False

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        for idx, (frame, frame_rgb) in enumerate(source):

            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh_connections.FACEMESH_TESSELATION,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec,
                    )

            source.show(frame)


if __name__ == "__main__":
    main()

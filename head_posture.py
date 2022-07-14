import cv2
import mediapipe as mp
import numpy as np

from videosource import WebcamSource

from custom.face_geometry import (  # isort:skip
    PCF,
    get_metric_landmarks,
    procrustes_landmark_basis,
)

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_face_mesh_connections = mp.solutions.face_mesh_connections
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=3)

points_idx = [33, 263, 61, 291, 199]
points_idx = points_idx + [key for (key, val) in procrustes_landmark_basis]
points_idx = list(set(points_idx))
points_idx.sort()

# uncomment next line to use all points for PnP algorithm
# points_idx = list(range(0,468)); points_idx[0:2] = points_idx[0:2:-1];

frame_height, frame_width, channels = (720, 1280, 3)

# pseudo camera internals
focal_length = frame_width
center = (frame_width / 2, frame_height / 2)
camera_matrix = np.array(
    [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
    dtype="double",
)

dist_coeff = np.zeros((4, 1))


def main():
    source = WebcamSource()

    refine_landmarks = True

    pcf = PCF(
        near=1,
        far=10000,
        frame_height=frame_height,
        frame_width=frame_width,
        fy=camera_matrix[1, 1],
    )

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        for idx, (frame, frame_rgb) in enumerate(source):
            results = face_mesh.process(frame_rgb)
            multi_face_landmarks = results.multi_face_landmarks

            if multi_face_landmarks:
                face_landmarks = multi_face_landmarks[0]
                landmarks = np.array(
                    [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
                )
                # print(landmarks.shape)
                landmarks = landmarks.T

                if refine_landmarks:
                    landmarks = landmarks[:, :468]

                metric_landmarks, pose_transform_mat = get_metric_landmarks(
                    landmarks.copy(), pcf
                )

                image_points = (
                    landmarks[0:2, points_idx].T
                    * np.array([frame_width, frame_height])[None, :]
                )
                model_points = metric_landmarks[0:3, points_idx].T

                # see here:
                # https://github.com/google/mediapipe/issues/1379#issuecomment-752534379
                pose_transform_mat[1:3, :] = -pose_transform_mat[1:3, :]
                mp_rotation_vector, _ = cv2.Rodrigues(pose_transform_mat[:3, :3])
                mp_translation_vector = pose_transform_mat[:3, 3, None]

                if False:
                    # sanity check
                    # get same result with solvePnP

                    success, rotation_vector, translation_vector = cv2.solvePnP(
                        model_points,
                        image_points,
                        camera_matrix,
                        dist_coeff,
                        flags=cv2.cv2.SOLVEPNP_ITERATIVE,
                    )

                    np.testing.assert_almost_equal(mp_rotation_vector, rotation_vector)
                    np.testing.assert_almost_equal(
                        mp_translation_vector, translation_vector
                    )

                for face_landmarks in multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh_connections.FACEMESH_TESSELATION,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec,
                    )

                nose_tip = model_points[0]
                nose_tip_extended = 2.5 * model_points[0]
                (nose_pointer2D, jacobian) = cv2.projectPoints(
                    np.array([nose_tip, nose_tip_extended]),
                    mp_rotation_vector,
                    mp_translation_vector,
                    camera_matrix,
                    dist_coeff,
                )

                nose_tip_2D, nose_tip_2D_extended = nose_pointer2D.squeeze().astype(int)
                frame = cv2.line(
                    frame, nose_tip_2D, nose_tip_2D_extended, (255, 0, 0), 2
                )

            source.show(frame)


if __name__ == "__main__":
    main()

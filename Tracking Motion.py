import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    with mp_face.FaceMesh(max_num_faces=4, refine_landmarks=True, min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as face:
        while cap.isOpened():
            ret, frame = cap.read()

            # BGR 2 RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Flip on horizontal
            image = cv2.flip(image, 1)

            # Set flag
            image.flags.writeable = False

            # Detections
            results_hand = hands.process(image)
            results_face = face.process(image)

            # Set flag to true
            image.flags.writeable = True

            # RGB 2 BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Rendering results
            if results_hand.multi_hand_landmarks:
                for num, hand in enumerate(results_hand.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                              mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2,
                                                                     circle_radius=2),
                                              )
            if results_face.multi_face_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
                    mp_drawing.draw_landmarks(image, face_landmarks,
                                              mp_face.FACEMESH_TESSELATION, None,
                                              mp_drawing_styles.DrawingSpec(color=(121, 22, 76), thickness=1,
                                                                            circle_radius=1),
                                              )
                    mp_drawing.draw_landmarks(image, face_landmarks,
                                              mp_face.FACEMESH_LIPS, None,
                                              mp_drawing_styles.DrawingSpec(color=(250, 350, 0), thickness=1,
                                                                            circle_radius=1)
                                              )
                    mp_drawing.draw_landmarks(image, face_landmarks,
                                              mp_face.FACEMESH_CONTOURS, None,
                                              mp_drawing_styles.get_default_face_mesh_contours_style()
                                              )
                    mp_drawing.draw_landmarks(image, face_landmarks,
                                              mp_face.FACEMESH_IRISES, None,
                                              mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                                              )

            cv2.imshow('Tracking', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()

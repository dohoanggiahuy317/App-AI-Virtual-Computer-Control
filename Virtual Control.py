import mediapipe as mp
import cv2
from cvzone.HandTrackingModule import HandDetector
from pynput.mouse import Button, Controller
import autopy
import osascript

# MEDIAPIPE UTILS
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# TAKING INPUT IMAGE FOR HAND AND FACE
detector = HandDetector(detectionCon=0.5, maxHands=2)
mp_face = mp.solutions.face_mesh
cap = cv2.VideoCapture(0)

# MOUSE & CONTROLLING & CURRENT VOLUME
mymouse = Controller()
pressing_bool = False
code, curr_vol, err = osascript.run("output volume of (get volume settings)")
curr_vol = int(curr_vol)
# print(curr_vol)


# METHODS FOR CONTROLLING MOUSE
def mouse_click(button):
    autopy.mouse.click(button, None)


def mouse_move(wrist_pos_x, wrist_pos_y):
    mouse_pos_x = wrist_pos_x * 3.69 - 2656
    mouse_pos_y = wrist_pos_y * 3.1 - 1085

    if 0 < mouse_pos_x < 1439 and 899 > mouse_pos_y > 0:
        autopy.mouse.move(mouse_pos_x, mouse_pos_y)


def left_hand_gesture(fingers1, landmarks_list):
    global curr_vol
    if fingers1 == [0, 1, 1, 1, 1] or fingers1 == [1, 0, 0, 0, 0]:  # mouse click
        mouse_click(autopy.mouse.Button.LEFT)

    if fingers1 == [1, 0, 1, 1, 1]:  # mouse scroll
        mymouse.scroll(0, -5)

    if fingers1 == [1, 1, 0, 0, 1]:  # Mouse click middle
        mouse_click(autopy.mouse.Button.MIDDLE)

    if fingers1 == [0, 0, 0, 0, 0]:  # increase volume
        curr_vol += 3
        vol = "set volume output volume " + str(curr_vol)
        osascript.osascript(vol)

    mouse_move(landmarks_list[0][0], landmarks_list[0][1])  # mouse move


def right_hand_gesture(fingers2):
    global curr_vol
    if fingers2 == [0, 1, 1, 1, 1]:  # right click
        mouse_click(autopy.mouse.Button.RIGHT)

    if fingers2 == [1, 0, 1, 1, 1]:  # mouse scroll
        mymouse.scroll(0, 5)

    if fingers2 == [0, 0, 0, 0, 0]:  # decrease volume
        curr_vol -= 3
        vol = "set volume output volume " + str(curr_vol)
        osascript.osascript(vol)


with mp_face.FaceMesh(max_num_faces=4, refine_landmarks=True, min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as face:
    while True:
        ret, frame = cap.read()

        # clean up frame input
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)

        # Detections
        results_face = face.process(frame)
        hands, frame = detector.findHands(frame, draw=True)

        # convert to viewable form frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Rendering results face
        face_lanmarks_list = []
        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame, face_landmarks,
                                          mp_face.FACEMESH_TESSELATION, None,
                                          mp_drawing_styles.DrawingSpec(color=(121, 22, 76), thickness=1,
                                                                        circle_radius=1),
                                          )
                mp_drawing.draw_landmarks(frame, face_landmarks,
                                          mp_face.FACEMESH_LIPS, None,
                                          mp_drawing_styles.DrawingSpec(color=(250, 350, 0), thickness=1,
                                                                        circle_radius=1)
                                          )
                mp_drawing.draw_landmarks(frame, face_landmarks,
                                          mp_face.FACEMESH_CONTOURS, None,
                                          mp_drawing_styles.get_default_face_mesh_contours_style()
                                          )
                mp_drawing.draw_landmarks(frame, face_landmarks,
                                          mp_face.FACEMESH_IRISES, None,
                                          mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                                          )
                face_lanmarks_list.append(face_landmarks)
                # lips_distance = results_face.multi_face_landmarks[67].y - results_face.multi_face_landmarks[63].y
                # print(lips_distance)

        # Rendering hands
        if hands:
            hand1 = hands[0]
            lmList1 = hand1["lmList"]  # List of 21 Landmark points
            fingers1 = detector.fingersUp(hand1)

            if hand1["type"] == "Left":
                left_hand_gesture(fingers1, lmList1)
            if hand1["type"] == "Right":
                right_hand_gesture(fingers1)

        if len(hands) == 2:
            hand2 = hands[1]
            lmList2 = hand2["lmList"]
            fingers2 = detector.fingersUp(hand2)

            if hand2["type"] == "Left":
                left_hand_gesture(fingers2, lmList2)
            if hand2["type"] == "Right":
                right_hand_gesture(fingers2)

        # QUIT
        cv2.imshow('Tracking', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

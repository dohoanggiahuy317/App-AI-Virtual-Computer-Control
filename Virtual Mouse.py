import cv2
import autopy
from cvzone.HandTrackingModule import HandDetector
from pynput.mouse import Button, Controller


def mouse_click(button):
    autopy.mouse.click(button, None)


def mouse_move(wrist_pos_x, wrist_pos_y):
    mouse_pos_x = wrist_pos_x * 3.69 - 2656
    mouse_pos_y = wrist_pos_y * 3.1 - 1085

    if 0 < mouse_pos_x < 1439 and 899 > mouse_pos_y > 0:
        autopy.mouse.move(mouse_pos_x, mouse_pos_y)


def left_hand_gesture(fingers1, landmarks_list):
    global pressing_bool
    if fingers1 == [0, 1, 1, 1, 1]:  # mouse click
        mouse_click(autopy.mouse.Button.LEFT)

    if fingers1 == [1, 0, 1, 1, 1]:  # mouse scroll
        mymouse.scroll(0, 4)

    if fingers1 == [1, 0, 0, 0, 0] and pressing_bool == False:
        mymouse.press(Button.left)
        pressing_bool = True

    if fingers1 == [1, 1, 1, 1, 1]:
        mymouse.release(Button.left)
        pressing_bool = False

    mouse_move(landmarks_list[0][0], landmarks_list[0][1])  # mouse move


def right_hand_gesture(fingers2):
    if fingers2 == [0, 1, 1, 1, 1]:
        mouse_click(autopy.mouse.Button.RIGHT)

    if fingers2 == [1, 0, 1, 1, 1]:  # mouse scroll
        mymouse.scroll(0, -4)


# TAKING INPUT IMAGE
detector = HandDetector(detectionCon=0.8, maxHands=2)
cap = cv2.VideoCapture(0)
mymouse = Controller()
pressing_bool = False

while True:
    # Get image frame
    success, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.flip(frame, 1)

    # Find the hand and its landmarks
    hands, frame = detector.findHands(frame, draw=True)

    # turn back to normal color
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
    cv2.imshow("Virtual Mouse", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
from cvzone.HandTrackingModule import HandDetector
import autopy
import osascript
from pynput.mouse import Button, Controller


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

# TAKING INPUT IMAGE
detector = HandDetector(detectionCon=0.8, maxHands=2)
cap = cv2.VideoCapture(0)
mymouse = Controller()
pressing_bool = False
code, curr_vol, err = osascript.run("output volume of (get volume settings)")
curr_vol = int(curr_vol)

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
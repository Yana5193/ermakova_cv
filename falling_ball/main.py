import cv2
import numpy as np
import math
import os

CAMERA_ID = 1 + cv2.CAP_DSHOW
WIDTH = 1280
HEIGHT = 720

cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

if not cap.isOpened():
    print("Camera not found")
    exit()

ret, frame = cap.read()
if not ret:
    print("Cannot read camera")
    exit()

frame = cv2.resize(frame, (WIDTH, HEIGHT))
h, w, _ = frame.shape

ball_radius = 12
center_x = w // 2
center_y = ball_radius + 15

ball_pos = np.array([float(center_x), float(center_y)])
ball_vel = np.array([0.0, 0.0])

gravity = 1.2
bounce = 0.05
slide_friction = 1.0
air_friction = 1.0
max_speed = 35.0

g_vec = np.array([0.0, gravity])

is_rolling = False
is_reset = True

window_game = "Physics Sandbox"
window_debug = "Debug Mask"

cv2.namedWindow(window_game, cv2.WINDOW_NORMAL)
cv2.namedWindow(window_debug, cv2.WINDOW_NORMAL)

cv2.startWindowThread()

def reset_ball():
    global ball_pos, ball_vel, is_rolling, is_reset
    ball_pos = np.array([float(center_x), float(center_y)])
    ball_vel = np.array([0.0, 0.0])
    is_rolling = False
    is_reset = True


def get_platforms(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    physics_mask = np.zeros((h, w), dtype=np.uint8)
    draw_frame = np.zeros((h, w, 3), dtype=np.uint8)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 500:
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)

        if x > w * 0.8 and y > h * 0.7:
            continue

        if cw < 40 and ch < 40:
            continue

        if area > h * w * 0.25:
            continue

        cv2.drawContours(physics_mask, [cnt], -1, 255, -1)
        cv2.drawContours(draw_frame, [cnt], -1, (0, 255, 0), 3)

    return physics_mask, draw_frame, thresh


while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (w, h))
    frame = cv2.flip(frame, 0)

    physics_mask, display_img, debug_mask = get_platforms(frame)

    if is_rolling:
        ball_vel += g_vec
        ball_vel *= air_friction

        speed = np.linalg.norm(ball_vel)

        if speed > max_speed:
            ball_vel = (ball_vel / speed) * max_speed

        next_pos = ball_pos + ball_vel

        if (
            next_pos[0] < -ball_radius or
            next_pos[0] > w + ball_radius or
            next_pos[1] < -ball_radius or
            next_pos[1] > h - ball_radius
        ):
            reset_ball()
            is_rolling = True
            is_reset = False
            ball_vel = np.array([np.random.uniform(-0.5, 0.5), 0.0])
            continue

        px = int(round(next_pos[0]))
        py = int(round(next_pos[1]))

        if 0 <= px < w and 0 <= py < h and physics_mask[py, px] == 255:
            r = ball_radius + 5

            x_min = max(0, px - r)
            x_max = min(w, px + r)
            y_min = max(0, py - r)
            y_max = min(h, py + r)

            roi = physics_mask[y_min:y_max, x_min:x_max]
            M = cv2.moments(roi)

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"]) + x_min
                cy = int(M["m01"] / M["m00"]) + y_min

                normal = ball_pos - np.array([cx, cy])
                norm_val = np.linalg.norm(normal)

                if norm_val > 0:
                    normal = normal / norm_val
                    dot = np.dot(ball_vel, normal)

                    if dot < 0:
                        ball_vel = ball_vel - dot * normal

                    ball_vel *= slide_friction

                    tangent = np.array([-normal[1], normal[0]])

                    if tangent[1] < 0:
                        tangent = -tangent

                    ball_vel += tangent * (gravity * 0.45)

            next_pos = ball_pos + ball_vel

        ball_pos = next_pos

    bx = int(round(ball_pos[0]))
    by = int(round(ball_pos[1]))

    if is_reset:
        cv2.circle(display_img, (center_x, center_y), ball_radius, (0, 0, 255), 2)
        cv2.circle(display_img, (center_x, center_y), 2, (0, 0, 255), -1)
    else:
        cv2.circle(display_img, (bx, by), ball_radius, (0, 0, 255), -1)
        cv2.circle(display_img, (bx - 4, by - 4), 3, (255, 255, 255), -1)
        cv2.circle(display_img, (bx, by), ball_radius, (0, 0, 100), 2)

    cv2.putText(
        display_img,
        "SPACE: Drop Ball | R: Reset | ESC: Exit",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.imshow(window_game, display_img)
    cv2.imshow(window_debug, debug_mask)

    cv2.pollKey()

    key = cv2.waitKey(10) & 0xFF

    if key == 27:
        break
    elif key == 32:
        if is_reset:
            is_rolling = True
            is_reset = False
            ball_vel = np.array([np.random.uniform(-0.5, 0.5), 0.0])
    elif key == ord("r") or key == ord("R"):
        reset_ball()

cap.release()
cv2.destroyAllWindows()

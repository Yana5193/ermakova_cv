import cv2
import numpy as np
from math import dist
import time
from pathlib import Path
import json
import random

save_path=Path(__file__).parent
config_path=save_path/"config.json"
target_sequence = ["red", "green", "yellow"]
random.shuffle(target_sequence)
print(f"Компьютер загадал: {target_sequence}")
cv2.namedWindow("Image", cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow("Mask", cv2.WINDOW_GUI_NORMAL)

position = (0,0)
clicked = False
def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at ({x}, {y})")
        global position
        global clicked
        position = (x, y)
        clicked = True

cv2.setMouseCallback("Image", on_click)
cam = cv2.VideoCapture(0)
lower=None
upper=None
if config_path.exists():
    with config_path.open("r") as f:
        js=json.load(f)
        if "lower" in js:
            lower=np.array(js["lower"],dtype="u1")
            upper=np.array(js["upper"],dtype="u1")
positions=[]
prev_time=time.time()
curr_time=time.time()
d=6.36 
saved_colors = {}
while cam.isOpened():
    ret,frame=cam.read()
    blurred=cv2.GaussianBlur(frame,(11,11),0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break
    if clicked:
        clicked = False
        color = hsv[position [1]][position [0]]
        lower = np.clip(color * 0.9, 0, 255).astype("u1")
        upper = np.clip(color * 1.1, 0, 255).astype("u1")
    
    if lower is not None:
        if key == ord('r'):
            saved_colors['red'] = (lower.copy(), upper.copy())
        if key == ord('g'):
            saved_colors['green'] = (lower.copy(), upper.copy())
        if key == ord('y'):
            saved_colors['yellow'] = (lower.copy(), upper.copy())
    found_this= []
    for name in saved_colors:
        l_lim, u_lim = saved_colors[name]
        mask = cv2.inRange(hsv, l_lim, u_lim)
        if lower is not None and np.array_equal(l_lim, lower):
            cv2.imshow("Mask", mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(cnt) > 100:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                found_this.append({"name": name, "x": int(x)})
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 4)
                cv2.putText(frame, name, (int(x), int(y)-10), 1, 1.5, (255,255,255), 2)
    print(found_this)
    found_this.sort(key=lambda item: item["x"])
    current_names = [item["name"] for item in found_this]
    if current_names == target_sequence:
        cv2.putText(frame, "WIN!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 10)  
    cv2.imshow("Image", frame)
cam.release()
cv2.destroyAllWindows()

with config_path.open("w") as f:
    json.dump(
        {"lower":None if lower is None else lower.tolist(),
         "upper":None if upper is None else lower.tolist()

        },
        f
    )
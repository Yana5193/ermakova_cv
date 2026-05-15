import mss
import cv2
import pyautogui
import numpy as np
import time
pyautogui.PAUSE = 0
screen = {"top": 260, "left": 545, "width": 770, "height": 207}
with mss.mss() as camera:
    print("Бот запущен")
    while True:
        img = camera.grab(screen)
        array = np.array(img)

        x1, x2 = 140, 210
        y1, y2 = 115, 145
        by1, by2 = 70, 110

        gray = cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)

        cactus_zone = gray[y1:y2, x1:x2]
        bird_zone = gray[by1:by2, x1:x2]

        cactus = np.ptp(cactus_zone) > 120
        bird = np.ptp(bird_zone) > 120

        if bird:
            pyautogui.keyDown('down')
            time.sleep(0.3)
            pyautogui.keyUp('down')
        elif cactus:
            pyautogui.keyDown('space')
            time.sleep(0.1) 
            pyautogui.keyUp('space')
            time.sleep(0.2)
        #cv2.rectangle(array, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #cv2.rectangle(array, (x1, by1), (x2, by2), (255, 0, 0), 2)
        #cv2.imshow("DinoBot", array)

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
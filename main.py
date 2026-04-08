import cv2
import numpy as np
import threading
import keyboard
import ctypes
import win32gui, win32ui, win32con, win32api
from ultralytics import YOLO
import math

def fire_action(dx, dy, trigger):
    mode = 0x0007 if trigger else 0x0001
    ctypes.windll.user32.mouse_event(mode, int(dx), int(dy), 0, 0)

class TwoK_WideSystem:
    def __init__(self, model_path="best.pt"):
        self.model = YOLO(model_path)
        self.running = False
        self.auto_fire = False
        self.win_name = "YOLO_VIEW"
        self.speed = 1.0
        self.precise_zone = 30
        self.conf = 0.55
        self.screen_w = 2560
        self.screen_h = 1440
        self.size = 1280
        self.left = (self.screen_w - self.size) // 2
        self.top = (self.screen_h - self.size) // 2
        self.rel_center = self.size // 2

    def grab_screen(self):
        hwin = win32gui.GetDesktopWindow()
        hwindc = win32gui.GetWindowDC(hwin)
        srcdc = win32ui.CreateDCFromHandle(hwindc)
        memdc = srcdc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(srcdc, self.size, self.size)
        memdc.SelectObject(bmp)
        memdc.BitBlt((0, 0), (self.size, self.size), srcdc, (self.left, self.top), win32con.SRCCOPY)
        bits = bmp.GetBitmapBits(True)
        img = np.frombuffer(bits, dtype='uint8').reshape(self.size, self.size, 4)
        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(hwin, hwindc)
        win32gui.DeleteObject(bmp.GetHandle())
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def run(self):
        if self.running: return
        self.running = True

        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win_name, 300, 300)
        cv2.moveWindow(self.win_name, 0, 0)

        while self.running:
            hwnd = win32gui.FindWindow(None, self.win_name)
            if hwnd: win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, 3)

            frame = self.grab_screen()
            res = self.model.predict(frame, verbose=False, half=True, imgsz=640, device=0)

            draw_frame = frame.copy()
            cv2.circle(draw_frame, (self.rel_center, self.rel_center), self.precise_zone, (255, 255, 255), 1)

            boxes = res[0].boxes
            target = None
            candidates = []

            if len(boxes) > 0:
                for b in boxes:
                    if b.conf > self.conf:
                        xy = b.xyxy[0].tolist()
                        cx, cy = (xy[0] + xy[2]) / 2, (xy[1] + xy[3]) / 2
                        candidates.append((cx, cy))
                        cv2.rectangle(draw_frame, (int(xy[0]), int(xy[1])), (int(xy[2]), int(xy[3])), (0, 255, 0), 2)

                if candidates:
                    candidates.sort(key=lambda c: (c[0], -c[1]), reverse=True)
                    target = candidates[0]

            if target:
                dx = (target[0] - self.rel_center) * self.speed
                dy = (target[1] - self.rel_center) * self.speed
                dist = math.dist((target[0], target[1]), (self.rel_center, self.rel_center))

                should_fire = self.auto_fire and (dist < self.precise_zone)
                cv2.line(draw_frame, (self.rel_center, self.rel_center), (int(target[0]), int(target[1])), (0, 0, 255), 2)

                if abs(dx) > 0.5 or abs(dy) > 0.5:
                    fire_action(dx, dy, should_fire)

            cv2.imshow(self.win_name, cv2.resize(draw_frame, (300, 300)))
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cv2.destroyAllWindows()
        self.running = False

if __name__ == "__main__":
    bot = TwoK_WideSystem(model_path="best.pt")
    keyboard.add_hotkey("F1", lambda: threading.Thread(target=bot.run, daemon=True).start())
    keyboard.add_hotkey("F2", lambda: setattr(bot, 'running', False))
    keyboard.add_hotkey("F3", lambda: setattr(bot, 'auto_fire', not bot.auto_fire))
    print("F1:啟動 | F2:停止 | F3:開火開關")
    keyboard.wait()
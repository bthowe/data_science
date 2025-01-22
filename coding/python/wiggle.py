import sys
import time
import joblib
import datetime
import pyautogui
import threading
import numpy as np

screenSize = pyautogui.size()


class Wiggle():
    def __init__(self, first_today=False):
        self.total_break_time, self.total_work_time = self._initialize_total_time(first_today)

    def _initialize_total_time(self, first_today):
        if first_today:
            joblib.dump(0, 'wiggle/break_time.pkl')
            joblib.dump(1, 'wiggle/work_time.pkl')
            return 0, 1
        else:
            return joblib.load('wiggle/break_time.pkl'), joblib.load('wiggle/work_time.pkl')

    def _time_writer(self):
        joblib.dump(self.total_break_time, 'wiggle/break_time.pkl')

        joblib.dump(self.total_work_time, 'wiggle/work_time.pkl')
        print(f'Total Break time: {joblib.load("wiggle/break_time.pkl") / 60} minutes')
        print(f'Total Work time: {joblib.load("wiggle/work_time.pkl") / 60} minutes')

    def _water_break(self):
        session_break_time = 150 + np.random.uniform(0, 120)

        print(f"This session's break time: {session_break_time / 60} minutes")

        self.total_break_time += session_break_time
        self.total_work_time += 300
        self._time_writer()

        threading.Timer(300 + session_break_time, self._moveMouse).start()

    def _grindstone(self):
        session_work_time = 300 - np.random.uniform(0, 120)

        print(f"This session's work time: {session_work_time / 60} minutes")

        self.total_work_time += session_work_time
        self._time_writer()

        threading.Timer(session_work_time, self._moveMouse).start()

    def _moveMouse(self):
        x, y = pyautogui.position()

        pyautogui.moveTo(x + 1, y + 1, duration=1)
        # pyautogui.moveTo(np.random.randint(0, screenSize[0]), np.random.randint(0, screenSize[1]), duration=1)
        self.decide_what_to_do()

    def _clickMouse(self):
        pyautogui.click()

        self.decide_what_to_do()

    def _time_check(self, hour, minute):
        if hour == 16 and minute > 30:
            print("end of day reached")

            quit()
        else:
            pass

    def _application_tab(self, app_num):
        pyautogui.keyDown('command')

        for _ in range(app_num + 1):
            pyautogui.press('tab')
        pyautogui.keyUp('command')

        time.sleep(3)
        pyautogui.keyDown('command')
        pyautogui.press('tab')
        pyautogui.keyUp('command')

    def decide_what_to_do(self, app_toggle=False):
        self._time_check(datetime.datetime.now().hour, datetime.datetime.now().minute)

        p = np.random.uniform(0, 1)
        if p < .25:
            print(
                f'\nWhew...needing this break! Fraction of time working: {self.total_work_time / (self.total_break_time + self.total_work_time)}')
            self._water_break()
        else:
            if app_toggle:
                app_num = np.random.randint(low=1, high=6, size=2)
            for _ in range(app_num[0] + 1):
                self._application_tab(app_num[1])
            print(
                f'\nStill working away...Fraction of time working: {self.total_work_time / (self.total_break_time + self.total_work_time)}')
            self._grindstone()


if __name__ == '__main__':
    Wiggle(first_today=True).decide_what_to_do(app_toggle=False)

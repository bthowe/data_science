import sys
import numpy as np
import pandas as pd
from scipy.stats import expon
from scipy.stats import uniform


# 9
# np.random.seed(40)

class Clinic(object):
    def __init__(self, opening_time, closing_time, num_drs, mean_wait):
        self.total_time = (closing_time - opening_time) * 60
        self.num_drs = num_drs
        self.l = mean_wait

        self.time_of_day = 0
        self.num_patients = 0
        self.patients = {}
        self.dr_busy_until = {num: 0 for num in range(self.num_drs)}

    def day_go(self):
        while True:
            next_enter_time = self._patient_enters_clinic()
            if next_enter_time != 'close clinic':
                self._patient_sign_in(next_enter_time)
                self._patient_see_dr()
            else:
                break

    def _patient_enters_clinic(self):
        next_enter_time = expon.rvs(scale=self.l)
        if self.time_of_day + next_enter_time <= self.total_time:
            return next_enter_time
        else:
            return 'close clinic'

    def _patient_sign_in(self, next_enter_time):
        self.time_of_day += next_enter_time
        self.num_patients += 1
        self.patients[self.num_patients] = {'enter_time': self.time_of_day}

    def _patient_see_dr(self):
        next_available_dr = min(self.dr_busy_until, key=self.dr_busy_until.get)
        next_available_time = min(self.dr_busy_until.values())

        self.patients[self.num_patients]['seen_time'] = max(next_available_time, self.time_of_day)

        visit_length = uniform.rvs(5, 20)
        self.patients[self.num_patients]['exit_time'] = self.time_of_day + visit_length
        self.dr_busy_until[next_available_dr] = self.time_of_day + visit_length

def wait_time_summary(patients_dict):
    wait_times = []
    for patient, dict in patients_dict.items():
        wait_times.append(dict['seen_time'] - dict['enter_time'])
    num_waiters = np.where(np.array(wait_times) > 0, 1, 0).sum()
    average_wait = np.array(wait_times)[np.where(np.array(wait_times) > 0)].mean()
    return num_waiters, average_wait

def close_time_format(close_time):
    hr_min = divmod(close_time, 60)
    hr = int((9 + hr_min[0]) % 12)
    minutes = int(round(hr_min[1]))
    if minutes < 10:
        minutes = '0' + str(minutes)
    return '{0}:{1} p.m.'.format(hr, minutes)

if __name__ == '__main__':
    sim_num = 100
    sum_data = []
    for iteration in range(sim_num):
        c = Clinic(9, 16, 3, 10)
        c.day_go()
        patient_info = c.patients

        num_patients = len(patient_info)
        wait = wait_time_summary(patient_info)
        close_time = max(d['exit_time'] for d in patient_info.values())
        sum_data.append([num_patients, wait[0], wait[1], close_time])

    df = pd.DataFrame(sum_data, columns=['n_patients', 'n_wait', 'ave_wait_time', 'close_time'])
    df.loc[df['ave_wait_time'] != df['ave_wait_time'], 'ave_wait_time'] = 0

    df_med = df.median()
    print(df.median())
    print(df.quantile(.75) - df.quantile(.25))

    print('The median number of patients who visited the office: {}'.format(int(df_med['n_patients'])))
    print('The median number of patients who had to wait to be seen: {}'.format(int(df_med['n_wait'])))
    print('Of these, their median average wait time: {} minutes'.format(round(df_med['ave_wait_time'], 2)))
    print('The median office close time was {}'.format(close_time_format(df_med['close_time'])))



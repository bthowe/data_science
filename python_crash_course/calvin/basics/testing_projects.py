


class Employee:

    def __init__(self, first, last, salary):
        self.first = first
        self.last = last
        self.salary = salary
        self.extra = 50

    def give_raise(self):
        self.new_income = (self.salary + self.extra)
        if self.extra == 0:
            print(f"{self.first} {self.last}, annual salary ${self.new_income}")
        else:
            print(f"{self.first} {self.last}, annual salary ${self.new_income}, just got a pay raise of ${self.extra}.")

me = Employee('Calvin', 'Howe', 200)
me.extra = 200
me.give_raise()

import unittest

class TestEmployee(unittest.TestCase):

    def test_give_default_raise(self):
        you = Employee('Samuel', 'Howe', 0)
        you.give_raise()

    def test_custom_give_raise(self):
        you = Employee('Samuel', 'Howe', 0)
        you.give_raise()



if __name__ == '__main__':
    unittest.main()


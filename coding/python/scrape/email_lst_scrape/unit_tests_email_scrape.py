import sys
import unittest

a = []
with open("ward_email_lst.txt", "r") as filestream:
    for line in filestream:
        a += line.split(',')

class TestEmailLists(unittest.TestCase):

    def test_duplicates(self):
        print(set([x for x in a if a.count(x) > 1]))
        print(len(a))
        print(len(set(a)))
        self.assertEqual(len(a), len(set(a)))

    # def test_isupper(self):
    #     self.assertTrue('FOO'.isupper())
    #     self.assertFalse('Foo'.isupper())
    #
    # def test_split(self):
    #     s = 'hello world'
    #     self.assertEqual(s.split(), ['hello', 'world'])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)

if __name__ == '__main__':
    unittest.main()

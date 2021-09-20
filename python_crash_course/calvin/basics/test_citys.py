import unittest
from functions import country_city


class CityTest(unittest.TestCase):

    def test_city_country(self):
        new_place = country_city('Athens', 'Greece', population=7999999)
        self.assertEqual(new_place)

    def test_city_country(self):
        populated_place = country_city('Crete', 'Greece', population=70000000)
        self.assertEqual(populated_place)



if __name__ == '__main__':
    unittest.main()
import unittest

from chinatravel.environment.language import canonical_poi_name


class EnvironmentLanguageTest(unittest.TestCase):
    def test_canonicalizes_known_english_poi_translation(self):
        self.assertEqual(canonical_poi_name("Sola Bistro", "en"), "Bistro Sola")
        self.assertEqual(canonical_poi_name("Sola Bistro", "zh"), "Sola Bistro")


if __name__ == "__main__":
    unittest.main()

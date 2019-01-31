from unittest import TestCase
import run_models

class TestJoke(TestCase):
    def test_is_string(self):
        s = run_models.joke()
        self.assertTrue(isinstance(s, str))

    def test_is_string2(self):
        s = run_models.joke() + 'waaaaaaat'
        self.assertTrue(isinstance(s, str))
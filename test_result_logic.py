import unittest

from result_logic import calculate_result


class TestStudentResult(unittest.TestCase):

    def test_pass(self):
        result = calculate_result(60, 70, 80)
        self.assertEqual(result, "Pass")

    def test_needs_support(self):
        result = calculate_result(20, 30, 35)
        self.assertEqual(result, "Needs Support")

    def test_exact_threshold(self):
        result = calculate_result(40, 40, 40)
        self.assertEqual(result, "Pass")


if __name__ == "__main__":
    unittest.main()

import unittest
import pkg_resources as res


class MyTestCase(unittest.TestCase):
    def test_something(self):
        fname = res.resource_filename(test.data, )
        print(fname)


if __name__ == '__main__':
    unittest.main()

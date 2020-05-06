import unittest
import core.ForwardAutoDiffOneD as fwd_ad1d
import math


class TestOneD(unittest.TestCase):
    def test_X(self):
        x = fwd_ad1d.X()
        self.assertEqual(x.value_at(4), 4)

    def test_polyterm(self):
        poly = fwd_ad1d.PolynomialTerm(3, 2)
        x = poly.value_at(2)
        self.assertEqual(x, 12)
        d = poly.derivative_at(5)
        self.assertEqual(d, 30)

        poly = fwd_ad1d.PolynomialTerm(3, 1. / 3)
        x = poly(27)
        self.assertEqual(9, x)
        d = poly.derivative_at(27)
        self.assertAlmostEqual(d, 1.0/9)

    def test_composite(self):
        sin = fwd_ad1d.Sine()
        x2 = fwd_ad1d.PolynomialTerm(1, 2)
        sinx2 = fwd_ad1d.SimpleComposite(sin, x2)
        x = math.pi/4
        got = sinx2(x)
        x_exp = math.sin(x*x)
        self.assertAlmostEqual(got, x_exp)

        got = sinx2.derivative_at(x)
        x_exp = 2*x*math.cos(x*x)
        self.assertAlmostEqual(got, x_exp)

    def test_sum(self):
        f = fwd_ad1d.PolynomialTerm(3, 4)
        g = fwd_ad1d.PolynomialTerm(2, 2)
        f_plus_g = fwd_ad1d.Sum([f, g])
        self.do_print(f_plus_g, "Sum")

    def test_simple_symbolics(self):
        f = fwd_ad1d.Sine()
        self.do_print(f, "Sine")

    def test_symbolic_poly(self):
        f = fwd_ad1d.PolynomialTerm(3, 2)
        self.do_print(f, "Polynomial")

    def test_simple_composite(self):
        sin = fwd_ad1d.Sine()
        x2 = fwd_ad1d.PolynomialTerm(1, 2)
        sinx2 = fwd_ad1d.SimpleComposite(sin, x2)
        self.do_print(sinx2, "Simple composite")

    def do_print(self, f, msg=""):
        print("{}:\t f(x)={}, f(x)'={}".format(msg, f.symbolic_value() , f.symbolic_deriv()))


if __name__ == '__main__':
    unittest.main()

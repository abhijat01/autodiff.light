import math
from . import debug


class Func:
    def value_at(self, x):
        raise Exception("Not implemented")

    def derivative_at(self, x):
        raise Exception("Not implemented")

    def __call__(self, x):
        return self.value_at(x)

    def symbolic_deriv(self, var_name='x'):
        raise Exception("Not implemented")

    def symbolic_value(self, var_name="x"):
        raise Exception("Not implemented")

    def derivative(self):
        raise Exception("Not implemented. This should return another "
                        "Func type representing the derivative of this function.")

    def dim(self):
        return 1


class Reciprocal(Func):

    def __init__(self, f):
        self.f = f

    def value_at(self, x):
        return 1 / self.f.value_at(x)

    def derivative_at(self, x):
        return -self.f.derivative_at(x) / (self.f.value_at(x) ** 2)

    def symbolic_value(self, var_name="x"):
        return "1/" + self.f.symbolic_value(var_name)

    def symbolic_deriv(self, var_name='x'):
        return "-" + self.f.symbolic_deriv(var_name) + "/(" + self.f.symbolic_value(var_name) + ")"


class Negative(Func):
    def __init__(self, func):
        self.f = func

    def value_at(self, x):
        return -self.f.value_at(x)

    def derivative_at(self, x):
        return -self.f.derivative_at(x)

    def symbolic_value(self, var_name="x"):
        return "-" + self.f.symbolic_value(var_name)

    def symbolic_deriv(self, var_name='x'):
        r"""
        :param var_name:
        :return:
        """
        return "-" + self.f.symbolic_deriv(var_name)


class Sum(Func):

    def __init__(self, funcs):
        self.funcs = funcs

    def value_at(self, x):
        sum_ = 0
        for f in self.funcs:
            sum_ += f(x)

        return sum_

    def derivative_at(self, x):
        sum_ = 0
        for f in self.funcs:
            sum_ += f.derivative_at(x)
        return sum_

    def symbolic_deriv(self, var_name='x'):
        dstring = None
        for f in self.funcs:
            if not dstring:
                dstring = f.symbolic_deriv(var_name)
            else:
                dstring = dstring + " + " + f.symbolic_deriv(var_name)

        return dstring

    def symbolic_value(self, var_name="x"):
        dstring = None
        for f in self.funcs:
            if not dstring:
                dstring = f.symbolic_value(var_name)
            else:
                dstring = dstring + " + " + f.symbolic_value(var_name)

        return dstring


class Product(Func):

    def __init__(self, funcs):
        self.funcs = set(funcs)

    def value_at(self, x):
        prod = 1
        for f in self.funcs:
            prod *= f(x)
        return prod

    def derivative_at(self, x):
        add = 0
        for f in self.funcs:
            p1 = f.derivative_at(x)
            other_funcs = self.funcs - set([f])
            cmp = Product(other_funcs)
            p2 = cmp(x)
            add = add + (p1 * p2)
        return add

    def symbolic_deriv(self, var_name='x'):
        d_string = ""
        for f in self.funcs:
            other_funcs = self.funcs - set([f])
            prod_func = Product(other_funcs)
            f_d = f.symbolic_deriv(var_name)

            if f_d == "0":
                part = ""
            else:
                part = f_d + "."+prod_func.symbolic_value(var_name)

            if not d_string:
                d_string = part
            else:
                if not (f_d == "0"):
                    d_string = d_string + " + " + part
        return d_string

    def symbolic_value(self, var_name="x"):
        d_string = ""
        for f in self.funcs:
            d_string = d_string + ""
            if not d_string:
                d_string = "(" + f.symbolic_value(var_name) + ")"
            else:
                d_string = d_string + ".(" + f.symbolic_value(var_name) + ")"
        return d_string


class SimpleComposite(Func):
    r"""
    Represents f(g(x))
    """

    def __init__(self, f, g):
        r"""
        f(g(x))
        :param f:  single valued function
        :param g:  single valued function
        """
        self.f = f
        self.g = g

    def value_at(self, x):
        return self.f(self.g(x))

    def derivative_at(self, x):
        local_gx = self.g(x)
        return self.f.derivative_at(local_gx) * self.g.derivative_at(x)

    def symbolic_value(self, var_name="x"):
        gx_string = self.g.symbolic_value(var_name)
        debug("gx_string:{}".format(gx_string))
        fx_string = self.f.symbolic_value(gx_string)
        debug("fx_string:{}".format(fx_string))
        return fx_string

    def symbolic_deriv(self, var_name='x'):
        gx = self.g.symbolic_value(var_name)
        gx_p = self.g.symbolic_deriv(var_name)
        if gx_p == "0":
            return "0"
        return self.f.symbolic_deriv(gx) + ".(" + gx_p + ")"


class X(Func):
    r"""
    represents f(x) = x
    """

    def value_at(self, x):
        return x

    def derivative_at(self, x):
        return 1.0

    def symbolic_deriv(self, var_name='x'):
        return "1"

    def symbolic_value(self, var_name="x"):
        return str(var_name)


class Const(Func):
    def __init__(self, c):
        self.c = c

    def value_at(self, x):
        return self.c

    def derivative_at(self, x):
        return 0.0

    def symbolic_deriv(self, var_name='x'):
        return "0"

    def symbolic_value(self, var_name="x"):
        return str(self.c)


class Exp(Func):
    def __init__(self, f):
        self.f = f

    def value_at(self, x):
        return math.exp(self.f(x))

    def derivative_at(self, x):
        pow_part = Sum(self.f, -1)


class Pow(Func):
    r"""
    represents g(x) = (f(x))^n
    Does not work as a composite.
    """

    def __init__(self, func, n):
        self.f = func
        self.n = n

    def value_at(self, x):
        return math.pow(self.f(x), self.n)

    def derivative_at(self, x):
        return self.n * math.pow(self.f(x), self.n - 1) * self.f.derivative_at(x)

    def symbolic_deriv(self, var_name='x'):
        if self.n == 1:
            return self.f.symbolic_deriv(var_name)
        if self.n == 2:
            n_string = "2"
            n_minus_1 = ""
        else:
            n_string = str(self.n)
            n_minus_1 = "^" + str(self.n - 1)
        return n_string + ".((" + self.f.symbolic_value(var_name) + ")" + \
               n_minus_1 + ")" + self.f.symbolic_deriv(var_name)

    def symbolic_value(self, var_name="x"):
        return var_name + "^" + (str(self.n))


class Sine(Func):

    def value_at(self, x):
        return math.sin(x)

    def derivative_at(self, x):
        return math.cos(x)

    def symbolic_value(self, var_name="x"):
        return "sin(" + var_name + ")"

    def symbolic_deriv(self, var_name='x'):
        return "cos(" + var_name + ")"


class PolynomialTerm(Func):
    r"""
    Represents a.(x^n)
    """

    def __init__(self, coeff, n):
        self.coeff = coeff
        self.n = n
        a = Const(coeff)
        x = X()
        func = Pow(x, n)
        self.poly = Product([a, func])

    def value_at(self, x):
        return self.poly.value_at(x)

    def derivative_at(self, x):
        return self.poly.derivative_at(x)

    def symbolic_value(self, var_name="x"):
        if self.coeff == 1:
            coeff_string = ""
        else:
            coeff_string = str(self.coeff)
        return coeff_string + "(" + var_name + "^" + str(self.n) + ")"

    def symbolic_deriv(self, var_name='x'):
        return self.poly.symbolic_deriv(var_name)

def generate_b_matrix(symbol, m, n):
    for i in range(1, m + 1):
        row_string = ""
        for j in range(1, n + 1):
            row_string = row_string + symbol + "_{" + str(i) + str(j) + "} "
            if j < n:
                row_string += "& "
        if i < m:
            row_string += "\\\\"
        print(row_string)


def generate_conv_expression(m1_symbol, m2_symbol, index_tuples_1, index_tuples_2):
    conv_string = ""
    for i, t1 in enumerate(index_tuples_1):
        t1i, t1j = t1
        t2i, t2j = index_tuples_2[i]
        conv_string = conv_string + m1_symbol + "_{" + str(t1i) + str(t1j) + "}" + m2_symbol + "_{" + str(t2i) + str(
            t2j) + "} "
        if i < len(index_tuples_1) - 1:
            conv_string += " + "
    print(conv_string)


def do_convolution(x, k, xm, xn, km, kn):
    w_tuples = []
    for m in range(1, km + 1):
        for n in range(1, kn + 1):
            w_tuples.append((m, n))

    for i in range(1, xm - km + 2):
        for j in range(1, xn - kn + 2):
            x_tuples = []
            for m in range(i, i + km):
                for n in range(j, j + kn):
                    x_tuples.append((m, n))
            generate_conv_expression(x, k, x_tuples, w_tuples)
            print("& ")
        print("\\\\")


generate_b_matrix('x', 3, 4)

generate_b_matrix('w', 2, 2)


generate_conv_expression("x", "w", [(1, 1), (1, 2), (2, 1), (2, 2)], [(1, 1), (1, 2), (2, 1), (2, 2)])

print("---------------------------")

do_convolution('x','w',3,4,2,2)

print("---------------")
generate_b_matrix('y',2,3)

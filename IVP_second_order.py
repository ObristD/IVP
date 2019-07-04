from termcolor import colored
import matplotlib.pyplot as plt
import numpy as np
from numpy import sqrt, sin, cos, exp
"""
**** Solve Algorithms for Initial Value Problems first order *****
@Author: Dario Obrist
"""
print("----------------Start---------------------")


def euler(h, n, x0, y0, reference, x, but, r_fun, stps, prnt, w_apx):
    error = []
    ref_error = []
    y_ret = []
    x = np.zeros((n + 1, 1))
    y = np.zeros((n + 1, 2))
    x[0] = x0
    y[0, 0] = y0[0]
    y[0, 1] = y0[1]
    for i in range(1, n + 1):
        x[i] = x0 + i * h  # Update x
        k1 = dydx(x[i - 1] + but[0][0] * h, y[i - 1, :])

        y[i, :] = y[i - 1, :] + h * (but[1][1] * k1)  # Update y

        error.append(abs(y[i, w_apx] - reference))
        ref_error.append(reference_function(y[i, 0], y[i, 1], x[i][0]))
        y_ret.append(y[i, w_apx])

        print("y({})\t= {}   Error: {} | step:".format(format(x[i][0], ".2e"),
                                                       "\033[1m" + colored(format(y[i, w_apx], ".2e"), "magenta") +
                                                       "\033[0;0m", colored(format(error[-1], ".2e"), "red")),
              i) if stps else None

        print("Error to reference fun:\t", colored(format(ref_error[-1], ".2e"), "red"), "| step:", i,
              "\n") if r_fun else None

    if not stps:
        print("y({})\t= {} \t Error: {}".format(round(x[i][0], 3),
                                                "\033[1m" + colored(format(y[i, w_apx], ".2e"),
                                                                    "magenta") + "\033[0;0m",
                                                colored(format(error[-1], ".2e"), "red"))) if prnt else None

        print("Error to reference fun:\t", colored(format(ref_error[-1], ".2e"), "red"), "| step:", i,
              "\n") if prnt else None

    print("------------------------------------------") if prnt else None

    return error, ref_error, y_ret


def rk2(h, n, x0, y0, reference, x, but, r_fun, stps, prnt, w_apx):
    error = []
    ref_error = []
    y_ret = []
    x = np.zeros((n + 1, 1))
    y = np.zeros((n + 1, 2))
    x[0] = x0
    y[0, 0] = y0[0]
    y[0, 1] = y0[1]
    for i in range(1, n + 1):
        x[i] = x0 + i * h  # Update x
        k1 = dydx(x[i - 1] + but[0][0] * h, y[i - 1, :])
        k2 = dydx(x[i - 1] + but[1][0] * h, y[i - 1, :] + but[1][1] * h * k1)

        y[i, :] = y[i - 1, :] + h * (but[2][1] * k1 + but[2][2] * k2)  # Update y

        error.append(abs(y[i, w_apx] - reference))
        ref_error.append(reference_function(y[i, 0], y[i, 1], x[i][0]))
        y_ret.append(y[i, w_apx])

        print("y({})\t= {}   Error: {} | step:".format(format(x[i][0], ".2e"),
                                                       "\033[1m" + colored(format(y[i, w_apx], ".2e"), "magenta") +
                                                       "\033[0;0m", colored(format(error[-1], ".2e"), "red")),
              i) if stps else None

        print("Error to reference fun:\t", colored(format(ref_error[-1], ".2e"), "red"), "| step:", i,
              "\n") if r_fun else None

    if not stps:
        print("y({})\t= {} \t Error: {}".format(round(x[i][0], 3),
                                                "\033[1m" + colored(format(y[i, w_apx], ".2e"),
                                                                    "magenta") + "\033[0;0m",
                                                colored(format(error[-1], ".2e"), "red"))) if prnt else None

        print("Error to reference fun:\t", colored(format(ref_error[-1], ".2e"), "red"), "| step:", i,
              "\n") if prnt else None

    print("------------------------------------------") if prnt else None

    return error, ref_error, y_ret


def rk3(h, n, x0, y0, reference, x, but, r_fun, stps, prnt, w_apx):
    error = []
    ref_error = []
    y_ret = []
    x = np.zeros((n + 1, 1))
    y = np.zeros((n + 1, 2))
    x[0] = x0
    y[0, 0] = y0[0]
    y[0, 1] = y0[1]
    for i in range(1, n + 1):
        x[i] = x0 + i * h  # Update x
        k1 = dydx(x[i - 1] + but[0][0] * h, y[i - 1, :])
        k2 = dydx(x[i - 1] + but[1][0] * h, y[i - 1, :] + but[1][1] * h * k1)
        k3 = dydx(x[i - 1] + but[2][0] * h, y[i - 1, :] + h*(but[2][1] * k1 + but[2][2] * k2))

        y[i, :] = y[i - 1, :] + h * (but[3][1] * k1 + but[3][2] * k2 + but[3][3] * k3)  # Update y

        error.append(abs(y[i, w_apx] - reference))
        ref_error.append(reference_function(y[i, 0], y[i, 1], x[i][0]))
        y_ret.append(y[i, w_apx])

        print("y({})\t= {}   Error: {} | step:".format(format(x[i][0], ".2e"),
                                                       "\033[1m" + colored(format(y[i, w_apx], ".2e"), "magenta") +
                                                       "\033[0;0m", colored(format(error[-1], ".2e"), "red")),
              i) if stps else None

        print("Error to reference fun:\t", colored(format(ref_error[-1], ".2e"), "red"), "| step:", i,
              "\n") if r_fun else None

    if not stps:
        print("y({})\t= {} \t Error: {}".format(round(x[i][0], 3),
                                                "\033[1m" + colored(format(y[i, w_apx], ".2e"),
                                                                    "magenta") + "\033[0;0m",
                                                colored(format(error[-1], ".2e"), "red"))) if prnt else None

        print("Error to reference fun:\t", colored(format(ref_error[-1], ".2e"), "red"), "| step:", i,
              "\n") if prnt else None

    print("------------------------------------------") if prnt else None

    return error, ref_error, y_ret


def rk4(h, n, x0, y0, reference, x, but, r_fun, stps, prnt, w_apx):
    error = []
    ref_error = []
    y_ret = []
    x = np.zeros((n + 1, 1))
    y = np.zeros((n + 1, 2))
    x[0] = x0
    y[0, 0] = y0[0]
    y[0, 1] = y0[1]
    for i in range(1, n + 1):
        x[i] = x0 + i * h  # Update x
        k1 = dydx(x[i - 1] + but[0][0] * h, y[i - 1, :])
        k2 = dydx(x[i - 1] + but[1][0] * h, y[i - 1, :] + but[1][1] * h * k1)
        k3 = dydx(x[i - 1] + but[2][0] * h, y[i - 1, :] + h*(but[2][1] * k1 + but[2][2] * k2))
        k4 = dydx(x[i - 1] + but[3][0] * h, y[i - 1, :] + h * (but[3][1] * k1 + but[3][2] * k2 + but[3][3] * k3))

        y[i, :] = y[i - 1, :] + h * (but[4][1] * k1 + but[4][2] * k2 + but[4][3] * k3 + but[4][4] * k4)  # Update y

        error.append(abs(y[i, w_apx] - reference))
        ref_error.append(reference_function(y[i, 0], y[i, 1], x[i][0]))
        y_ret.append(y[i, w_apx])

        print("y({})\t= {}   Error: {} | step:".format(format(x[i][0], ".2e"),
                                                       "\033[1m" + colored(format(y[i, w_apx], ".2e"), "magenta") +
                                                       "\033[0;0m", colored(format(error[-1], ".2e"), "red")),
              i) if stps else None

        print("Error to reference fun:\t", colored(format(ref_error[-1], ".2e"), "red"), "| step:", i,
              "\n") if r_fun else None

    if not stps:
        print("y({})\t= {} \t Error: {}".format(round(x[i][0], 3),
                                                "\033[1m" + colored(format(y[i, w_apx], ".2e"),
                                                                    "magenta") + "\033[0;0m",
                                                colored(format(error[-1], ".2e"), "red"))) if prnt else None

        print("Error to reference fun:\t", colored(format(ref_error[-1], ".2e"), "red"), "| step:", i,
              "\n") if prnt else None

    print("------------------------------------------") if prnt else None

    return error, ref_error, y_ret


def method(x0, y0, x, n, h, reference, meth, but, r_fun, stps, w_apx, prnt=True):
    err_list = []
    ref_error_list = []
    y_retruns = []
    if h is not None:
        n = []
        for e in range(len(h)):
            n.append(round((x - x0) / h[e]))

        if meth == "euler":
            print("\033[1m" + colored("Method:" + meth,
                                      "yellow") + "\033[0;0m\n---------") if prnt else None
            for e in range(len(n)):
                print("Iterations:", colored(n[e], "green")) if prnt else None
                print("Step width:", colored(format(float(h[e]), ".2e"), "blue") + "\n---------") if prnt else None
                y = y0
                residual = euler(h[e], n[e], x0, y, reference, x, but, r_fun, stps, prnt, w_apx)
                err_list.append(residual[0])
                ref_error_list.append(residual[1])
                y_retruns.append(residual[2])

        if meth == "rk2" or meth == "heun" or meth == "ralston":
            print("\033[1m" + colored("Method:" + meth, "yellow") + "\033[0;0m\n---------") if prnt else None
            for e in range(len(n)):
                print("Iterations:", colored(n[e], "green")) if prnt else None
                print("Step width:", colored(format(float(h[e]), ".2e"), "blue") + "\n---------") if prnt else None
                y = y0
                residual = rk2(h[e], n[e], x0, y, reference, x, but, r_fun, stps, prnt, w_apx)
                err_list.append(residual[0])
                ref_error_list.append(residual[1])
                y_retruns.append(residual[2])

        if meth == "rk3":
            print("\033[1m" + colored("Method:" + meth, "yellow") + "\033[0;0m\n---------") if prnt else None
            for e in range(len(n)):
                print("Iterations:", colored(n[e], "green")) if prnt else None
                print("Step width:", colored(format(float(h[e]), ".2e"), "blue") + "\n---------") if prnt else None
                y = y0
                residual = rk3(h[e], n[e], x0, y, reference, x, but, r_fun, stps, prnt, w_apx)
                err_list.append(residual[0])
                ref_error_list.append(residual[1])
                y_retruns.append(residual[2])

        if meth == "rk4":
            print("\033[1m" + colored("Method:" + meth, "yellow") + "\033[0;0m\n---------") if prnt else None
            for e in range(len(n)):
                print("Iterations:", colored(n[e], "green")) if prnt else None
                print("Step width:", colored(format(float(h[e]), ".2e"), "blue") + "\n---------") if prnt else None
                y = y0
                residual = rk4(h[e], n[e], x0, y, reference, x, but, r_fun, stps, prnt, w_apx)
                err_list.append(residual[0])
                ref_error_list.append(residual[1])
                y_retruns.append(residual[2])

    elif n is not None:
        h = []
        for e in range(len(n)):
            h.append(((x - x0) / n[e]))

        if meth == "euler":
            print("\033[1m" + colored("Method:" + meth,
                                      "yellow") + "\033[0;0m\n---------") if prnt else None
            for e in range(len(n)):
                print("Iterations:", colored(n[e], "green")) if prnt else None
                print("Step width:", colored(format(float(h[e]), ".2e"), "blue") + "\n---------") if prnt else None
                y = y0
                residual = euler(h[e], n[e], x0, y, reference, x, but, r_fun, stps, prnt, w_apx)
                err_list.append(residual[0])
                ref_error_list.append(residual[1])
                y_retruns.append(residual[2])

        if meth == "rk2" or meth == "heun" or meth == "ralston":
            print("\033[1m" + colored("Method:" + meth, "yellow") + "\033[0;0m\n---------") if prnt else None
            for e in range(len(n)):
                print("Iterations:", colored(n[e], "green")) if prnt else None
                print("Step width:", colored(format(float(h[e]), ".2e"), "blue") + "\n---------") if prnt else None
                y = y0
                residual = rk2(h[e], n[e], x0, y, reference, x, but, r_fun, stps, prnt, w_apx)
                err_list.append(residual[0])
                ref_error_list.append(residual[1])
                y_retruns.append(residual[2])

        if meth == "rk3":
            print("\033[1m" + colored("Method:" + meth, "yellow") + "\033[0;0m\n---------") if prnt else None
            for e in range(len(n)):
                print("Iterations:", colored(n[e], "green")) if prnt else None
                print("Step width:", colored(format(float(h[e]), ".2e"), "blue") + "\n---------") if prnt else None
                y = y0
                residual = rk3(h[e], n[e], x0, y, reference, x, but, r_fun, stps, prnt, w_apx)
                err_list.append(residual[0])
                ref_error_list.append(residual[1])
                y_retruns.append(residual[2])

        if meth == "rk4":
            print("\033[1m" + colored("Method:" + meth, "yellow") + "\033[0;0m\n---------") if prnt else None
            for e in range(len(n)):
                print("Iterations:", colored(n[e], "green")) if prnt else None
                print("Step width:", colored(format(float(h[e]), ".2e"), "blue") + "\n---------") if prnt else None
                y = y0
                residual = rk4(h[e], n[e], x0, y, reference, x, but, r_fun, stps, prnt, w_apx)
                err_list.append(residual[0])
                ref_error_list.append(residual[1])
                y_retruns.append(residual[2])

    if w_aprox == 0:
        print(colored("Approximation for y(x) -> second derivative! \n"
                      "for first (y'(x)) change w_aprox to 1!", "red")) if prnt else None
    else:
        print(colored("Approximation for y'(x) -> first derivative! \n "
                      "for second (y(x)) change w_aprox to 0!", "red")) if prnt else None
    print("------------------------------------------") if prnt else None

    return h, err_list, ref_error_list, n, y_retruns


def turn_to_power(list, power):
    return [number**power for number in list]


def plot_residuals(sol, logaxis, invertx, inverty, w_err, met):
    stepwidth = sol[0]
    errors = [sol[w_err][i][-1] for i in range(len(sol[w_err]))]
    gca = plt.gca()
    if logaxis:
        plt.loglog(stepwidth, errors, '-o', color="blue")
        plt.autoscale(False)
        plt.plot(stepwidth, turn_to_power(stepwidth, power=1), '-', color="black", label="O(h)")
        plt.plot(stepwidth, turn_to_power(stepwidth, power=2), '--', color="black", label="O(h{})".format(u"\u00B2"))
        plt.plot(stepwidth, turn_to_power(stepwidth, power=3), '-.', color="black", label="O(h{})".format(u"\u00B3"))
        plt.plot(stepwidth, turn_to_power(stepwidth, power=4), ':', color="black", label="O(h\N{SUPERSCRIPT FOUR})")

    plt.plot(stepwidth, errors, '-o', color="blue") if not logaxis else None
    gca.invert_xaxis() if invertx else None
    gca.invert_yaxis() if inverty else None
    plt.xlabel("Step width h")
    plt.ylabel("Global error |y_N -y(X)|")
    plt.legend(title="Method:{}".format(met))
    plt.grid()
    plt.show()


def plot_results(sol, x0, X, exact_sol, met):
    if exact_sol:
        stps = 1000
        xexact = np.linspace(start=x0, stop=X, num=stps)
        yexact = np.zeros(stps)
        for i in range(stps):
            yexact[i] = exact_solution(xexact[i])
        plt.plot(xexact, yexact, color="red", label="Exact solution")

    if len(sol[3]) > 1:
        n1 = sol[3][-0]
        n2 = sol[3][-1]
        xax1 = np.linspace(start=x0, stop=X, num=len(sol[4][0]))
        plt.plot(xax1, sol[4][0], '-o', color="blue", label="n={}".format(n1))
        xax2 = np.linspace(start=x0, stop=X, num=len(sol[4][-1]))
        plt.plot(xax2, sol[4][-1], '--', color="green", label="n={}".format(n2))
        plt.legend(title="Method:{}".format(met))
        plt.grid()
        plt.show()

    else:
        n1 = sol[3][0]
        xax1 = np.linspace(start=x0, stop=X, num=len(sol[4][0]))
        plt.plot(xax1, sol[4][0], '-o', color="blue", label="n={}".format(n1))
        plt.legend(title="Method:{}".format(met))
        plt.grid()
        plt.show()


def n_for_error_smaller_than(r_err, w_err, x0_, y_, x_, reference_, metho, butc, uplmt, w_apx=0):
    print("\033[1m" + "Number of Steps:" + "\033[0;0m")
    n_steps = None
    last_error = None
    low1 = low2 = low3 = low4 = 1
    hig1 = hig2 = hig3 = hig4 = 1000
    for i1 in range(1, uplmt, 10000):
        res1 = method(x0_, y_, x_, [i1], h=None, reference=reference_,
                      meth=metho, but=butc, r_fun=False, stps=False, w_apx=w_apx, prnt=False)
        last1 = res1[w_err][0][-1]
        if last1 < r_err:
            low1 = i1 - 10000
            hig1 = i1
            print("Searching between: {} and {}".format(low1, hig1))
            break

    for i2 in range(low1, hig1+1000, 1000):
        res2 = method(x0_, y_, x_, [i2], h=None, reference=reference_,
                      meth=metho, but=butc, r_fun=False, stps=False, w_apx=w_apx, prnt=False)
        last2 = res2[w_err][0][-1]
        if last2 < r_err:
            low2 = i2 - 1000
            hig2 = i2
            print("Searching between: {} and {}".format(low2, hig2))
            break

    for i3 in range(low2, hig2+100, 100):
        res3 = method(x0_, y_, x_, [i3], h=None, reference=reference_,
                      meth=metho, but=butc, r_fun=False, stps=False, w_apx=w_apx, prnt=False)
        last3 = res3[w_err][0][-1]
        if last3 < r_err:
            low3 = i3 - 100
            hig3 = i3
            print("Searching between: {} and {}".format(low3, hig3))
            break

    for i4 in range(low3, hig3+10, 10):
        res4 = method(x0_, y_, x_, [i4], h=None, reference=reference_,
                      meth=metho, but=butc, r_fun=False, stps=False, w_apx=w_apx, prnt=False)
        last4 = res4[w_err][0][-1]
        if last4 < r_err:
            low4 = i4 - 10
            hig4 = i4
            print("Searching between: {} and {}".format(low4, hig4))
            break

    print("----")

    for iiter in range(low4, hig4+1):
        iiter = [iiter]
        res = method(x0_, y_, x_, iiter, h=None, reference=reference_,
                     meth=metho, but=butc, r_fun=False, stps=False, w_apx=w_apx, prnt=False)
        last_error = res[w_err][0][-1]
        if last_error < r_err:
            n_steps = iiter[0]
            break

    if n_steps is not None:
        print("Steps for error < {}: {}".format(colored(format(r_err, ".1e"), "blue"), colored(n_steps, "red")))
        print("Error at {} steps: {}".format(n_steps, colored(format(last_error, ".4e"), "magenta")))
    else:
        print(colored("No error < {} with N <= 100000".format(r_err), "red"))

    print("-------------------")


def difference_of_solutions(sol, first, second, w_apx, absolute=True):
    ind1 = ""
    ind2 = ""
    if first != -1:
        ind1 = first + 1
    if second != -1:
        ind2 = second + 1
    print("Difference between Y_N{} and Y_N{}".format(ind1, ind2))
    n = 1
    if len(sol[4][0]) <= abs(second):
        n = abs(second)

    for i in range(n-1, len(sol[4])):
        if absolute:
            difference = abs(sol[4][i][first] - sol[4][i][second])
        else:
            difference = sol[4][i][first] - sol[4][i][second]

        difference = format(difference, ".2e")
        print("Difference for N = {} : {}".format(sol[3][i], colored(difference, "red")))
    print("-------------")


def get_butch(meth):
    but = None

    if meth == "euler":
        but = np.array([[0, 0],
                        [0, 1]])

    if meth == "rk2":
        but = np.array([[0, 0, 0],
                        [0.5, 0.5, 0],
                        [0, 0, 1]])

    if meth == "rk3":
        but = np.array([[0, 0, 0, 0],
                        [0.5, 0.5, 0, 0],
                        [1, -1, 2, 0],
                        [0, 1/6, 2/3, 1/6]])

    if meth == "rk4":
        but = np.array([[0, 0, 0, 0, 0],
                        [0.5, 0.5, 0, 0, 0],
                        [0.5, 0, 0.5, 0, 0],
                        [1, 0, 0, 1, 0],
                        [0, 1/6, 1/3, 1/3, 1/6]])

    if meth == "heun":
        but = np.array([[0, 0, 0],
                        [1, 1, 0],
                        [0, 0.5, 0.5]])

    if meth == "ralston":
        but = np.array([[0, 0, 0],
                        [2/3, 2/3, 0],
                        [0, 1/4, 3/4]])

    return but*1.


def reference_function(y, ydif, x):
    r_fun = ydif**2 + y**2 + y**4 - 20  # Ref fun TODO: parameters: y(x) = y  y'(x) = ydif
    return abs(r_fun)


def dydx(x, y):  # Differential function transform to system first order
    z = np.array([0., 0.])
    z[0] = y[1]  # y'(x) = y'(x)
    z[1] = -y[0] - 2*y[0]**3  # f(x,y,y') = ... TODO: parameters: y(x) = y[0] |  y'(x) = y[1]
    return z

"""
Parameters to change!
if n_ not used set to None, else -> List
if h_ not used set to None, else -> List 
One of either n_ or h_ must be None, otherwise only h_ will be taken to account
Methods:
euler | rk2 | rk3 | rk4 | heun | ralston 
"""
x0_ = 0                     # Initial value of x
y_ = np.array([2, 0])  # Initial values np.array([y(0) , y'(0)])
x_ = 10                      # X final
n_ = [100, 1000, 10000]    # Number of iterations | Type: List / NoneType
h_ = None                  # Step with  | Type: List / NoneType
reference_ = 0       # reference value
method_ = "heun"           # Method
show_steps = False          # Show Steps | Type Bool
ref_fun = False             # Steps with reference Function! | Type Bool

w_aprox = 0  # 0: y(x) | 1: y'(x)

butch = get_butch(method_)
solution = method(x0_, y_, x_, n_, h_, reference_, method_, butch, ref_fun, show_steps, w_aprox)
'''Return: (h, err_list, ref_error_list, n, y) tuple with nested list'''

"""
Access results: eg. solution[4][0][1] (second value in first list):
Uncomment following function for difference in results:
first: Index of first value (eg. -1 -> last value: yN)
second: Index of second value
absolute: Set to false if positive and negative results are desired
"""

# difference_of_solutions(solution, first=-1, second=-2, w_apx=w_aprox, absolute=True)

"""
Number of steps needed until error under threshold (reference_error)
Add reference function in reference_function(y, x) [line 438]
If reference value is given: function = abs(y - ref_value) !!
Function is searching between 1 an 100'000 steps, if more steps desired raise uplmt by 10'000*x
"""
reference_error = 1e-2  # eg. 1e-3 = 10^-3 = 0.001
which_error = 1  # 1: Error to value | 2: Error to function
# n_for_error_smaller_than(reference_error, which_error, x0_, y_, x_, reference_, method_, butch, uplmt=100000)


"""
Plot residuals:
logaxis: If True plot with logarithmic axis 
"""
w_e = 2  # 1: Error to value | 2: Error to function
plot_residuals(solution, logaxis=True, invertx=False, inverty=False, w_err=w_e, met=method_)

""""
Plot results:
Only works for first and last value of n_
Plot without exact solution: exact_sol=False
"""
def exact_solution(x):
    return exp(-3/4*x)*np.sinh(5/4*x) - 0.5*x**2 - 3/2*x - 15/4


plot_results(solution, x0_, x_, exact_sol=False, met=method_)

print("\033[1m" + "\t\t==== Butcher Table ====" + "\033[0;0m")
print(butch)
print("-----------------End----------------------")

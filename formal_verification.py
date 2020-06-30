from math import erf, exp, pi, sin, cos

# Pr[Z > x] where Z is a guassian
def gauss_tail(x):
    return (1-erf(x/2**0.5))/2

# Characteristic function of a standard normal variable
def f_Z(x):
    return exp(-x*x/2.0)

# An upper bound on |f_X(v)|, given an upper bound on a1
def f_X_bound(v, a1):
    # The solution of exp(-x^2/2)+cos(x) = 0 with x in [0, pi]
    theta = 1.7780882886686339603
    if a1 * v < theta: return exp(-v*v/2)
    elif a1*v < pi   : return (-cos(a1*v))**(1/a1**2)
    else             : return 1

# An upper bound on |f_X(v)-f_Z(v)|, given an upper bound on a1
def f_X_f_Z_bound(v, a1):
    if a1 * v < pi / 2:
        return exp(-v*v/2.) - cos(a1 * v) ** (1. / a1**2)
    return 1 + exp(-v*v/2.)

# returns a bound B on
#   Pr[Z < x] - Pr[X < x] <= B.
# where X is a Rademacher sum with weights <= a1.
# Supplemental parameters: T > 0 and 0 < q < 1
def prawitz_bound(a1, x, T, q):
    # quad(f, a, b)[0] approximates the integral of f from a to b
    from scipy.integrate import quad

    k = lambda u: ((1-u) * sin(pi*u-T*u*x)/sin(pi*u) - sin(T*u*x)/pi)
    
    S1 = quad(lambda u: abs(k(u)) * f_X_f_Z_bound(u*T, a1), 0, q, epsabs=1E-11, epsrel=1E-11)
    S2 = quad(lambda u: abs(k(u)) * f_X_bound(u*T, a1),     q, 1, epsabs=1E-11, epsrel=1E-11)
    S3 = quad(lambda u:     k(u)  * f_Z(u*T),               0, q, epsabs=1E-11, epsrel=1E-11)
    S4 = gauss_tail(-x) - 0.5

    # estimation for the integration error
    assert S1[1] + S2[1] + S3[1] < 1E-10

    # estimation of the sum of the integrals
    return (S1[0] + S2[0] + S3[0] + S4)

# Aids the proof of Lemma 4.6
def berry_esseen_a1_leq_022(prawitz_bounder):
    xs = [0.35, 0.358, 0.366, 0.374, 0.38, 0.386, 0.39, 0.395, 0.399, 0.403,
          0.406, 0.409, 0.412, 0.415, 0.417, 0.419, 0.421, 0.423, 0.425, 0.427,
          0.428, 0.429, 0.43, 0.431, 0.432, 0.433, 0.434, 0.435, 0.436, 0.437,
          0.438, 0.439, 0.44, 0.441, 0.442, 0.443, 0.444, 0.445, 0.446, 0.447,
          0.448, 0.449, 0.45, 0.451, 0.452, 0.453, 0.454, 0.455, 0.456, 0.457,
          0.458, 0.459, 0.46, 0.461, 0.462, 0.463, 0.464, 0.466, 0.468, 0.47,
          0.472, 0.474, 0.476, 0.478, 0.481, 0.484, 0.487, 0.49, 0.494, 0.499,
          0.504, 0.51, 0.517, 0.526, 0.537, 0.55, 0.567, 0.589, 0.61, 0.63,
          0.65, 0.67, 0.69, 0.71, 0.73, 0.76, 0.8, 0.85, 0.91, 0.98, 1.07,
          1.2, 1.38, 1.65
    ]
    # 0.084 is enough. This accounts for (much smaller) numerical errors
    bound = 0.08399
    # verifies Pr[X < 1.65] > 1-bound.
    if gauss_tail(1.65) + prawitz_bounder(0.22, 1.65, 14.5, 0.4) >= bound:
        return False
    # For all i, verifies:
    #   Pr[Z in [xs[i], xs[i+1]]] + (Pr[Z < xs[i]] - Pr[X < xs[i]]) < bound
    for i in range(len(xs)-1):
        if ((gauss_tail(xs[i]) - gauss_tail(xs[i+1])) +
            prawitz_bounder(0.22, xs[i], 14.5, 0.4)) >= bound:
            return False
    return True

# Aids the proof of Inequality (82)
def a3_leq_015():
    for i in range(0, 211):
        for j in range(0, i+1):
            a1 = 0.185 + i * 0.0015
            a2 = 0.185 + j * 0.0015
            sig = (1-a1**2-a2**2)**0.5
            R0 = (1-a1-a2)/sig
            R1 = (1-a1+a2)/sig
            R2 = (1+a1-a2)/sig
            R3 = (1+a1+a2)/sig
            # 0.6597 is enough. This accounts for (much smaller) numerical errors
            if 0.65965 <= (gauss_tail(R0) + gauss_tail(R1) +
                           gauss_tail(R2) + gauss_tail(R3)):
                return False
    return True

# estimates the integral of f from a to b
# to within eps additive error, given
# a bound B on |f'|, and a bound C
# on the additive error involved in
# the computation (and summation) of f.
def integrate(f, a, b, eps, B, C):
    N = int(2 + B*(a-b)**2 / (4*eps + 4*C*(a-b)))
    # ensures the implied error is smaller than eps
    assert (B * (b-a)**2 / (4*N) + (b-a) * C) < eps
    sm = 0
    for k in range(1, N+1):
        sm += f(a + (2*k-1)*(b-a)/(2*N))
    return (b-a) * sm / N

# provides a provable version of prawitz_bound,
#     being more precise as eps gets lower.
def prawitz_bound2(a1, x, T, q, eps=1E-5):
    # expression appearing in S_1, S_2, S_3
    k = lambda u: ((1-u) * sin((pi - T*x)*u)/sin(pi*u) - sin(T*u*x)/pi)
    # bounds from the paper
    B_1 = T*(1+2*T*x/pi)+1.1*((T*x)**2/(2*pi)+pi)
    B_2 = T*(1+2*T*x/pi)+(T*x)**2/(2*pi)+pi
    B_3 = 2*T/3*(1+2*T*x/pi)+(T*x)**2/(2*pi)+pi
    B_4 = 0.25
    # Computing the integrated functions has absolute error < abs_error
    abs_error = 2**-40 * (2+T*x)
    # the maximal additive errors sum to < 1E-5.
    S1 = integrate(lambda u: abs(k(u)) * f_X_f_Z_bound(u*T, a1), 0, q, 6*eps/20, B_1, abs_error)
    S2 = integrate(lambda u: abs(k(u)) * f_X_bound(u*T, a1),     q, 1, 8*eps/20, B_2, abs_error)
    S3 = integrate(lambda u:     k(u)  * f_Z(u*T),               0, q, 5*eps/20, B_3, abs_error)
    S4 = integrate(lambda u: exp(-u*u/2)/(2*pi)**0.5,            0, x, 1*eps/20, B_4, abs_error)
    # estimation of the sum of the integrals plus a bound on the additive error.
    return (S1 + S2 + S3 + S4) + eps

# main check
print("performing checks")
assert a3_leq_015()
# verifies the 93 inequalities the fast way
assert berry_esseen_a1_leq_022(prawitz_bound)
# verifies Inequality (54)
assert prawitz_bound(0.31, 1, 10, 0.4) <= 0.09115
print("checks passed")

##################################################

# excessive pedantic check
print("performing (pedantic) slow check")
# verifies the 93 inequalities the slow way
assert berry_esseen_a1_leq_022(prawitz_bound2)
# verifies Inequality (54)
assert prawitz_bound2(0.31, 1, 10, 0.4, 5E-6) <= 0.09115
print("slow check passed")

# 
# Relevant values, computed with higher accuracy.
# Denote by E(a_1, x) the right hand side of (150), applied for specific a_1, x.
#
#
# Note that:
# E(0.31, 1) < 0.9115,
# E(0.22, 1.65) + Pr[Z > 1.65] < 0.084,
#         where Z ~ N(0,1) a standard normal variable,
# E(0.22, xs[i]) + Pr[Z in (xs[i], xs[i+1])] < 0.084,
#         with xs as in berry_esseen_a1_leq_022, and i in [0, 92].
#
##
##
## E(0.31, 1) = 0.09114105 \pm 10^{-8},
##
## E(0.22, 0.35) = 0.08061047 \pm 10^{-8}
## E(0.22, 0.358) = 0.08066061 \pm 10^{-8}
## E(0.22, 0.366) = 0.08089529 \pm 10^{-8}
## E(0.22, 0.374) = 0.08123427 \pm 10^{-8}
## E(0.22, 0.38) = 0.08151824 \pm 10^{-8}
## E(0.22, 0.386) = 0.08180668 \pm 10^{-8}
## E(0.22, 0.39) = 0.0819951 \pm 10^{-8}
## E(0.22, 0.395) = 0.08222157 \pm 10^{-8}
## E(0.22, 0.399) = 0.08239318 \pm 10^{-8}
## E(0.22, 0.403) = 0.08255469 \pm 10^{-8}
## E(0.22, 0.406) = 0.08266851 \pm 10^{-8}
## E(0.22, 0.409) = 0.0827756 \pm 10^{-8}
## E(0.22, 0.412) = 0.08287563 \pm 10^{-8}
## E(0.22, 0.415) = 0.08296828 \pm 10^{-8}
## E(0.22, 0.417) = 0.08302581 \pm 10^{-8}
## E(0.22, 0.419) = 0.08307986 \pm 10^{-8}
## E(0.22, 0.421) = 0.08313035 \pm 10^{-8}
## E(0.22, 0.423) = 0.08317723 \pm 10^{-8}
## E(0.22, 0.425) = 0.08322042 \pm 10^{-8}
## E(0.22, 0.427) = 0.08325987 \pm 10^{-8}
## E(0.22, 0.428) = 0.08327817 \pm 10^{-8}
## E(0.22, 0.429) = 0.08329552 \pm 10^{-8}
## E(0.22, 0.43) = 0.0833119 \pm 10^{-8}
## E(0.22, 0.431) = 0.08332731 \pm 10^{-8}
## E(0.22, 0.432) = 0.08334175 \pm 10^{-8}
## E(0.22, 0.433) = 0.0833552 \pm 10^{-8}
## E(0.22, 0.434) = 0.08336766 \pm 10^{-8}
## E(0.22, 0.435) = 0.08337913 \pm 10^{-8}
## E(0.22, 0.436) = 0.08338959 \pm 10^{-8}
## E(0.22, 0.437) = 0.08339905 \pm 10^{-8}
## E(0.22, 0.438) = 0.0834075 \pm 10^{-8}
## E(0.22, 0.439) = 0.08341494 \pm 10^{-8}
## E(0.22, 0.44) = 0.08342135 \pm 10^{-8}
## E(0.22, 0.441) = 0.08342673 \pm 10^{-8}
## E(0.22, 0.442) = 0.08343109 \pm 10^{-8}
## E(0.22, 0.443) = 0.08343441 \pm 10^{-8}
## E(0.22, 0.444) = 0.08343669 \pm 10^{-8}
## E(0.22, 0.445) = 0.08343793 \pm 10^{-8}
## E(0.22, 0.446) = 0.08343812 \pm 10^{-8}
## E(0.22, 0.447) = 0.08343726 \pm 10^{-8}
## E(0.22, 0.448) = 0.08343535 \pm 10^{-8}
## E(0.22, 0.449) = 0.08343237 \pm 10^{-8}
## E(0.22, 0.45) = 0.08342834 \pm 10^{-8}
## E(0.22, 0.451) = 0.08342324 \pm 10^{-8}
## E(0.22, 0.452) = 0.08341708 \pm 10^{-8}
## E(0.22, 0.453) = 0.08340985 \pm 10^{-8}
## E(0.22, 0.454) = 0.08340154 \pm 10^{-8}
## E(0.22, 0.455) = 0.08339216 \pm 10^{-8}
## E(0.22, 0.456) = 0.08338171 \pm 10^{-8}
## E(0.22, 0.457) = 0.08337017 \pm 10^{-8}
## E(0.22, 0.458) = 0.08335756 \pm 10^{-8}
## E(0.22, 0.459) = 0.08334387 \pm 10^{-8}
## E(0.22, 0.46) = 0.08332909 \pm 10^{-8}
## E(0.22, 0.461) = 0.08331323 \pm 10^{-8}
## E(0.22, 0.462) = 0.08329628 \pm 10^{-8}
## E(0.22, 0.463) = 0.08327825 \pm 10^{-8}
## E(0.22, 0.464) = 0.08325913 \pm 10^{-8}
## E(0.22, 0.466) = 0.08321762 \pm 10^{-8}
## E(0.22, 0.468) = 0.08317177 \pm 10^{-8}
## E(0.22, 0.47) = 0.08312157 \pm 10^{-8}
## E(0.22, 0.472) = 0.08306703 \pm 10^{-8}
## E(0.22, 0.474) = 0.08300814 \pm 10^{-8}
## E(0.22, 0.476) = 0.08294493 \pm 10^{-8}
## E(0.22, 0.478) = 0.0828774 \pm 10^{-8}
## E(0.22, 0.481) = 0.08276805 \pm 10^{-8}
## E(0.22, 0.484) = 0.08264908 \pm 10^{-8}
## E(0.22, 0.487) = 0.08252058 \pm 10^{-8}
## E(0.22, 0.49) = 0.08238264 \pm 10^{-8}
## E(0.22, 0.494) = 0.08218421 \pm 10^{-8}
## E(0.22, 0.499) = 0.08191329 \pm 10^{-8}
## E(0.22, 0.504) = 0.08161755 \pm 10^{-8}
## E(0.22, 0.51) = 0.08123098 \pm 10^{-8}
## E(0.22, 0.517) = 0.08073822 \pm 10^{-8}
## E(0.22, 0.526) = 0.08004279 \pm 10^{-8}
## E(0.22, 0.537) = 0.07910732 \pm 10^{-8}
## E(0.22, 0.55) = 0.07791051 \pm 10^{-8}
## E(0.22, 0.567) = 0.07649992 \pm 10^{-8}
## E(0.22, 0.589) = 0.07549218 \pm 10^{-8}
## E(0.22, 0.61) = 0.07538169 \pm 10^{-8}
## E(0.22, 0.63) = 0.07566508 \pm 10^{-8}
## E(0.22, 0.65) = 0.07596262 \pm 10^{-8}
## E(0.22, 0.67) = 0.07604335 \pm 10^{-8}
## E(0.22, 0.69) = 0.0757856 \pm 10^{-8}
## E(0.22, 0.71) = 0.07513624 \pm 10^{-8}
## E(0.22, 0.73) = 0.07408539 \pm 10^{-8}
## E(0.22, 0.76) = 0.07181184 \pm 10^{-8}
## E(0.22, 0.8) = 0.06841516 \pm 10^{-8}
## E(0.22, 0.85) = 0.06649778 \pm 10^{-8}
## E(0.22, 0.91) = 0.06581545 \pm 10^{-8}
## E(0.22, 0.98) = 0.06231347 \pm 10^{-8}
## E(0.22, 1.07) = 0.05626879 \pm 10^{-8}
## E(0.22, 1.2) = 0.05159314 \pm 10^{-8}
## E(0.22, 1.38) = 0.04286319 \pm 10^{-8}
## E(0.22, 1.65) = 0.03136671 \pm 10^{-8}

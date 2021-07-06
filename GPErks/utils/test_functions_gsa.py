from itertools import combinations

import numpy as np
import pandas as pd


def Ishigami(x, a=7.0, b=0.1):
    """
    =========================================================================
    Ishigami function: f(X1, X2, X3) = sin(X1) + a*sin^2(X2) + b*X3^4*sin(X1)
    for Xi ~ U([-pi, pi]), a > 0, b > 0.
    =========================================================================
    """
    return (
        np.sin(x[0])
        + a * np.power(np.sin(x[1]), 2)
        + b * np.power(x[2], 4) * np.sin(x[0])
    )


def Ishigami_theoretical_Si(a=7.0, b=0.1):
    D = 3

    index_i = ["X{}".format(i + 1) for i in range(D)]
    index_ij = [
        "({}, {})".format(c[0], c[1]) for c in combinations(index_i, 2)
    ]

    V1 = 0.5 * np.power(1 + b * np.power(np.pi, 4) / 5, 2)
    V2 = np.power(a, 2) / 8
    V3 = 0
    Vi = [V1, V2, V3]

    VT1 = (
        0.5 * np.power(1 + b * np.power(np.pi, 4) / 5, 2)
        + 8 * np.power(b, 2) * np.power(np.pi, 8) / 225
    )
    VT2 = V2
    VT3 = 8 * np.power(b, 2) * np.power(np.pi, 8) / 225
    VTi = [VT1, VT2, VT3]

    V12 = 0
    V13 = (1 / 18 - 1 / 50) * np.power(b, 2) * np.power(np.pi, 8)
    V23 = 0
    Vij = [V12, V13, V23]

    V = (
        0.5
        + np.power(a, 2) / 8
        + b * np.power(np.pi, 4) / 5
        + np.power(b, 2) * np.power(np.pi, 8) / 18
    )

    STi = [VTi[i] / V for i in range(D)]
    Si = [Vi[i] / V for i in range(D)]
    Sij = [Vij[i] / V for i in range(len(index_ij))]

    df_STi = pd.DataFrame(
        data=np.round(STi, 6).reshape(-1, 1), index=index_i, columns=["STi"]
    )
    df_Si = pd.DataFrame(
        data=np.round(Si, 6).reshape(-1, 1), index=index_i, columns=["Si"]
    )
    df_Sij = pd.DataFrame(
        data=np.round(Sij, 6).reshape(-1, 1), index=index_ij, columns=["Sij"]
    )

    return df_STi, df_Si, df_Sij


def SobolGstar(x, a, delta, alpha):
    """
    ===================================================================================
    Sobol's-Saltelli G* function: g(X1,...,Xd) = ∏_{i=1}^{d} g_i,
    where g_i = (|2 * (Xi + delta_i - int(Xi + delta_i) - 1|^alpha_i + a_i) / (1 + a_i)
    for Xi ~ U([0, 1]), delta_i ~ U([0, 1]), alpha_i > 0, a_i > 0.
    ===================================================================================
    """
    D = len(a)

    G = 1
    for i in range(D):
        num = (1 + alpha[i]) * np.power(
            np.abs(2 * (x[i] + delta[i] - np.floor(x[i] + delta[i])) - 1),
            alpha[i],
        ) + a[i]
        den = 1 + a[i]
        gi = num / den
        G *= gi

    return G


def SobolGstar_theoretical_Si(a, delta, alpha):
    D = len(a)

    index_i = ["X{}".format(i + 1) for i in range(D)]
    index_ij = [
        "({}, {})".format(c[0], c[1]) for c in combinations(index_i, 2)
    ]

    Vi = []
    V = 1
    for i in range(D):
        vi = np.power(alpha[i], 2) / (
            (1 + 2 * alpha[i]) * np.power(1 + a[i], 2)
        )
        Vi.append(vi)
        V = V * (1 + vi)
    V -= 1

    VTi = []
    for i in range(D):
        vti = Vi[i]
        for j in list(set(range(D)) - {i}):
            vti = vti * (1 + Vi[j])
        VTi.append(vti)

    Vij = []
    for c in combinations(range(D), 2):
        Vij.append(Vi[c[0]] * Vi[c[1]])

    STi = [VTi[i] / V for i in range(D)]
    Si = [Vi[i] / V for i in range(D)]
    Sij = [Vij[i] / V for i in range(len(index_ij))]

    df_STi = pd.DataFrame(
        data=np.round(STi, 6).reshape(-1, 1), index=index_i, columns=["STi"]
    )
    df_Si = pd.DataFrame(
        data=np.round(Si, 6).reshape(-1, 1), index=index_i, columns=["Si"]
    )
    df_Sij = pd.DataFrame(
        data=np.round(Sij, 6).reshape(-1, 1), index=index_ij, columns=["Sij"]
    )

    return df_STi, df_Si, df_Sij

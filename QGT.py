import numpy as np
from math import *
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from tabulate import tabulate



C = np.array([1,0])
D = np.array([0,1])

CC = np.kron(C,C)
DD = np.kron(D,D)


# To find conjugate transpose
def H(j):
    return j.conjugate().T  

# Entanglemet operator J(gamma)
def J(g):
    j = np.zeros((4,4), dtype = complex)
    for i in range(4):
        j[i][i] = cos(g/2)
    j[0][3] = -1j*sin(g/2)
    j[1][2] = 1j*sin(g/2)
    j[2][1] = 1j*sin(g/2)
    j[3][0] = -1j*sin(g/2)
    return j


# two parameters staegy operator
def U1(theta, phi):
    u = np.array([[np.exp(1j*phi)*cos(theta/2), sin(theta/2)], 
                  [-sin(theta/2), np.exp(-1j*phi)*cos(theta/2)]])
    return u

# Three parameters staegy operator

def U2(theta, a, b):
    u = np.array([[np.exp(1j*a)*cos(theta/2), 1j*np.exp(1j*b)*sin(theta/2)], 
                  [1j*np.exp(-1j*b)*sin(theta/2), np.exp(-1j*a)*cos(theta/2)]])
    return u

# final state

def Psi(J, Ua, Ub):
    psi = np.matmul(np.matmul(H(J), np.kron(Ua,Ub)),np.matmul(J, CC))
    return psi

def expected_payoff(p, g, Ua, Ub):
    a, b= 0, 0
    psi = Psi(J(g), Ua, Ub)
    for i in range(len(p[0])):
        a += p[0][i]*(abs(psi[i]))**2
        b += p[1][i]*(abs(psi[i]))**2
    return a, b

# For plotting

def payoff_plot(gamma, p, x, y):
    
    j = J(gamma)
    Ua = U(x*pi,0) if x >= 0 else U(0,-x*pi/2)
    Ub = U(y*pi,0) if y >= 0 else U(0,-y*pi/2)
    psi = Psi(j,Ua,Ub)
    a, b = expected_payoff(p, psi)
    return a

def HD_payoff_matrix(v, i, d):
    return np.array([[(v - i)/2, v, 0, v/2 -d],[(v - i)/2, 0, v, v/2 -d]])

def Psi_dense(J, Ua, Ub):
    rhoi = np.outer(CC,CC)
    rho1 = np.matmul(J, np.matmul(rhoi, H(J)))
    rho2 = np.matmul(np.kron(Ua, Ub), np.matmul(rho1, H(np.kron(Ua, Ub))))
    rhof = np.matmul(H(J), np.matmul(rho2, J))
    return rhof

# The payoff operator 
def payoff_op(p):
    C = np.array([1,0])
    D = np.array([0,1])
    basis = {
        'b0' : np.kron(C,C),
        'b1' : np.kron(C,D),
        'b2' : np.kron(D,C),
        'b3' : np.kron(D,D)
    }
    pa, pb = 0, 0
    for i in range(len(p[0])):
        pa += p[0][i]*np.outer(basis['b'+str(i)], basis['b'+str(i)])
        pb += p[1][i]*np.outer(basis['b'+str(i)], basis['b'+str(i)]) 
    return pa, pb

# expected payoff for mixed strategies with probability p and q 
def mixed_payoff(j, ua1, ua2, ub1, ub2, p, q, payoff):
    rho = p*q*Psi_dense(j, ua1, ub1) + p*(1-q)*Psi_dense(j, ua1, ub2) + (1-p)*q*Psi_dense(j, ua2, ub1) + (1-p)*(1-q)*Psi_dense(j, ua2, ub2) 
    P = payoff_op(payoff)
    a = np.trace(np.matmul(P[0],rho))
    b = np.trace(np.matmul(P[1],rho))
    return a.real, b.real


def payoff_tableg(U, po, g, sl, sa):
    t = [['', '']]
    t[0] += sl
    def mp(a1, a2, b1, b2, p, q):
        al, bo = mixed_payoff(J(g), a1, a2, b1, b2, p, q, po)
        return round(al.real,2), round(bo.real,2)
    for i in range(len(sl)):
        t.append(['', sl[i]])
        for j in range(len(sl)):
            if len(sa[sl[i]][0]) == 3:
                t[1+i].append(mp(U(sa[sl[i]][0][0], sa[sl[i]][0][1]), U(sa[sl[i]][1][0], sa[sl[i]][1][1]), U(sa[sl[j]][0][0], sa[sl[j]][0][1]), U(sa[sl[j]][1][0], sa[sl[j]][1][1]), sa[sl[i]][0][2], sa[sl[j]][1][2]))
            elif len(sa[sl[i]][0]) == 4:
                t[1+i].append(mp(U(sa[sl[i]][0][0], sa[sl[i]][0][1], sa[sl[i]][0][2]), U(sa[sl[i]][1][0], sa[sl[i]][1][1], sa[sl[i]][1][2]), U(sa[sl[j]][0][0], sa[sl[j]][0][1], sa[sl[j]][0][2]), U(sa[sl[j]][1][0], sa[sl[j]][1][1], sa[sl[j]][1][2]), sa[sl[i]][0][3], sa[sl[j]][1][3]))
                
    t[1][0] = 'Al'
    headers = ["Bob",'']
    print(tabulate(t, headers, tablefmt="pretty"))

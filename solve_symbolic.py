import sympy as sp
from sympy.matrices import Matrix
from sympy.physics.mechanics import dynamicsymbols, LagrangesMethod
import dill

dill.settings['recurse'] = True

def solve_symbolic(N=3, verbose=True):
    if verbose: print('Solving Symbolic Problem')
    if verbose: print(37*"=")

    #############################################################################
    if verbose: print("Formulating Symbolic Problem... ", end='', flush=True)   #
    t, g = sp.symbols('t g')                                                    #
                                                                                #
    if N == 1:                                                                  #
        l, m = Matrix([sp.symbols("l"),]), Matrix([sp.symbols("m"),])           #
                                                                                #
        q = [dynamicsymbols("q"),]                                              #
        p = [dynamicsymbols("q", 1),]                                           #
        a = [dynamicsymbols("q", 2),]                                           #
    else:                                                                       #
        l = Matrix(sp.symbols(" ".join(f"l_{_}" for _ in range(N))))            #
        m = Matrix(sp.symbols(" ".join(f"m_{_}" for _ in range(N))))            #
                                                                                #
        q = dynamicsymbols(" ".join(f"q_{_}" for _ in range(N)))                #
        p = dynamicsymbols(" ".join(f"q_{_}" for _ in range(N)), 1)             #
        a = dynamicsymbols(" ".join(f"q_{_}" for _ in range(N)), 2)             #
                                                                                #
    x, y = [l[0] * sp.sin(q[0])], [- l[0] * sp.cos(q[0])]                       #
    for i in range(1, N):                                                       #
        x.append(x[i - 1] + l[i] * sp.sin(q[i]))                                #
        y.append(y[i - 1] - l[i] * sp.cos(q[i]))                                #
                                                                                #
    v_sqr = Matrix([_x.diff(t) ** 2 + _y.diff(t) ** 2 for _x,_y in zip(x, y)])  #
    V = g * m.dot(y)                                                            #
    T = m.dot(v_sqr) / 2                                                        #
    L, H = T - V, T + V                                                         #
                                                                                #
    LM = LagrangesMethod(L, q)                                                  #
    EL = LM.form_lagranges_equations()                                          #
    if verbose: print('Done!')                                                  #
    #############################################################################

    #############################################################################
    if verbose: print("Solving Lagrange's Equations... ", end='', flush=True)   #
    EOM = Matrix([                                                              #
        sp.solve(LM.eom[0], _a)[0] for _a in a                                  #
            ])                                                                  #
    if verbose: print('Done!')                                                  #
    #############################################################################

    #############################################################################
    if verbose: print('Caching solution... ' + 12*' ', end='', flush=True)      #
    ODEsystem = sp.utilities.lambdify([t, [*q, *p], *l, *m, g], [*p, *EOM]) #
    Hamiltonian = sp.utilities.lambdify([t, [*q, *p], *l, *m, g], H)            #
    Kinetic = sp.utilities.lambdify([t, [*q, *p], *l, *m, g], T)          #
    Potential = sp.utilities.lambdify([t, [*q, *p], *l, *m, g], V)      #
                                                                                #
    dill.dump(ODEsystem, open("./cache/pypendula_cached_soln", "wb"))           #
    dill.dump(Hamiltonian, open("./cache/pypendula_cached_h", "wb"))            #
    dill.dump(Kinetic, open("./cache/pypendula_cached_kinetic", "wb"))          #
    dill.dump(Potential, open("./cache/pypendula_cached_potential", "wb"))      #
    if verbose: print('Done!')                                                  #
    #############################################################################

    if verbose: print(37*"=")    
    if verbose: print('Finished!')
    return {
        'LM': LM, 
        'EL': EL, 
        'EOM': EOM, 
        'T': T, 
        'V': V, 
        'H': H,
        'ODEsystem': ODEsystem, 
        'Hamiltonian': Hamiltonian, 
        'Kinetic': Kinetic, 
        'Potential': Potential
        }
import sympy as sp
from sympy.matrices import Matrix
from sympy.physics.mechanics import dynamicsymbols, Lagrangian, LagrangesMethod, ReferenceFrame, Point, Particle
import dill

dill.settings['recurse'] = True
G = 10

def solve_symbolic(N=3, verbose=True, substitute=False):
    if verbose: print('Solving Symbolic Problem')
    if verbose: print(37*"=")

    #############################################################################
    if verbose: print("Formulating Symbolic Problem... ", end='', flush=True)   #
    _m, g, _l, t = sp.symbols('m g l t')                                          #
    l = Matrix([_l for _ in range(N)])                                           #
    m = Matrix([_m for _ in range(N)])                                           #
                                                                                #
    q = dynamicsymbols(f"q_0:{N}") if N > 1 else [dynamicsymbols("q"),]              #
    p = dynamicsymbols(f"q_0:{N}", level=1) if N > 1 else [dynamicsymbols("q", 1),]  #
    dp = dynamicsymbols(f"q_0:{N}", level=2) if N > 1 else [dynamicsymbols("q", 2),] #

    x, y = [l[0] * sp.sin(q[0])], [- l[0] * sp.cos(q[0])]                       #
    for i in range(1, N):                                                       #
        x.append(x[i - 1] + l[i] * sp.sin(q[i]))                                #
        y.append(y[i - 1] - l[i] * sp.cos(q[i]))                                #
                                                                                #
    x, y = Matrix(x), Matrix(y)
    v_sqr = Matrix([_x.diff(t) ** 2 + _y.diff(t) ** 2 for _x,_y in zip(x, y)])  #
    V = g * m.dot(y)                                                            #
    T = m.dot(v_sqr) / 2                                                        #

    H = T + V # Hamiltonian
    L = T - V  # Lagrangian
    LM = LagrangesMethod(L, q)                                         #
    EL = LM.form_lagranges_equations()                                          #
    if verbose: print('Done!')                                                  #
    #############################################################################

    #############################################################################
    if verbose: print("Solving Lagrange's Equations... ", end='', flush=True)   #
    # EOM = Matrix([                                                  #
    #     sp.solve(LM.eom[_], dp[_])[0] for _ in range(N)                         #
    #     ])                                                                     #
    EOM = sp.simplify(LM.eom)                                                   #
    if verbose: print('Done!')                                                  #
    #############################################################################

    #############################################################################
    if substitute: 
        print("Performing Substitutions... ", end='', flush=True)          #    
        EOM = EOM.subs([
            (sp.symbols('l'), 1 / N), 
            (sp.symbols('m'), 1 / N),
            (sp.symbols('g'), G),
            ])
        EOM = sp.simplify(EOM)                                                      #
        print('Done!')                                                 #
    #############################################################################

    # #############################################################################
    # if verbose: print('Caching solution... ' + 12*' ', end='', flush=True)      #
    # ODEsystem = sp.utilities.lambdify([t, [*q, *p], *l, *m, g], [*p, *EOM])     #
    # Hamiltonian = sp.utilities.lambdify([t, [*q, *p], *l, *m, g], H)            #
    # Kinetic = sp.utilities.lambdify([t, [*q, *p], *l, *m, g], T)                #
    # Potential = sp.utilities.lambdify([t, [*q, *p], *l, *m, g], V)              #
    #                                                                             #
    # dill.dump(ODEsystem, open("./cache/pypendula_cached_soln", "wb"))           #
    # dill.dump(Hamiltonian, open("./cache/pypendula_cached_h", "wb"))            #
    # dill.dump(Kinetic, open("./cache/pypendula_cached_kinetic", "wb"))          #
    # dill.dump(Potential, open("./cache/pypendula_cached_potential", "wb"))      #
    # if verbose: print('Done!')                                                  #
    # #############################################################################

    if verbose: print(37*"=")    
    if verbose: print('Finished!')
    return {
        'LM': LM, 
        'EL': EL, 
        'EOM': EOM, 
        'T': T, 
        'V': V,
        'H': H,
        # 'ODEsystem': ODEsystem, 
        # 'Hamiltonian': Hamiltonian, 
        # 'Kinetic': Kinetic, 
        # 'Potential': Potential
        }
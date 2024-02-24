import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Arc, Circle

import sympy as sp
from sympy.matrices import Matrix
from sympy.physics.mechanics import dynamicsymbols, Lagrangian, LagrangesMethod, ReferenceFrame, Point, Particle
import dill

dill.settings['recurse'] = True
rng = np.random.default_rng()

class PyPendula:
    def __init__(
            self, 
            N=3, 
            params={
                'm' : 1,
                'g' : 10,
                'l' : 1,
            },
            ics = None,
            alpha=3,
            ):
        self.params = params
        self.N = N
        self.alpha = alpha

        if ics is None:
            ics_q = rng.uniform(-np.pi / self.alpha, np.pi / self.alpha, size=self.N)
            ics_p = np.zeros(self.N)
            self.ics = np.hstack([ics_q, ics_p])
        else:
            self.ics = ics

    def show_diagram(self):
        assert self.N == 3


        theta1, theta2, theta3, _, _, _ = self.ics
        l = self.params['l']

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim([-3.0 * l, 3.0 * l])
        ax.set_ylim([-3.5 * l, 3.0 * l])

        x0, y0 =                       0,                       0
        x1, y1 =      l * np.sin(theta1),    - l * np.cos(theta1)
        x2, y2 = x1 + l * np.sin(theta2), y1 - l * np.cos(theta2)
        x3, y3 = x2 + l * np.sin(theta3), y2 - l * np.cos(theta3)

        ax.vlines([0, x1, x2], [y1, y2, y3], [0, y1, y2], linestyles='dashed', colors='black', zorder=1)
        ax.hlines([y1, y2, y3], [x1, x1, x2], [0, x2, x3], linestyles='dashed', colors='black', zorder=1)

        # Rods
        ax.plot([x0, x1], [y0, y1], lw=3, color='darkgray')  # , lw=2 * r_l, color='blue',  zorder=4)
        ax.plot([x1, x2], [y1, y2], lw=3, color='darkgray')  # , lw=2 * r_l, color='red',   zorder=3)
        ax.plot([x2, x3], [y2, y3], lw=3, color='darkgray')  # , lw=2 * r_l, color='green', zorder=2)

        # Masses
        ax.add_patch(Circle([x1, y1], radius=l / 6, color='blue', label=r'$(x_1,y_1)$',  zorder=4))
        ax.text(x1, y1, r'$m_1$', zorder=10)

        ax.add_patch(Circle([x2, y2], radius=l / 6, color='red', label=r'$(x_2,y_2)$', zorder=3))
        ax.text(x2, y2, r'$m_2$', zorder=10)

        ax.add_patch(Circle([x3, y3], radius=l / 6, color='green', label=r'$(x_3,y_3)$', zorder=2))
        ax.text(x3, y3, r'$m_3$', zorder=10)

        # Pins
        ax.add_patch(Circle(             [x0, y0], radius=l / 12, color='black',  zorder=4))
        
        # plt.legend()

        return fig

    def solve_symbolic(self):
        q = dynamicsymbols(f'q:{self.N}')
        dq = dynamicsymbols(f'q:{self.N}', level=1)
        ddq = dynamicsymbols(f'q:{self.N}', level=2)
        p = dynamicsymbols(f'p:{self.N}')
        dp = dynamicsymbols(f'p:{self.N}', level=1)
        l, m, g, t = sp.symbols('l m g t')

        ###################
        x, y = [l * sp.sin(q[0])], [- l * sp.cos(q[0])]
        for i in range(1, N):
            x.append(x[i - 1] + l * sp.sin(q[i]))
            y.append(y[i - 1] - l * sp.cos(q[i]))
    
        v_sqr = Matrix([_x.diff(t) ** 2 + _y.diff(t) ** 2 for _x,_y in zip(x, y)])
        x, y = Matrix(x), Matrix(y)
        ####################

        potential = -m * g * sum(y)
        kinetic = sp.Rational(1, 2) * m * sum(v_sqr)
        lagrangian = kinetic - potential
        self.lagranges_method = LagrangesMethod(lagrangian, q)
        self.euler_lagrange_eqns = self.lagranges_method.form_lagranges_equations()
        self.eom = sp.simplify(self.lagranges_method.eom.subs([(dq[_], p[_]) for _ in range(self.N)]))
        self.symbolic_dp = Matrix(list(sp.solve(self.eom, *dp).values()))

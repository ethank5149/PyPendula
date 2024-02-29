import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Arc, Circle

import sympy as sp
from sympy.matrices import Matrix
from sympy.physics.mechanics import dynamicsymbols, Lagrangian, LagrangesMethod, ReferenceFrame, Point, Particle
from scipy.integrate import solve_ivp
from functools import partial
import dill


dill.settings['recurse'] = True
rng = np.random.default_rng()


## Inspired by stackoverflow user greenstick's comment: 
## https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
def progress_bar (
        frame,                         # (Required): current frame (Int)
        total,                         # (Required): total frames (Int)
        prefix = 'Saving animation:',  # (Optional): prefix string (Str)
        suffix = '',                   # (Optional): suffix string (Str)
        decimals = 1,                  # (Optional): positive number of decimals in percent complete (Int)        
        length = 100,                  # (Optional): character length of bar (Int)
        fill = '█',                    # (Optional): bar fill character (Str)
        printEnd = "\r"                # (Optional): end character (e.g. "\r", "\r\n") (Str)
        ):
    iteration = frame + 1
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))  # f"{100 * (iteration / float(total)):.1f}"
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    if iteration == total:
        print()

class PyPendula:
    def __init__(
            self, 
            N=3, 
            params={
                'm' : 1,
                'g' : 1,
                'l' : 9.807,
            },
            ics = None,
            alpha=3,
            t_f=30,
            n_t=100,
            fps=60,
            ):
        self.params = params
        self.N = N
        self.alpha = alpha
        self.t_f = t_f
        self.n_t = n_t
        self.fps = fps

        self.soln = None
        self.t_eval = None

        if ics is None:
            ics_q = rng.uniform(-np.pi / self.alpha, np.pi / self.alpha, size=self.N)
            ics_p = np.zeros(self.N)
            self.ics = np.hstack([ics_q, ics_p])
        else:
            self.ics = ics
        
        self.ics_tag = 'ics=[' + ','.join((f'{ic:.4f}' for ic in self.ics)) + ']'

    def solve_symbolic(self):
        q = dynamicsymbols(f'q:{self.N}')
        dq = dynamicsymbols(f'q:{self.N}', level=1)
        ddq = dynamicsymbols(f'q:{self.N}', level=2)
        p = dynamicsymbols(f'p:{self.N}')
        dp = dynamicsymbols(f'p:{self.N}', level=1)
        l, m, g, t = sp.symbols('l m g t')

        ###################
        x, y = [l * sp.sin(q[0])], [- l * sp.cos(q[0])]
        for i in range(1, self.N):
            x.append(x[i - 1] + l * sp.sin(q[i]))
            y.append(y[i - 1] - l * sp.cos(q[i]))
    
        v_sqr = Matrix([_x.diff(t) ** 2 + _y.diff(t) ** 2 for _x,_y in zip(x, y)])
        x, y = Matrix(x), Matrix(y)
        ####################

        # potential = -m * g * sum(y)
        potential = m * g * sum(y)
        kinetic = sp.Rational(1, 2) * m * sum(v_sqr)
        lagrangian = kinetic - potential
        self.lagranges_method = LagrangesMethod(lagrangian, q)
        self.euler_lagrange_eqns = self.lagranges_method.form_lagranges_equations()
        self.eom = sp.simplify(self.lagranges_method.eom.subs([(dq[_], p[_]) for _ in range(self.N)]))
        self.symbolic_dp = Matrix(list(sp.solve(self.eom, *dp).values()))
        self.numeric_dp = sp.utilities.lambdify([t, 
                                       [*q, *p], 
                                       m, g, l], 
                                      [*p, *self.symbolic_dp])
        dill.dump(self.numeric_dp, open(f"./cache/pypendula_cached_soln_n{self.N}", "wb"))

    def solve_numeric(self):
        try:
            self.numeric_dp = dill.load(open("./cache/pypendula_cached_soln_n{self.N}", "rb"))
        except:
            self.solve_symbolic()
        
        self.t_eval = np.linspace(0, self.t_f, self.n_t)
        self.n_dp = partial(self.numeric_dp, **self.params)
        self.soln = solve_ivp(self.n_dp, [0, self.t_f], self.ics, t_eval=self.t_eval)
        return self.soln

    def simulate(self):
        assert self.N == 3

        if self.soln is None:
            self.solve_numeric()

        t, y = self.soln.t, self.soln.y
        q, p = np.vsplit(y, 2)[0], np.vsplit(y, 2)[1]

        x, y = [self.params['l'] * np.sin(q[0])], [- self.params['l'] * np.cos(q[0])]
        for i in range(1, 3):
            x.append(x[i - 1] + self.params['l'] * np.sin(q[i]))
            y.append(y[i - 1] - self.params['l'] * np.cos(q[i]))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))

        min_x, max_x = min([-3 * self.params['l'], np.min(x)]), max([3 * self.params['l'], np.max(x)])
        min_y, max_y = max([-3 * self.params['l'], np.min(y)]), max([self.params['l'], np.max(y)])
        ax1.set_xlim((1.15 * min_x, 1.15 * max_x))
        ax1.set_ylim((1.15 * min_y, 1.15 * max_y))

        ax1.set_aspect('equal')
        ax1.set_title('PyPendula-N3\nWritten by: Ethan Knox')
        ax1.set_xlabel(r'X [m]')
        ax1.set_ylabel(r'Y [m]')
        ax2.set_xlabel(r'$q$ [rad]')
        ax2.set_ylabel(r'$p$ [rad]/[s]')

        mass0, = ax1.plot([], [], lw=3, color='darkgray')
        mass1, = ax1.plot([], [], 'o', lw=2, color='red')
        mass2, = ax1.plot([], [], 'o', lw=2, color='blue')
        mass3, = ax1.plot([], [], 'o', lw=2, color='green')

        ax2.plot(q[0], p[0], lw=1.5, color='red', alpha=0.5)
        ax2.plot(q[1], p[1], lw=1.5, color='blue', alpha=0.5)
        ax2.plot(q[2], p[2], lw=1.5, color='green', alpha=0.5)
        point1, = ax2.plot([], [], 'o', lw=3, color='red')
        point2, = ax2.plot([], [], 'o', lw=3, color='blue')
        point3, = ax2.plot([], [], 'o', lw=3, color='green')

        def animate(i):
            mass0.set_data(
                [0, x[0][i], x[1][i], x[2][i]],
                [0, y[0][i], y[1][i], y[2][i]]
                )
            
            mass3.set_data([x[2][i]], [y[2][i]])
            mass2.set_data([x[1][i]], [y[1][i]])
            mass1.set_data([x[0][i]], [y[0][i]])
            point3.set_data([q[2][i]], [p[2][i]])
            point2.set_data([q[1][i]], [p[1][i]])
            point1.set_data([q[0][i]], [p[0][i]])
            return mass0, mass1, mass2, mass3, point1, point2, point3,
 
        dt = self.t_f / self.n_t

        anim = animation.FuncAnimation(fig, animate, len(self.t_eval), interval=dt * 1000, blit=True)
        anim.save(
        './resources/pypendula_n3_' + self.ics_tag + '.mp4',
        progress_callback = progress_bar
        )
        plt.close()
        return anim


    # def show_diagram(self):
    #     assert self.N == 3

    #     theta1, theta2, theta3, _, _, _ = self.ics
    #     l = self.params['l']

    #     fig, ax = plt.subplots(figsize=(10, 10))
    #     ax.set_xlim([-3.0 * l, 3.0 * l])
    #     ax.set_ylim([-3.5 * l, 3.0 * l])

    #     x0, y0 =                       0,                       0
    #     x1, y1 =      l * np.sin(theta1),    - l * np.cos(theta1)
    #     x2, y2 = x1 + l * np.sin(theta2), y1 - l * np.cos(theta2)
    #     x3, y3 = x2 + l * np.sin(theta3), y2 - l * np.cos(theta3)

    #     ax.vlines([0, x1, x2], [y1, y2, y3], [0, y1, y2], linestyles='dashed', colors='black', zorder=1)
    #     ax.hlines([y1, y2, y3], [x1, x1, x2], [0, x2, x3], linestyles='dashed', colors='black', zorder=1)

    #     # Rods
    #     ax.plot([x0, x1], [y0, y1], lw=3, color='darkgray')  # , lw=2 * r_l, color='blue',  zorder=4)
    #     ax.plot([x1, x2], [y1, y2], lw=3, color='darkgray')  # , lw=2 * r_l, color='red',   zorder=3)
    #     ax.plot([x2, x3], [y2, y3], lw=3, color='darkgray')  # , lw=2 * r_l, color='green', zorder=2)

    #     # Masses
    #     ax.add_patch(Circle([x1, y1], radius=l / 6, color='blue', label=r'$(x_1,y_1)$',  zorder=4))
    #     ax.text(x1, y1, r'$m_1$', zorder=10)

    #     ax.add_patch(Circle([x2, y2], radius=l / 6, color='red', label=r'$(x_2,y_2)$', zorder=3))
    #     ax.text(x2, y2, r'$m_2$', zorder=10)

    #     ax.add_patch(Circle([x3, y3], radius=l / 6, color='green', label=r'$(x_3,y_3)$', zorder=2))
    #     ax.text(x3, y3, r'$m_3$', zorder=10)

    #     # Pins
    #     ax.add_patch(Circle(             [x0, y0], radius=l / 12, color='black',  zorder=4))
        
    #     # plt.legend()

    #     return fig
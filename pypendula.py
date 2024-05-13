import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation

import sympy as sp
from sympy.matrices import Matrix
from sympy.physics.mechanics import dynamicsymbols, Lagrangian, LagrangesMethod, ReferenceFrame, Point, Particle
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from functools import partial
import dill


dill.settings['recurse'] = True
rng = np.random.default_rng()
plt.rcParams['animation.ffmpeg_path'] = './ffmpeg/ffmpeg.exe'

## Inspired by stackoverflow user greenstick's comment: 
## https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
def progress_bar (
        frame,                         # (Required): current frame (Int)
        total,                         # (Required): total frames (Int)
        prefix = 'Saving Animation:',  # (Optional): prefix string (Str)
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
            m=None,
            l=None,
            g=9.807,
            ics = None,
            alpha=3,
            beta=128,
            t_f=15,
            fps=60,
            ):
        
        if abs(alpha) < 1: alpha = 1
        if alpha < 0: alpha = -alpha
        if beta < 0: beta = -beta
        if l is None: l = 1. / N
        if m is None: m = 1. / N
        
        self.N = N
        self.m = m
        self.l = l
        self.g = g
        self.alpha = alpha
        self.beta = beta
        self.t_f = t_f
        self.fps = fps

        self.params = {
                'm' : self.m,
                'g' : self.g,
                'l' : self.l,
            }
        
        # # Default to None until appropriate calculations are made
        # self.potential = None
        # self.kinetic = None
        # self.lagrangian = None
        # self.hamiltonian = None
        # self.soln_hamiltonian = None
        # self.soln = None
        # self.t_eval = None
        self.numeric_dp = None
        self.numeric_hamiltonian = None
        # self.ics_tag = None

        if ics is None:
            self.ics = self.gen_ics()
        else:
            self.ics = ics

        self.ics_tag = 'ics=[' + ','.join((f'{ic:.4f}' for ic in self.ics)) + ']'

    def gen_ics(self):
        ics_q = rng.uniform(0, np.pi / self.alpha, size=self.N)
        ics_p = rng.uniform(0, np.pi / self.beta, size=self.N)
        return np.hstack([ics_q, ics_p])       

    def solve_symbolic(self):
        print("Solving Symbolic Problem... ", end='', flush=True)

        q = dynamicsymbols(f'q:{self.N}')
        dq = dynamicsymbols(f'q:{self.N}', level=1)
        p = dynamicsymbols(f'p:{self.N}')
        dp = dynamicsymbols(f'p:{self.N}', level=1)
        l, m, g, t = sp.symbols('l m g t')

        x, y = [l * sp.sin(q[0])], [- l * sp.cos(q[0])]
        for i in range(1, self.N):
            x.append(x[i - 1] + l * sp.sin(q[i]))
            y.append(y[i - 1] - l * sp.cos(q[i]))

        v_sqr = Matrix([_x.diff(t) ** 2 + _y.diff(t) ** 2 for _x,_y in zip(x, y)])
        x, y = Matrix(x), Matrix(y)
        
        potential = m * g * sum(y)
        kinetic = sp.Rational(1, 2) * m * sum(v_sqr)
        lagrangian = kinetic - potential
        lagranges_method = LagrangesMethod(lagrangian, q)
        euler_lagrange_eqns = lagranges_method.form_lagranges_equations()
        hamiltonian = sp.simplify(kinetic + potential).subs([(dq[_], p[_]) for _ in range(self.N)])
        eom = sp.simplify(lagranges_method.eom.subs([(dq[_], p[_]) for _ in range(self.N)]))
        symbolic_dp = Matrix(list(sp.solve(eom, *dp).values()))
        print("Done!")

        print("Caching Solution... ", end='', flush=True)
        self.numeric_dp = sp.utilities.lambdify([t, 
                                       [*q, *p], 
                                       m, g, l], 
                                      [*p, *symbolic_dp])
        dill.dump(self.numeric_dp, open(f"./cache/pypendula_cached_soln_n{self.N}", "wb"))

        self.numeric_hamiltonian = sp.utilities.lambdify([t, 
                                       [*q, *p], 
                                       m, g, l], 
                                      hamiltonian)
        dill.dump(self.numeric_hamiltonian, open(f"./cache/pypendula_cached_hamiltonian_n{self.N}", "wb"))
        print("Done!")

    def solve_numeric(self):
        try:
            self.numeric_dp = dill.load(open(f"./cache/pypendula_cached_soln_n{self.N}", "rb"))
            self.numeric_hamiltonian = dill.load(open(f"./cache/pypendula_cached_hamiltonian_n{self.N}", "rb"))
        except:
            self.solve_symbolic()

        print("Solving Numeric Problem... ", end='', flush=True)
        frames = self.t_f * self.fps
        self.t_eval = np.linspace(0, self.t_f, frames)
        self.n_dp = partial(self.numeric_dp, **self.params)
        self.n_hamiltonian = partial(self.numeric_hamiltonian, **self.params)
        self.soln = solve_ivp(self.n_dp, [0, self.t_f], self.ics, t_eval=self.t_eval, method='DOP853')
        
        self.t, self.soln_y = self.soln.t, self.soln.y
        self.energy = self.n_hamiltonian(self.t_eval, self.soln_y)
        self.percent_energy_loss = 100 * (self.energy - self.energy[0]) / self.energy[0]

        self.q, self.p = np.vsplit(self.soln_y, 2)[0], np.vsplit(self.soln_y, 2)[1]
        
        self.x, self.y = [self.params['l'] * np.sin(self.q[0])], [- self.params['l'] * np.cos(self.q[0])]
        for i in range(1, self.N):
            self.x.append(self.x[i - 1] + self.params['l'] * np.sin(self.q[i]))
            self.y.append(self.y[i - 1] - self.params['l'] * np.cos(self.q[i]))
        
        self.vx, self.vy = [self.params['l'] * self.p[0] * np.cos(self.q[0])], [self.params['l'] * self.p[0] * np.sin(self.q[0])]
        for i in range(1, self.N):
            self.vx.append(self.vx[i - 1] + self.params['l'] * self.p[i] * np.cos(self.q[i]))
            self.vy.append(self.vy[i - 1] + self.params['l'] * self.p[i] * np.sin(self.q[i]))
        
        df = pd.DataFrame({
            't': self.t,
            'Energy': self.energy,
            'PercentEnergyLoss': self.percent_energy_loss,
            'q1': self.q[0],
            'q2': self.q[1],
            'q3': self.q[2],
            'p1': self.p[0],
            'p2': self.p[1],
            'p3': self.p[2],
            'x1': self.x[0],
            'x2': self.x[1],
            'x3': self.x[2],
            'y1': self.y[0],
            'y2': self.y[1],
            'y3': self.y[2],
            'vx1': self.vx[0],
            'vx2': self.vx[1],
            'vx3': self.vx[2],
            'vy1': self.vy[0],
            'vy2': self.vy[1],
            'vy3': self.vy[2],
            })
        print("Done!")
        return df

    def simulate(self,
                 m=None,
                 g=None,
                 l=None
                 ):
        
        if self.numeric_dp or self.numeric_hamiltonian is None:
            self.solve_numeric()
        
        print("Setting Up Simulation... ", end='', flush=True)
        
        # t, y = self.soln.t, self.soln.y
        # q, p = np.vsplit(y, 2)[0], np.vsplit(y, 2)[1]

        # energy = self.soln_hamiltonian
        # energy_loss_percent = 100 * (energy - energy[0]) / energy[0]


        # x, y = [self.params['l'] * np.sin(q[0])], [- self.params['l'] * np.cos(q[0])]
        # for i in range(1, self.N):
        #     x.append(x[i - 1] + self.params['l'] * np.sin(q[i]))
        #     y.append(y[i - 1] - self.params['l'] * np.cos(q[i]))

        fig = plt.figure(layout="constrained", figsize=(19.2, 10.80))
        gs = GridSpec(2, 2, figure=fig)
        ax3 = fig.add_subplot(gs[1, 1])
        ax2 = fig.add_subplot(gs[0, 1])
        ax1 = fig.add_subplot(gs[:, 0])

        min_x, max_x = min([-self.N * self.params['l'], np.min(self.x)]), max([self.N * self.params['l'], np.max(self.x)])
        min_y, max_y = max([-self.N * self.params['l'], np.min(self.y)]), max([         self.params['l'], np.max(self.y)])
        ax1.set_xlim((1.15 * min_x, 1.15 * max_x))
        ax1.set_ylim((1.15 * min_y, 1.15 * max_y))

        ax1.set_aspect('equal')
        ax1.set_title('PyPendula-N3\nWritten by: Ethan Knox')
        ax1.set_xlabel(r'X [m]')
        ax1.set_ylabel(r'Y [m]')
        ax2.set_xlabel(r'$q$ [rad]')
        ax2.set_ylabel(r'$p$ [rad]/[s]')
        ax3.set_ylabel(r'Energy Loss [%]')
        ax3.set_xlabel(r'$t$ $[s]$')

        bbox_props = dict(boxstyle='round', alpha=0., facecolor='white')
        label = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, verticalalignment='top', bbox=bbox_props)
        pin, = ax1.plot(0, 0, 'o', markersize=6, color='black', zorder=10)
        mass0, = ax1.plot([], [], lw=3, color='darkgray')

        # masses, = [ax1.plot([], [], 'o', markersize=12, label=rf'$q_{i}(0)={self.ics[0]:.4f}, p_{i}(0)={self.ics[3]:.6f}$') for i in range(self.N)]
        # points, = [ax2.plot([], [], 'o', markersize=6) for i in range(self.N)]
        # for i in range(self.N):
        #     ax2.plot(self.q[i], self.p[i], lw=1.5, alpha=0.5)
            
        mass1, = ax1.plot([], [], 'o', markersize=12, color='red', label=rf'$q_1(0)={self.ics[0]:.4f}, p_1(0)={self.ics[3]:.6f}$')
        mass2, = ax1.plot([], [], 'o', markersize=12, color='blue', label=rf'$q_2(0)={self.ics[1]:.4f}, p_2(0)={self.ics[4]:.6f}$')
        mass3, = ax1.plot([], [], 'o', markersize=12, color='green', label=rf'$q_3(0)={self.ics[2]:.4f}, p_3(0)={self.ics[5]:.6f}$')
        ax2.plot(self.q[0], self.p[0], lw=1.5, color='red', alpha=0.5)
        ax2.plot(self.q[1], self.p[1], lw=1.5, color='blue', alpha=0.5)
        ax2.plot(self.q[2], self.p[2], lw=1.5, color='green', alpha=0.5)
        point1, = ax2.plot([], [], 'o', markersize=6, color='red')
        point2, = ax2.plot([], [], 'o', markersize=6, color='blue')
        point3, = ax2.plot([], [], 'o', markersize=6, color='green')

        ax1.legend() 

        ax3.plot(self.t_eval, self.percent_energy_loss, '-', lw=1.5, color='purple')
        ax3.axhline(y=0, xmin=self.t_eval[0], xmax=self.t_eval[-1], linestyle='--', color='black')
        energy_loss_plot, = ax3.plot([], [], 'o', markersize=6, color='purple', label='Total Energy')

        frames = self.t_f * self.fps
        dt = self.t_f / frames

        def animate(i):
            energy_loss_plot.set_data([self.t_eval[i]], [self.percent_energy_loss[i]])

            label_text = '\n'.join((
                rf"$m={self.params['m']:.3f}, l={self.params['l']:.3f}$",
                rf"$g={self.params['g']:.3f}, t={i * dt:.1f}$"
            ))   
            label.set_text(label_text)   

            # mass0.set_data(
            #     [0,].append([self.x[_][i] for _ in range(self.N)]),
            #     [0,].append([self.y[_][i] for _ in range(self.N)])
            #     )

            # for _, mass in enumerate(reversed(masses)):
            #     mass.set_data([self.x[_][i]], [self.y[_][i]])
            
            # for _, point in enumerate(reversed(points)):
            #     point.set_data([self.q[_][i]], [self.p[_][i]])

            # return mass0, *masses, *points, energy_loss_plot, label,

            mass0.set_data(
                [0, self.x[0][i], self.x[1][i], self.x[2][i]],
                [0, self.y[0][i], self.y[1][i], self.y[2][i]]
                )
            mass3.set_data([self.x[2][i]], [self.y[2][i]])
            mass2.set_data([self.x[1][i]], [self.y[1][i]])
            mass1.set_data([self.x[0][i]], [self.y[0][i]])
            point3.set_data([self.q[2][i]], [self.p[2][i]])
            point2.set_data([self.q[1][i]], [self.p[1][i]])
            point1.set_data([self.q[0][i]], [self.p[0][i]])
            return mass0, mass1, mass2, mass3, point1, point2, point3, energy_loss_plot, label,
 
        print("Done!")
        anim = animation.FuncAnimation(fig, animate, len(self.t_eval), interval=dt * 1000, blit=True)
        anim.save(
        f'./results/pypendula_n{self.N}_' + self.ics_tag + '.mp4',
        progress_callback = progress_bar
        )
        plt.close()
        return anim


def main():
    p = PyPendula()
    p.simulate()


if __name__ == "__main__":
    main()

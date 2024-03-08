import numpy as np
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
            t_f=10,
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
        
        # Default to None until appropriate calculations are made
        self.potential = None
        self.kinetic = None
        self.lagrangian = None
        self.hamiltonian = None
        self.soln_hamiltonian = None
        self.soln = None
        self.t_eval = None
        self.n_dp = None
        self.n_hamiltonian = None
        self.ics_tag = None

        self.set_ics(ics)

    def set_ics(self, ics=None):
        if ics is None:
            ics_q = rng.uniform(0, np.pi / self.alpha, size=self.N)
            ics_p = rng.uniform(0, np.pi / self.beta, size=self.N)
            self.ics = np.hstack([ics_q, ics_p])
        else:
            self.ics = ics
        
        self.ics_tag = 'ics=[' + ','.join((f'{ic:.4f}' for ic in self.ics)) + ']'

    def solve_symbolic(self):
        print("Solving Symbolic Problem... ", end='', flush=True)

        q = dynamicsymbols(f'q:{self.N}')
        dq = dynamicsymbols(f'q:{self.N}', level=1)
        p = dynamicsymbols(f'p:{self.N}')
        dp = dynamicsymbols(f'p:{self.N}', level=1)
        l, m, g, t = sp.symbols('l m g t')

        # Translating coordinates for convenience
        ############################################################################
        x, y = [l * sp.sin(q[0])], [- l * sp.cos(q[0])]                            #
        for i in range(1, self.N):                                                 #
            x.append(x[i - 1] + l * sp.sin(q[i]))                                  #
            y.append(y[i - 1] - l * sp.cos(q[i]))                                  #
                                                                                   #
        v_sqr = Matrix([_x.diff(t) ** 2 + _y.diff(t) ** 2 for _x,_y in zip(x, y)]) #
        x, y = Matrix(x), Matrix(y)                                                #
        ############################################################################

        self.potential = m * g * sum(y)
        self.kinetic = sp.Rational(1, 2) * m * sum(v_sqr)
        self.lagrangian = self.kinetic - self.potential
        self.lagranges_method = LagrangesMethod(self.lagrangian, q)
        self.euler_lagrange_eqns = self.lagranges_method.form_lagranges_equations()
        self.hamiltonian = sp.simplify(self.kinetic + self.potential).subs([(dq[_], p[_]) for _ in range(self.N)])
        self.eom = sp.simplify(self.lagranges_method.eom.subs([(dq[_], p[_]) for _ in range(self.N)]))
        self.symbolic_dp = Matrix(list(sp.solve(self.eom, *dp).values()))
        print("Done!")

        print("Caching Solution... ", end='', flush=True)
        self.numeric_dp = sp.utilities.lambdify([t, 
                                       [*q, *p], 
                                       m, g, l], 
                                      [*p, *self.symbolic_dp])
        dill.dump(self.numeric_dp, open(f"./cache/pypendula_cached_soln_n{self.N}", "wb"))

        self.numeric_hamiltonian = sp.utilities.lambdify([t, 
                                       [*q, *p], 
                                       m, g, l], 
                                      self.hamiltonian)
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
        self.soln_hamiltonian = self.n_hamiltonian(self.t_eval, self.soln.y)
        print("Done!")
        return self.soln

    def simulate(self):
        if self.soln is None:
            self.solve_numeric()
        
        print("Setting Up Simulation... ", end='', flush=True)
        
        t, y = self.soln.t, self.soln.y
        q, p = np.vsplit(y, 2)[0], np.vsplit(y, 2)[1]

        energy = self.soln_hamiltonian
        energy_loss_percent = 100 * (energy - energy[0]) / energy[0]
        # fit_a, fit_b = curve_fit(lambda t, a, b: a * t + b, t, energy_loss_percent)
        # energy_loss_percent_fit = fit_a * t + fit_b

        x, y = [self.params['l'] * np.sin(q[0])], [- self.params['l'] * np.cos(q[0])]
        for i in range(1, self.N):
            x.append(x[i - 1] + self.params['l'] * np.sin(q[i]))
            y.append(y[i - 1] - self.params['l'] * np.cos(q[i]))

        fig = plt.figure(layout="constrained", figsize=(19.2, 10.80))
        gs = GridSpec(2, 2, figure=fig)
        ax3 = fig.add_subplot(gs[1, 1])
        ax2 = fig.add_subplot(gs[0, 1])
        ax1 = fig.add_subplot(gs[:, 0])

        min_x, max_x = min([-self.N * self.params['l'], np.min(x)]), max([self.N * self.params['l'], np.max(x)])
        min_y, max_y = max([-self.N * self.params['l'], np.min(y)]), max([         self.params['l'], np.max(y)])
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

        if self.N == 3:
            mass1, = ax1.plot([], [], 'o', markersize=12, color='red', label=rf'$q_1(0)={self.ics[0]:.4f}, p_1(0)={self.ics[3]:.6f}$')
            mass2, = ax1.plot([], [], 'o', markersize=12, color='blue', label=rf'$q_2(0)={self.ics[1]:.4f}, p_2(0)={self.ics[4]:.6f}$')
            mass3, = ax1.plot([], [], 'o', markersize=12, color='green', label=rf'$q_3(0)={self.ics[2]:.4f}, p_3(0)={self.ics[5]:.6f}$')
            ax2.plot(q[0], p[0], lw=1.5, color='red', alpha=0.5)
            ax2.plot(q[1], p[1], lw=1.5, color='blue', alpha=0.5)
            ax2.plot(q[2], p[2], lw=1.5, color='green', alpha=0.5)
            point1, = ax2.plot([], [], 'o', markersize=6, color='red')
            point2, = ax2.plot([], [], 'o', markersize=6, color='blue')
            point3, = ax2.plot([], [], 'o', markersize=6, color='green')
        elif self.N == 2:
            mass1, = ax1.plot([], [], 'o', markersize=12, color='red', label=rf'$q_1(0)={self.ics[0]:.4f}, p_1(0)={self.ics[2]:.6f}$')
            mass2, = ax1.plot([], [], 'o', markersize=12, color='blue', label=rf'$q_2(0)={self.ics[1]:.4f}, p_2(0)={self.ics[3]:.6f}$')
            ax2.plot(q[0], p[0], lw=1.5, color='red', alpha=0.5)
            ax2.plot(q[1], p[1], lw=1.5, color='blue', alpha=0.5)
            point1, = ax2.plot([], [], 'o', markersize=6, color='red')
            point2, = ax2.plot([], [], 'o', markersize=6, color='blue')
        elif self.N == 1:
            mass1, = ax1.plot([], [], 'o', markersize=12, color='red', label=rf'$q_1(0)={self.ics[0]:.4f}, p_1(0)={self.ics[1]:.6f}$')
            ax2.plot(q[0], p[0], lw=1.5, color='red', alpha=0.5)
            point1, = ax2.plot([], [], 'o', markersize=6, color='red')


        ax1.legend() 

        ax3.plot(self.t_eval, energy_loss_percent, '-', lw=1.5, color='purple')
        ax3.axhline(y=0, xmin=self.t_eval[0], xmax=self.t_eval[-1], linestyle='--', color='black')
        energy_loss_plot, = ax3.plot([], [], 'o', markersize=6, color='purple', label='Total Energy')    

        frames = self.t_f * self.fps
        dt = self.t_f / frames

        def animate(i):

            energy_loss_plot.set_data([self.t_eval[i]], [energy_loss_percent[i]])

            
            if self.N == 3:
                label_text = '\n'.join((
                rf"$m_1={self.params['m']:.3f}, m_2={self.params['m']:.3f}, m_3={self.params['m']:.3f}$",
                rf"$l_1={self.params['l']:.3f}, l_2={self.params['l']:.3f}, l_3={self.params['l']:.3f}$",
                rf"$g={self.params['g']:.3f}, t={i * dt:.1f}$"
            ))             
                label.set_text(label_text)   
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
                return mass0, mass1, mass2, mass3, point1, point2, point3, energy_loss_plot, label,
            elif self.N == 2:
                label_text = '\n'.join((
                rf"$m_1={self.params['m']:.3f}, m_2={self.params['m']:.3f}$",
                rf"$l_1={self.params['l']:.3f}, l_2={self.params['l']:.3f}$",
                rf"$g={self.params['g']:.3f}, t={i * dt:.1f}$"
            ))             
                label.set_text(label_text)   
                mass0.set_data(
                [0, x[0][i], x[1][i]],
                [0, y[0][i], y[1][i]]
                )
                mass2.set_data([x[1][i]], [y[1][i]])
                mass1.set_data([x[0][i]], [y[0][i]])
                point2.set_data([q[1][i]], [p[1][i]])
                point1.set_data([q[0][i]], [p[0][i]])
                return mass0, mass1, mass2, point1, point2, energy_loss_plot, label,
            elif self.N == 1:
                label_text = '\n'.join((
                rf"$m_1={self.params['m']:.3f}$",
                rf"$l_1={self.params['l']:.3f}$",
                rf"$g={self.params['g']:.3f}, t={i * dt:.1f}$"
            ))             
                label.set_text(label_text)   
                mass0.set_data(
                [0, x[0][i]],
                [0, y[0][i]]
                )
                mass1.set_data([x[0][i]], [y[0][i]])
                point1.set_data([q[0][i]], [p[0][i]])
                return mass0, mass1, point1, energy_loss_plot, label,
 
        print("Done!")
        anim = animation.FuncAnimation(fig, animate, len(self.t_eval), interval=dt * 1000, blit=True)
        anim.save(
        f'./results/pypendula_n{self.N}_' + self.ics_tag + '.mp4',
        progress_callback = progress_bar
        )
        plt.close()
        return anim

    def preview(self):
        if self.soln is None:
            self.solve_numeric()
              
        t, y = self.soln.t, self.soln.y
        q, p = np.vsplit(y, 2)[0], np.vsplit(y, 2)[1]

        energy = self.soln_hamiltonian
        energy_loss_percent = 100 * (energy - energy[0]) / energy[0]

        x, y = [self.params['l'] * np.sin(q[0])], [- self.params['l'] * np.cos(q[0])]
        for i in range(1, self.N):
            x.append(x[i - 1] + self.params['l'] * np.sin(q[i]))
            y.append(y[i - 1] - self.params['l'] * np.cos(q[i]))

        fig = plt.figure(layout="constrained", figsize=(19.2, 10.80))
        gs = GridSpec(2, 2, figure=fig)
        ax3 = fig.add_subplot(gs[1, 1])
        ax2 = fig.add_subplot(gs[0, 1])
        ax1 = fig.add_subplot(gs[:, 0])

        min_x, max_x = min([-self.N * self.params['l'], np.min(x)]), max([self.N * self.params['l'], np.max(x)])
        min_y, max_y = max([-self.N * self.params['l'], np.min(y)]), max([         self.params['l'], np.max(y)])
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


        if self.N == 3:
            mass0, = ax1.plot([0, x[0][0], x[1][0], x[2][0]], [0, y[0][0], y[1][0], y[2][0]], lw=3, color='darkgray')
            mass1, = ax1.plot(x[0][0], y[0][0], 'o', markersize=12, color='red', label=rf'$q_1(0)={self.ics[0]:.4f}, p_1(0)={self.ics[3]:.6f}$')
            mass2, = ax1.plot(x[1][0], y[1][0], 'o', markersize=12, color='blue', label=rf'$q_2(0)={self.ics[1]:.4f}, p_2(0)={self.ics[4]:.6f}$')
            mass3, = ax1.plot(x[2][0], y[2][0], 'o', markersize=12, color='green', label=rf'$q_3(0)={self.ics[2]:.4f}, p_3(0)={self.ics[5]:.6f}$')
            ax2.plot(q[0], p[0], lw=1.5, color='red', alpha=0.5)
            ax2.plot(q[1], p[1], lw=1.5, color='blue', alpha=0.5)
            ax2.plot(q[2], p[2], lw=1.5, color='green', alpha=0.5)
        elif self.N == 2:
            mass0, = ax1.plot([0, x[0][0], x[1][0]], [0, y[0][0], y[1][0]], lw=3, color='darkgray')
            mass1, = ax1.plot(x[0][0], y[0][0], 'o', markersize=12, color='red', label=rf'$q_1(0)={self.ics[0]:.4f}, p_1(0)={self.ics[2]:.6f}$')
            mass2, = ax1.plot(x[1][0], y[1][0], 'o', markersize=12, color='blue', label=rf'$q_2(0)={self.ics[1]:.4f}, p_2(0)={self.ics[3]:.6f}$')
            ax2.plot(q[0], p[0], lw=1.5, color='red', alpha=0.5)
            ax2.plot(q[1], p[1], lw=1.5, color='blue', alpha=0.5)
        elif self.N == 1:
            mass0, = ax1.plot([0, x[0][0]], [0, y[0][0]], lw=3, color='darkgray')
            ax1.plot(x[0][0], y[0][0], 'o', markersize=12, color='red', label=rf'$q_1(0)={self.ics[0]:.4f}, p_1(0)={self.ics[1]:.6f}$')
            ax2.plot(q[0], p[0], lw=1.5, color='red', alpha=0.5)

        ax1.legend() 
        ax3.plot(self.t_eval, energy_loss_percent, '-', lw=1.5, color='purple')
        ax3.axhline(y=0, xmin=self.t_eval[0], xmax=self.t_eval[-1], linestyle='--', color='black')
        plt.savefig(f'./results/pypendula_n{self.N}_' + self.ics_tag + '_preview.png')
        return fig


def main():
    p = PyPendula()
    p.preview()
    p.simulate()


if __name__ == "__main__":
    main()

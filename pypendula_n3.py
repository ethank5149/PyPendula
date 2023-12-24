import numpy as np
import sympy as sp
from sympy.matrices import Matrix
from sympy.physics.mechanics import dynamicsymbols, LagrangesMethod
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
from functools import partial
from tqdm import trange
import dill


rng = np.random.default_rng()
dill.settings['recurse'] = True
plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg'          # Local
##                                      '/usr/bin/ffmpeg' # Linux
##                                      'C://ffmpeg'      # Windows


RAND_PARAMS = {
    'rand_q_param' : 1 / 3.,  # Ranges from (0.0, 1.0), damping the random initial angles about zero 
    'rand_p_param' : 1 / 128.,  # Has similar 'damping' effect on initial omegas, although unbounded
}
DEFAULT_PARAMS = {
    'l_0' : 1. / 3.,
    'l_1' : 1. / 3.,
    'l_2' : 1. / 3.,
    'm_0' : 1. / 3.,
    'm_1' : 1. / 3.,
    'm_2' : 1. / 3.,
    'g' : 9.80665,
}
DEFAULT_ICS = np.array([
    -np.pi/6, 
    -np.pi/3, 
    -np.pi/2,
     0, 
     0, 
     0
])


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

def gen_rand_ics(
    rand_q_param=RAND_PARAMS['rand_q_param'],
    rand_p_param=RAND_PARAMS['rand_p_param']
    ):
    return np.stack([
        rand_q_param * rng.uniform(low=-np.pi, high=np.pi, size=3),
        rand_p_param * rng.random(size=3),
        ]).flatten()

def solve_symbolic(log_level='info', N=3):
    if log_level == 'info' or 'debug': print('Solving symbolic problem... ', end='', flush=True)
    t, g = sp.symbols('t g')
    l = Matrix(sp.symbols(" ".join(f"l_{_}" for _ in range(N))))
    m = Matrix(sp.symbols(" ".join(f"m_{_}" for _ in range(N))))
    q = dynamicsymbols(" ".join(f"q_{_}" for _ in range(N)))
    p = dynamicsymbols(" ".join(f"q_{_}" for _ in range(N)), 1)
    a = dynamicsymbols(" ".join(f"q_{_}" for _ in range(N)), 2)
    
    x, y = [l[0] * sp.sin(q[0])], [- l[0] * sp.cos(q[0])]
    for i in range(1, N):
        x.append(x[i - 1] + l[i] * sp.sin(q[i]))
        y.append(y[i - 1] - l[i] * sp.cos(q[i]))

    # x1 =      l[0] * sp.sin(q[0])                                                        #
    # x2 = x1 + l[1] * sp.sin(q[1])                                                        #
    # x3 = x2 + l[2] * sp.sin(q[2])                                                        #

    # y1 =    - l[0] * sp.cos(q[0])                                                        #
    # y2 = y1 - l[1] * sp.cos(q[1])                                                        #
    # y3 = y2 - l[2] * sp.cos(q[2])                                                        #
    
    v_sqr = [_x.diff(t) ** 2 + _y.diff(t) ** 2 for _x,_y in zip(x, y)]
    # v1_sqr = x1.diff(t) ** 2 + y1.diff(t) ** 2
    # v2_sqr = x2.diff(t) ** 2 + y2.diff(t) ** 2
    # v3_sqr = x3.diff(t) ** 2 + y3.diff(t) ** 2

    V = g * (m[0] * y[0] + m[1] * y[1] + m[2] * y[2])
    # V = m[0] * g * y1 + m[1] * g * y2 + m[2] * g * y3
    T = m[0] * v_sqr[0] / 2 + m[1] * v_sqr[1] / 2 + m[2] * v_sqr[2] / 2
    # T = m[0] * v1_sqr / 2 + m[1] * v2_sqr / 2 + m[2] * v3_sqr / 2
    L = T - V
    Kinetic, Potential, H = T, V, T + V

    if log_level == 'debug': print("DEBUG: Forming Lagrange's Equations... ", end='', flush=True)
    LM = LagrangesMethod(L, q)
    EL = LM.form_lagranges_equations()
    if log_level == 'debug': print('Done!')
    if log_level == 'debug': print("DEBUG: Solving Lagrange's Equations for EOM... ", end='', flush=True)

    # sEL = [eqn.subs({
    #     q[0].diff(t) : p[0],
    #     q[1].diff(t) : p[1],
    #     q[2].diff(t) : p[2],
    #     p[0].diff(t) : a[0],
    #     p[1].diff(t) : a[1],
    #     p[2].diff(t) : a[2]
    #     }).simplify() for eqn in EL
    #     ]
    # EOM = sp.solve(sEL, a)

    EOM = sp.solve(EL, a)
    EOMsoln = Matrix([EOM[a[_]] for _ in range(len(a))])
    if log_level == 'info' or 'debug': print('Done!')

    if log_level == 'info' or 'debug': print('Caching solution... ', end='', flush=True)
    ODEsystem = sp.utilities.lambdify([t, [*q, *p], *l, *m, g], [*p, *EOMsoln])
    dill.dump(ODEsystem, open("./cache/pypendula_cached_soln", "wb"))
    Hamiltonian = sp.utilities.lambdify([t, [*q, *p], *l, *m, g], H)
    dill.dump(Hamiltonian, open("./cache/pypendula_cached_h", "wb"))
    Kinetic = sp.utilities.lambdify([t, [*q, *p], *l, *m, g], Kinetic)
    dill.dump(Kinetic, open("./cache/pypendula_cached_kinetic", "wb"))
    Potential = sp.utilities.lambdify([t, [*q, *p], *l, *m, g], Potential)
    dill.dump(Potential, open("./cache/pypendula_cached_potential", "wb"))
    if log_level == 'info' or 'debug': print('Done!')

    return ODEsystem, Hamiltonian, Kinetic, Potential

def solve_numeric(tf, fps, ics, params, log_level='info', N=3):
    if log_level == 'info': print('\nChecking for cached solutions... ', end='', flush=True)
    try:
        ODEsystem = dill.load(open("./cache/pypendula_cached_soln", "rb"))
        Hamiltonian = dill.load(open("./cache/pypendula_cached_h", "rb"))
        Kinetic = dill.load(open("./cache/pypendula_cached_kinetic", "rb"))
        Potential = dill.load(open("./cache/pypendula_cached_potential", "rb"))
        if log_level == 'info': print('Done! (Solutions found)')
    except:
        if log_level == 'info': print('Done! (One or more missing)')
        ##################################################################################
        if log_level == 'info': print('Solving symbolic problem from scratch... ', end='',           #
                          flush=True)
        ODEsystem, Hamiltonian, Kinetic, Potential = solve_symbolic(log_level=log_level, N=N)

    frames = tf * fps                                                                         #
    dt = tf / frames                                                                          #
    t_eval = np.linspace(0, tf, frames)                                                       #
    ode = partial(ODEsystem, **params)                                                        #
    sol = solve_ivp(ode, [0, tf], ics, t_eval=t_eval)                                         #

    q, p = np.split(sol.y, 2)
    l = [params['l_0'], params['l_1'], params['l_2']]
    m = [params['m_0'], params['m_1'], params['m_2']]
    g = params['g']

    # Translating coordinates for convenience                                                 #
    x, y = [l[0] * np.sin(q[0])], [- l[0] * np.cos(q[0])]
    for i in range(1, N):
        x.append(x[i - 1] + l[i] * np.sin(q[i]))
        y.append(y[i - 1] - l[i] * np.cos(q[i]))

    energy = Hamiltonian(t_eval, sol.y, *l, *m, g)
    kinetic = Kinetic(t_eval, sol.y, *l, *m, g)
    potential = Potential(t_eval, sol.y, *l, *m, g)
    if log_level == 'info': print('Done!')                                                                #
    return [x, y, q, p], [energy, kinetic, potential]

def simulate(tf, fps, ics, params, log_level='info'):
    [x, y, q, p], [energy, kinetic, potential] = solve_numeric(tf=tf, fps=fps, ics=ics, params=params, log_level=log_level)
    energy_loss_percent = 100 * (energy - energy[0]) / energy[0]
    if log_level == 'info': print('Creating animation... ', end='', flush=True)
    frames = tf * fps
    dt = tf / frames
    t_eval = np.linspace(0, tf, frames)
    fig = plt.figure(layout="constrained", figsize=(19.2, 10.80))
    gs = GridSpec(2, 2, figure=fig)
    ax3 = fig.add_subplot(gs[1, 1])
    ax2 = fig.add_subplot(gs[0, 1])
    ax1 = fig.add_subplot(gs[:, 0])

    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), max([DEFAULT_PARAMS['l_1'], np.max(y)])
    ax1.set_xlim((1.75 * min_x, 1.75 * max_x))
    ax1.set_ylim((1.15 * min_y, 1.15 * max_y))        
    ax1.set_aspect('equal')
    ax1.set_xlabel(r'X [m]')
    ax1.set_ylabel(r'Y [m]')
    ax1.set_title('PyPendula-N3\nWritten by: Ethan Knox')
    ax2.set_ylabel(r'Energy Loss [%]')
    ax2.set_xlabel(r'$t$ $[s]$')
    ax3.set_xlabel(r'$q$ [rad]')
    ax3.set_ylabel(r'$p$ [rad]/[s]')

    ax3.plot(q[0], p[0], lw=1.5, color='red', alpha=0.5)
    ax3.plot(q[1], p[1], lw=1.5, color='blue', alpha=0.5)
    ax3.plot(q[2], p[2], lw=1.5, color='green', alpha=0.5)
    particle1, = ax3.plot([], [], 'o', lw=3, color='red')
    particle2, = ax3.plot([], [], 'o', lw=3, color='blue')
    particle3, = ax3.plot([], [], 'o', lw=3, color='green')

    ax2.plot(t_eval, energy_loss_percent, '-', lw=1.5, color='purple')
    ax2.axhline(y=0, xmin=t_eval[0], xmax=t_eval[-1], linestyle='--', color='black')
    energy_loss_plot, = ax2.plot([], [], 'o', lw=3, color='purple', label='Total Energy')    
    negliblemass, = ax1.plot([], [], '-', lw=1.5, color='black')
    mass1, = ax1.plot([], [], 'o', lw=2, color='red')
    mass2, = ax1.plot([], [], 'o', lw=2, color='blue')
    mass3, = ax1.plot([], [], 'o', lw=2, color='green')
    ax2.grid()
    ax2.legend()
    ax3.grid()
    ics_tag = 'ics=[' + ','.join((f'{ic:.6f}' for ic in ics)) + ']'


    def animate(i):
        energy_loss_plot.set_data([t_eval[i]], [energy_loss_percent[i]])
        negliblemass.set_data(
            [0, x[0][i], x[1][i], x[2][i]],
            [0, y[0][i], y[1][i], y[2][i]]
        )
        mass3.set_data([x[2][i]], [y[2][i]])
        mass2.set_data([x[1][i]], [y[1][i]])
        mass1.set_data([x[0][i]], [y[0][i]])
        particle3.set_data([q[2][i]], [p[2][i]])
        particle2.set_data([q[1][i]], [p[1][i]])
        particle1.set_data([q[0][i]], [p[0][i]])
        return mass1, mass2, mass3, negliblemass, energy_loss_plot, particle1, particle2, particle3,
    
    
    anim = animation.FuncAnimation(fig, animate, len(t_eval), interval=dt * 1000)
    anim.save(
        './resources/pypendula_n3_' + ics_tag + '.mp4',
        progress_callback = progress_bar,
        metadata=dict(
            title='PyPendula',
            artist='Ethan Knox',
            comment=ics_tag
            )
        )
    plt.close()
    if log_level == 'info': print('Simulation Complete!')
    return None

def main(
        tf=5, 
        fps=120,  # 60-120 yield acceptable results
        params=DEFAULT_PARAMS, 
        ics=DEFAULT_ICS, 
        runs=3
        ):
    for run in trange(runs):
        ics = gen_rand_ics()
        simulate(tf=tf, fps=fps, ics=ics, params=params, log_level=None)
#     if log_level == 'info': print('\nChecking for cached EOM solution... ', end='', flush=True)
#     try:
#         ODEsystem = dill.load(open("./pypendula_cached_soln", "rb"))
#         if log_level == 'info': print('Done! (Solution found)')
#     except:
#         if log_level == 'info': print('Done! (None found)')
#         ##################################################################################
#         if log_level == 'info': print('Solving symbolic problem from scratch... ', end='',           #
#                           flush=True)                                                    #
#         t, l1, l2, l3, m1, m2, m3, g = sp.symbols('t l1 l2 l3 m1 m2 m3 g')               #
#         q1, q2, q3 = dynamicsymbols('q_1 q_2 q_3')                                       #
#         q1d, q2d, q3d = dynamicsymbols('q_1 q_2 q_3', 1)                                 #
#         q1dd, q2dd, q3dd = dynamicsymbols('q_1 q_2 q_3', 2)                              #
#         x1 =      l1 * sp.sin(q1)                                                        #
#         y1 =    - l1 * sp.cos(q1)                                                        #
#         x2 = x1 + l2 * sp.sin(q2)                                                        #
#         y2 = y1 - l2 * sp.cos(q2)                                                        #
#         x3 = x2 + l3 * sp.sin(q3)                                                        #
#         y3 = y2 - l3 * sp.cos(q3)                                                        #
#         v1_sqr = x1.diff(t) ** 2 + y1.diff(t) ** 2                                       #
#         v2_sqr = x2.diff(t) ** 2 + y2.diff(t) ** 2                                       #
#         v3_sqr = x3.diff(t) ** 2 + y3.diff(t) ** 2                                       #
#         V = m1 * g * y1 + m2 * g * y2 + m3 * g * y3                                      #
#         T = m1 * v1_sqr / 2 + m2 * v2_sqr / 2 + m3 * v3_sqr / 2                          #
#         L = T - V                                                                        #
#         LM = LagrangesMethod(L, [q1, q2, q3])                                            #
#         EL = LM.form_lagranges_equations()                                               #
#         sEL = [eqn.subs({                                                                #
#             q1.diff(t) : q1d,                                                            #
#             q2.diff(t) : q2d,                                                            #
#             q3.diff(t) : q3d,                                                            #
#             q1d.diff(t) : q1dd,                                                          #
#             q2d.diff(t) : q2dd,                                                          #
#             q3d.diff(t) : q3dd                                                           #
#             }).simplify() for eqn in EL                                                  #
#             ]                                                                            #
#         EOM = sp.solve(sEL, [q1dd, q2dd, q3dd])                                          #
#         if log_level == 'info': print('Done!')                                                       #
# ##########################################################################################

# ##########################################################################################
#         if log_level == 'info': print('Caching for later... ',                                       #
#               end='', flush=True)                                                        #
#         ODEsystem = sp.utilities.lambdify(                                               #
#             [t, [q1, q1d, q2, q2d, q3, q3d], l1, l2, l3, m1, m2, m3, g],                 #
#             [q1d, EOM[q1dd], q2d, EOM[q2dd], q3d, EOM[q3dd]]                             #
#             )                                                                            #
#         dill.dump(ODEsystem, open("pypendula_cached_soln", "wb"))                        #
#         if log_level == 'info': print('Done!')                                                       #
# ##########################################################################################

# ###############################################################################################
#     if log_level == 'info': print('Solving numerical EOM... ', end='', flush=True)                        #
#     frames = tf * fps                                                                         #
#     dt = tf / frames                                                                          #
#     t_eval = np.linspace(0, tf, frames)                                                       #
#                                                                                               #
#     ode = partial(ODEsystem, **params)                                                        #
#     sol = solve_ivp(ode, [0, tf], ics, t_eval=t_eval)                                         #
#                                                                                               #
#     # Translating coordinates for convenience                                                 #
#     q1, p1, q2, p2, q3, p3 = sol.y                                                            #
#     l1, l2, l3 = params['l1'], params['l2'], params['l3']                                     #
#     x1 =      l1 * np.sin(q1)                                                                 #
#     y1 =    - l1 * np.cos(q1)                                                                 #
#     x2 = x1 + l2 * np.sin(q2)                                                                 #
#     y2 = y1 - l2 * np.cos(q2)                                                                 #
#     x3 = x2 + l3 * np.sin(q3)                                                                 #
#     y3 = y2 - l3 * np.cos(q3)                                                                 #
#     if log_level == 'info': print('Done!')                                                                #
# ###############################################################################################

# ###############################################################################################
#     if animate:  # Pass in False to just save results to file, for example                    #
#         if log_level == 'info': print('Creating animation... ', end='\n', flush=True)                     #
#         fig, (ax1, ax2) = plt.subplots(1, 2, squeeze=True, figsize=(19.20,10.80))             #
#         fig.suptitle('PyPendula-N3\nWritten by: Ethan Knox')                                  #
#         # ax1.set_title("")                                                                   #
#         ax1.set_aspect('equal')                                                               #
#         ax1.set_xlim((-1.15 * (l1 + l2 + l3), 1.15 * (l1 + l2 + l3)))                         #
#         ax1.set_ylim((-1.15 * (l1 + l2 + l3), 1.15 * (l1 + l2 + l3)))                         #
#         ax1.set_xlabel('X [m]')                                                               #
#         ax1.set_ylabel('Y [m]')                                                               #
#         ax2.set_xlabel(r'$q$ [rad]')                                                          #
#         ax2.set_ylabel(r'$p$ [rad]/[s]')                                                      #
#         # ax2.yaxis.tick_right()                                                              #
#                                                                                               #
#         ax2.plot(q1, p1, lw=1.5, color='red', alpha=0.5)                                      #
#         ax2.plot(q2, p2, lw=1.5, color='blue', alpha=0.5)                                     #
#         ax2.plot(q3, p3, lw=1.5, color='green', alpha=0.5)                                    #
#                                                                                               #
#         bbox_props = dict(boxstyle='round', alpha=0., facecolor='white')                      #
#         label = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,                             #
#                          verticalalignment='top', bbox=bbox_props)                            #
#         negliblemass, = ax1.plot([], [], '-', lw=1.5, color='black')                          #
#         mass1, = ax1.plot([], [], 'o', lw=2, color='red',                                     #
#                           label=rf'$q_1(0)={ics[0]:.6f}, p_1(0)={ics[1]:.6f}$')               #
#         mass2, = ax1.plot([], [], 'o', lw=2, color='blue',                                    #
#                           label=rf'$q_2(0)={ics[2]:.6f}, p_2(0)={ics[3]:.6f}$')               #
#         mass3, = ax1.plot([], [], 'o', lw=2, color='green',                                   #
#                           label=rf'$q_3(0)={ics[4]:.6f}, p_3(0)={ics[5]:.6f}$')               #
#                                                                                               #
#         point1, = ax2.plot([], [], 'o', lw=3, color='red')                                    #
#         point2, = ax2.plot([], [], 'o', lw=3, color='blue')                                   #
#         point3, = ax2.plot([], [], 'o', lw=3, color='green')                                  #
#         ax1.legend()                                                                          #        
#         ics_tag = 'ics=[' + ','.join((f'{ic:.6f}' for ic in ics)) + ']'                       #
#                                                                                               #
#                                                                                               #
#         def animate(i):                                                                       #
#             label_text = '\n'.join((                                                          #
#                 rf"$m_1={params['m1']:.3f}, m_2={params['m2']:.3f}, m_3={params['m3']:.3f}$", #
#                 rf"$l_1={params['l1']:.3f}, l_2={params['l2']:.3f}, l_3={params['l3']:.3f}$", #
#                 rf"$g={params['g']:.3f}, t={i * dt:.1f}$"                                     #
#             ))                                                                                #
#                                                                                               #
#             point3.set_data([q3[i]], [p3[i]])                                                 #
#             point2.set_data([q2[i]], [p2[i]])                                                 #
#             point1.set_data([q1[i]], [p1[i]])                                                 #
#             label.set_text(label_text)                                                        #
#                                                                                               #
#             negliblemass.set_data(                                                            #
#                 [0, x1[i], x2[i], x3[i]],                                                     #
#                 [0, y1[i], y2[i], y3[i]]                                                      #
#             )                                                                                 #
#             mass3.set_data([x3[i]], [y3[i]])                                                  #
#             mass2.set_data([x2[i]], [y2[i]])                                                  #
#             mass1.set_data([x1[i]], [y1[i]])                                                  #
#             return point1, mass1, point2, mass2, point3, mass3, negliblemass, label,          #
#                                                                                               #
#                                                                                               #
#         anim = animation.FuncAnimation(fig, animate, len(t_eval), interval=dt * 1000)         #
#         anim.save(                                                                            #
#             filepath + '/pypendula_n3_' + ics_tag + '.mp4',                                   #
#             progress_callback = progress_bar,                                                 #
#             metadata=dict(                                                                    #
#                 title='PyPendula',                                                            #
#                 artist='Ethan Knox',                                                          #
#                 comment=ics_tag                                                               #
#                 )                                                                             #
#             )                                                                                 #
#         plt.close()                                                                           #
#         if log_level == 'info': print('Done!')                                                            #
# ###############################################################################################
#     return sol, [x1, y1, x2, y2, x3, y3]
    return None


if __name__ == "__main__":
    main()

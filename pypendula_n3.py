import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
from functools import partial
import dill


rng = np.random.default_rng()
dill.settings['recurse'] = True
plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg'          # Local
##                                      '/usr/bin/ffmpeg' # Linux
##                                      'C://ffmpeg'      # Windows

RAND_PARAMS = {
    'rand_q_param' : 1 / 3.,  # Ranges from (0.0, 1.0), damping the random initial angles about zero 
    'rand_p_param' : 1 / 12.,  # Has similar 'damping' effect on initial omegas, although unbounded
}
DEFAULT_PARAMS = {
    'l1' : 1. / 3.,
    'l2' : 1. / 3.,
    'l3' : 1. / 3.,
    'm1' : 1. / 3.,
    'm2' : 1. / 3.,
    'm3' : 1. / 3.,
    'g' : 9.80665,
}
DEFAULT_ICS = np.array([
    -np.pi / 6, 
     np.pi / 16, 
    -np.pi / 5, 
     np.pi / 16, 
    -np.pi / 4, 
     np.pi / 16
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
    return np.asarray([
        rand_q_param * np.pi * (2 * rng.random() - 1),
        rand_p_param * np.pi * (2 * rng.random() - 1),
        rand_q_param * np.pi * (2 * rng.random() - 1),
        rand_p_param * np.pi * (2 * rng.random() - 1),
        rand_q_param * np.pi * (2 * rng.random() - 1),
        rand_p_param * np.pi * (2 * rng.random() - 1),
    ])
                                
def main(params=DEFAULT_PARAMS, ics=DEFAULT_ICS, tf=10, fps=60, run=0, animate=True, verbose=True):
    if verbose: print('\nChecking for cached EOM solution... ', end='', flush=True)
    try:
        ode3 = dill.load(open("./pypendula_n3", "rb"))
        if verbose: print('Done! (Solution found)')
    except:
        if verbose: print('Done! (None found)')
        ##################################################################################
        if verbose: print('Solving symbolic problem from scratch... ', end='', flush=True)

        t, l1, l2, l3, m1, m2, m3, g = sp.symbols('t l1 l2 l3 m1 m2 m3 g')               #
        q1, p1, a1 = sp.Function('q_1')(t), sp.Function('p_1')(t), sp.Function('a_1')(t) #
        q2, p2, a2 = sp.Function('q_2')(t), sp.Function('p_2')(t), sp.Function('a_2')(t) #
        q3, p3, a3 = sp.Function('q_3')(t), sp.Function('p_3')(t), sp.Function('a_3')(t) #
        x1 =      l1 * sp.sin(q1)                                                        #
        y1 =    - l1 * sp.cos(q1)                                                        #
        x2 = x1 + l2 * sp.sin(q2)                                                        #
        y2 = y1 - l2 * sp.cos(q2)                                                        #
        x3 = x2 + l3 * sp.sin(q3)                                                        #
        y3 = y2 - l3 * sp.cos(q3)                                                        #
        v1_sqr = x1.diff(t) ** 2 + y1.diff(t) ** 2                                       #
        v2_sqr = x2.diff(t) ** 2 + y2.diff(t) ** 2                                       #
        v3_sqr = x3.diff(t) ** 2 + y3.diff(t) ** 2                                       #
        V = m1 * g * y1 + m2 * g * y2 + m3 * g * y3                                      #
        T = m1 * v1_sqr / 2 + m2 * v2_sqr / 2 + m3 * v3_sqr / 2                          #
        L = T - V                                                                        #
        L = L.subs({                                                                     #
            q1.diff(t) : p1,                                                             #
            q2.diff(t) : p2,                                                             #
            q3.diff(t) : p3,                                                             #
            p1.diff(t) : a1,                                                             #
            p2.diff(t) : a2,                                                             #
            p3.diff(t) : a3                                                              #
        }).simplify()                                                                    #
                                                                                         #
        ELeqns = [  # Euler-Lagrange Equations                                           #
            L.diff(q) - L.diff(p).diff(t) for                                            #
            q,p in zip([q1, q2, q3], [p1, p2, p3])                                       #
        ]                                                                                #
        simplifiedEL = [                                                                 #
            ELeqn.subs({                                                                 #
                q1.diff(t) : p1,                                                         #
                q2.diff(t) : p2,                                                         #
                q3.diff(t) : p3,                                                         #
                p1.diff(t) : a1,                                                         #
                p2.diff(t) : a2,                                                         #
                p3.diff(t) : a3                                                          #
            }).simplify() for ELeqn in ELeqns                                            #
        ]                                                                                #
        n3system = sp.solve(simplifiedEL, a1, a2, a3)                                    #
        if verbose: print('Done!')                                                                   #
##########################################################################################

##########################################################################################
        if verbose: print('Caching for later... ',                                                   #
              end='', flush=True)                                                        #
        ode3 = sp.utilities.lambdify(                                                    #
            [t, [q1, p1, q2, p2, q3, p3], l1, l2, l3, m1, m2, m3, g],                    #
            [p1, n3system[a1], p2, n3system[a2], p3, n3system[a3]]                       #
        )                                                                                #
        dill.dump(ode3, open("pypendula_n3", "wb"))                                      #
        if verbose: print('Done!')                                                                   #
##########################################################################################

###############################################################################################
    if verbose: print('Solving numerical EOM... ', end='', flush=True)                                    #
    frames = tf * fps                                                                         #
    dt = tf / frames                                                                          #
    t_eval = np.linspace(0, tf, frames)                                                       #
                                                                                              #
    ode = partial(ode3, **params)                                                             #
    sol = solve_ivp(ode, [0, tf], ics, t_eval=t_eval)                                         #
                                                                                              #
    # Translating coordinates for convenience                                                 #
    q1, p1, q2, p2, q3, p3 = sol.y                                                            #
    l1, l2, l3 = params['l1'], params['l2'], params['l3']                                     #
    x1 =      l1 * np.sin(q1)                                                                 #
    y1 =    - l1 * np.cos(q1)                                                                 #
    x2 = x1 + l2 * np.sin(q2)                                                                 #
    y2 = y1 - l2 * np.cos(q2)                                                                 #
    x3 = x2 + l3 * np.sin(q3)                                                                 #
    y3 = y2 - l3 * np.cos(q3)                                                                 #
    if verbose: print('Done!')                                                                            #
###############################################################################################

###############################################################################################
    if animate:  # Pass in False to just save results to file, for example                    #
        if verbose: print('Creating animation... ', end='\n', flush=True)                                 #
        fig, (ax1, ax2) = plt.subplots(1, 2, squeeze=True, figsize=(19.20,10.80))             #
        fig.suptitle('PyPendula-N3\nWritten by: Ethan Knox')                                  #
        # ax1.set_title("")                                                                   #
        ax1.set_aspect('equal')                                                               #
        ax1.set_xlim((-1.15 * (l1 + l2 + l3), 1.15 * (l1 + l2 + l3)))                         #
        ax1.set_ylim((-1.15 * (l1 + l2 + l3), 1.15 * (l1 + l2 + l3)))                         #
        ax1.set_xlabel('X [m]')                                                               #
        ax1.set_ylabel('Y [m]')                                                               #
        ax2.set_xlabel(r'$q$ [rad]')                                                          #
        ax2.set_ylabel(r'$p$ [rad]/[s]')                                                      #
        # ax2.yaxis.tick_right()                                                              #
                                                                                              #
        ax2.plot(q1, p1, lw=1.5, color='red', alpha=0.5)                                      #
        ax2.plot(q2, p2, lw=1.5, color='blue', alpha=0.5)                                     #
        ax2.plot(q3, p3, lw=1.5, color='green', alpha=0.5)                                    #
                                                                                              #
        bbox_props = dict(boxstyle='round', alpha=0., facecolor='white')                      #
        label = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,                             #
                         verticalalignment='top', bbox=bbox_props)                            #
        negliblemass, = ax1.plot([], [], '-', lw=1.5, color='black')                          #
        mass1, = ax1.plot([], [], 'o', lw=2, color='red',                                     #
                          label=rf'$q_1(0)={ics[0]:.6f}, p_1(0)={ics[1]:.6f}$')               #
        mass2, = ax1.plot([], [], 'o', lw=2, color='blue',                                    #
                          label=rf'$q_2(0)={ics[2]:.6f}, p_2(0)={ics[3]:.6f}$')               #
        mass3, = ax1.plot([], [], 'o', lw=2, color='green',                                   #
                          label=rf'$q_3(0)={ics[4]:.6f}, p_3(0)={ics[5]:.6f}$')               #
                                                                                              #
        point1, = ax2.plot([], [], 'o', lw=3, color='red')                                    #
        point2, = ax2.plot([], [], 'o', lw=3, color='blue')                                   #
        point3, = ax2.plot([], [], 'o', lw=3, color='green')                                  #
        ax1.legend()                                                                          #
                                                                                              #
                                                                                              #
        def animate(i):                                                                       #
            label_text = '\n'.join((                                                          #
                rf"$m_1={params['m1']:.3f}, m_2={params['m2']:.3f}, m_3={params['m3']:.3f}$", #
                rf"$l_1={params['l1']:.3f}, l_2={params['l2']:.3f}, l_3={params['l3']:.3f}$", #
                rf"$g={params['g']:.3f}, t={i * dt:.1f}$"                                     #
            ))                                                                                #
                                                                                              #
            point3.set_data([q3[i]], [p3[i]])                                                 #
            point2.set_data([q2[i]], [p2[i]])                                                 #
            point1.set_data([q1[i]], [p1[i]])                                                 #
            label.set_text(label_text)                                                        #
                                                                                              #
            negliblemass.set_data(                                                            #
                [0, x1[i], x2[i], x3[i]],                                                     #
                [0, y1[i], y2[i], y3[i]]                                                      #
            )                                                                                 #
            mass3.set_data([x3[i]], [y3[i]])                                                  #
            mass2.set_data([x2[i]], [y2[i]])                                                  #
            mass1.set_data([x1[i]], [y1[i]])                                                  #
            return point1, mass1, point2, mass2, point3, mass3, negliblemass, label,          #
                                                                                              #
                                                                                              #
        anim = animation.FuncAnimation(fig, animate, len(t_eval), interval=dt * 1000)         #
        anim.save(                                                                       #
            # f'./resources/pypendula_n3_{str(run).zfill(4)}.mp4',
            './resources/pypendula_n3_ics=[' + ','.join((f'{ic:.6f}' for ic in ics)) + '].mp4',
            progress_callback = progress_bar,
            metadata=dict(
                title='PyPendula', 
                artist='Ethan Knox',
                comment="ics=[" + ','.join((f'{ic:.6f}' for ic in ics)) + "]"
                )
            )                                                                                 #
        plt.close()                                                                           #
        if verbose: print('Done!')                                                                        #
###############################################################################################
    return sol, [x1, y1, x2, y2, x3, y3]


if __name__ == "__main__":
    main()
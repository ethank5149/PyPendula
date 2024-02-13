import sympy as sp
from sympy.matrices import Matrix
from sympy.physics.mechanics import dynamicsymbols, Lagrangian, LagrangesMethod, ReferenceFrame, Point, Particle
import dill

dill.settings['recurse'] = True

class PyPendula:
    def __init__(self, N = 3, gravity=10, simplify=True, lambdify=False):
        self.N = N
        self.gravity = 10

        q = dynamicsymbols(f'q:{self.N}') if self.N > 1 else [dynamicsymbols(f'q'),]
        p = dynamicsymbols(f'q:{self.N}', level=1) if self.N > 1 else [dynamicsymbols(f'q', level=1),]
        l, m, g, t = sp.symbols('l, m, g, t')

        # Compose World Frame
        I = ReferenceFrame('I')
        O = Point('O')
        O.set_vel(I, 0)

        points = [O.locatenew('P_0', l * sp.sin(q[0]) * I.x - l * sp.cos(q[0]) * I.y),]
        points[0].set_vel(I, points[0].pos_from(O).dt(I))
        particles = [Particle('p_0', points[0], m),]
        particles[0].potential_energy = m * g * (-l * sp.cos(q[0]))

        # Append additional masses and apply properties
        for _ in range(1, N):
            points.append(O.locatenew(f'P_{_}', points[_ - 1].pos_from(O) + l * sp.sin(q[_]) * I.x - l * sp.cos(q[_]) * I.y))
            points[_].set_vel(I, points[_].pos_from(O).dt(I))
            particles.append(Particle(f'p_{_}', points[_], m))
            particles[_].potential_energy = m * g * (-l * sp.cos(q[_]))

        # Calculate the lagrangian, and form the equations of motion
        self.Lagrangian = Lagrangian(I, *particles)
        self.LagrangesMethod = LagrangesMethod(self.Lagrangian, q, frame=I)
        self.EulerLagrangeEqns = self.LagrangesMethod.form_lagranges_equations()
        self.eom = self.LagrangesMethod.eom
        self.ddq = self.solve_eom()
        
        if lambdify:
            self.substitute_params()
            self.simplify()
            self.lambdify()
        elif simplify:
            self.simplify()

    def simplify(self):
        self.eom = sp.simplify(self.eom)
        self.ddq = sp.simplify(self.ddq)

    def substitute_params(self):
        self.ddq = self.ddq.subs([
             (sp.symbols('l'), 1 / self.N), 
             (sp.symbols('m'), 1 / self.N),
             (sp.symbols('g'), self.gravity),
             ])

    def solve_eom(self):
        dp = dynamicsymbols(f'q:{self.N}', level=2) if self.N > 1 else [dynamicsymbols(f'q', level=2),]
        return Matrix([                                                  #
            sp.solve(self.eom[_], dp[_])[0] for _ in range(self.N)                         #
        ])
        
    def lambdify(self):
        q = dynamicsymbols(f'q:{self.N}') if self.N > 1 else [dynamicsymbols(f'q'),]
        p = dynamicsymbols(f'q:{self.N}', level=1) if self.N > 1 else [dynamicsymbols(f'q', level=1),]
        t = sp.symbols('l, m, g, t')

        ODEsystem = sp.utilities.lambdify([t, [*q, *p]], [*p, *self.ddq])
        dill.dump(ODEsystem, open(f"./cache/pypendula_cached_soln_n{self.N}", "wb")) 

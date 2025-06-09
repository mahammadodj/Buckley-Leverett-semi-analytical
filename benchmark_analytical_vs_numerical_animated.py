import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sympy as sp
import copy
from scipy.sparse import diags
import warnings
warnings.filterwarnings('ignore')

#rock properties
k = 100
phi = 0.2

#grid properties
N = 200
h = 10
L = 1000
dx = L/N
area = 2000
dt = 0.1

#fluid properties
Swcon, Swcrit = 0.2, 0.2
Soirw, Sorw = 0.4, 0.4
ko_rw = 0.3
ko_ro = 0.8
Nw = 2
No = 2
Qwi = 1000

#Buckley Leveret assumes fluid is incompressible
cw = 0
co = 0

Bw, Bo = 1, 1
muw , muo = 0.1, 10 #cp
cf = 0
betaw = np.zeros((N, N))
np.fill_diagonal(betaw, Bw)
Bw_ = np.zeros((N, N))
np.fill_diagonal(Bw_, Bw)
betao = np.zeros((N, N))
np.fill_diagonal(betao, Bo)
cw_ = np.zeros((N, N))
np.fill_diagonal(cw_, cw)

#well properties
Pi = np.full((N,1), 1000)
Sw = np.full((N,1), 0.2)
center_distances = [dx/2 + i*dx for i in range(N)]

#Numerical solution
class BuckleyLeverett_numerical:
    def __init__(self, Sw, Swcrit, Swcon, Soirw, Sorw, ko_rw, ko_ro, Nw, No, N):
        self.N = N
        self.Sw = Sw
        self.Swcrit = Swcrit
        self.Swcon = Swcon
        self.Soirw = Soirw
        self.Sorw = Sorw
        self.Nw = Nw
        self.No = No
        self.ko_rw = ko_rw
        self.ko_ro = ko_ro
    def calculate_relative_permeability(self):
        S = (self.Sw - self.Swcon) / (1 - self.Swcon - self.Swcrit)
        k_rw = 0.2 * S**3
        k_ro = (1 - S)**3
        return k_rw, k_ro
    
    def compute_T_matrix(self, k, k_rf, dx, area, muf, Bf, P, conversion_factor=None):

        T_scalar = (area*k) / (muf * dx*Bf)
        transmissibility = np.zeros((self.N, self.N))

        for i in range(self.N - 1):
            permeability_factor = k_rf[i+1] * T_scalar if P[i+1] > P[i] else k_rf[i] * T_scalar
            transmissibility[i, i] += permeability_factor
            transmissibility[i, i+1] -= permeability_factor
            transmissibility[i+1, i] -= permeability_factor
            transmissibility[i+1, i+1] += permeability_factor
        if conversion_factor is not None:
            return transmissibility/conversion_factor
        return transmissibility
    
    def compute_betaw(self, Pd, lambda_, cw):
        Se = (self.Sw-self.Swcon)/(1-self.Swcon-self.Soirw)
        Pc_prime = (-Pd/lambda_)*(Se**(-(1+lambda_)/lambda_))
        betaw_array = np.zeros((self.N, self.N))
        betaw = 1/(1-self.Sw*cw*Pc_prime)
        np.fill_diagonal(betaw_array, betaw)
        return betaw_array
    
    def compute_total_compressibility_matrix(self, cf, cw, co):
        ct = (cf + cw*self.Sw + co*(1-self.Sw))
        ct_matrix = np.zeros((self.N, self.N))
        np.fill_diagonal(ct_matrix, ct.flatten())
        return ct_matrix
    
    def compute_A_matrix(self, dx, area, phi, dt, N):
        A = dx*area*phi/dt
        A_matrix = np.diag(np.full(N, A))
        return A_matrix
    
    def compute_J(self, kx, ky, x, y, z, k_rf, muf, rw, s, block_number, conversion_factor=None):
        k2 = np.array([ky])
        k1 = np.array([kx])
        w1 = x
        w2 = y
        w3 = z
        r_eq = 0.28*(np.sqrt(np.sqrt(k2/k1)*w1**2 + np.sqrt(k1/k2)*w2**2))/((k2/k1)**0.25+(k1/k2)**0.25)
        J = 2*np.pi*w3*k_rf[block_number]*np.sqrt(k1[block_number]*k2[block_number])/(muf*(np.log(r_eq/rw)+s))
        J_matrix = np.zeros((self.N, self.N))
        J_matrix[block_number][block_number] = J
        if conversion_factor is not None:
            return J_matrix/conversion_factor
        return J_matrix
    
    def compute_Q(self, Qwi, k_rw, k_ro, muw, muo, Bw, Bo):
        Qw = np.zeros((self.N,1))
        Qo = np.copy(Qw)
        Qw[0] = Qwi
        Qw[-1] = -(k_rw[0]/(muw*Bw))/((k_rw[0]/(muw*Bw))+(k_ro[0]/(muo*Bo)))*Qwi
        Qo[-1] = -(Qwi+Qw[-1])
        return Qw, Qo

P_list = []
Sw_list = []

twophase = BuckleyLeverett_numerical(Sw, Swcrit, Swcon, Soirw, Sorw, ko_rw, ko_ro, Nw, No, N=N)
A_matrix = twophase.compute_A_matrix(area=area, phi=phi, dt=dt, dx=dx, N=N)
iter = 1200

for time in range(1, iter):
    print(f't: {time}', '-'*20)
    twophase = BuckleyLeverett_numerical(Sw, Swcrit, Swcon, Soirw, Sorw, ko_rw, ko_ro, Nw, No, N=N)
    k_rw, k_ro = twophase.calculate_relative_permeability()
    Tw = twophase.compute_T_matrix(k, k_rf=k_rw, dx=dx, area=area, muf=muw, Bf=Bw, P=Pi, conversion_factor=158)
    To = twophase.compute_T_matrix(k, k_rf=k_ro, dx=dx, area=area, muf=muo, Bf=Bo, P=Pi, conversion_factor=158)
    T = Tw + To
    ct = twophase.compute_total_compressibility_matrix(cf=cf, cw=cw, co=co)
    Qw, Qo = twophase.compute_Q(Qwi=Qwi, k_rw=k_rw, k_ro=k_ro, muw=muw, muo=muo, Bw=Bw, Bo=Bo)
    Qt = np.matmul(betaw,Qw) + np.matmul(betao,Qo)
    T = Tw+To+diags(np.ones((N,))*1e-10)
    Pnext = np.linalg.solve(T, Qt)
    
    Sw = Sw + np.linalg.solve(A_matrix, np.matmul(-Tw, Pnext)+Qw)
    
    Pi = copy.deepcopy(Pnext)
    Sw_list.append(Sw)    
    P_list.append(Pi)

timestep = iter-2
plt.plot(center_distances, Sw_list[timestep], label=f't={timestep}th day')
# plt.plot(center_distances, Sw_list[200], label=f't={200}th day')
# plt.plot(center_distances, Sw_list[400], label=f't={400}th day')
# plt.plot(center_distances, Sw_list[600], label=f't={600}th day')

plt.title(f'Numerical solution with grid numbers = {N}')
plt.xlabel('x distance, ft')
plt.ylabel('Water Saturation, Sw')
plt.ylim(0,1)
plt.xlim(0, max(center_distances))
plt.legend()
plt.show()

# Analytic solution
class BuckleyLeverett_analytical:
    
    def __init__(self, Swirr, Swinit, Sor, mu_o, mu_w, N):
        self.N = N
        self.Sor = Sor
        self.Sw = np.linspace(Swirr, 1-self.Sor, self.N)
        self.Swirr = Swirr
        self.Swinit = Swinit
        self.mu_o = mu_o
        self.mu_w = mu_w
    
    def calculate_fwd(self, fw, Sw, Sw_symbol):
        dfw_dS = sp.diff(fw, Sw_symbol)
        gfunc = sp.lambdify(Sw_symbol, dfw_dS, "numpy")
        return dfw_dS, gfunc(Sw)

    def calculate_fractional_flow(self, Sw):
        S = (Sw-self.Swirr)/(1-self.Swirr-self.Swinit)
        k_rw = 0.2*S**3
        k_ro = (1-S)**3
        self.Mo = (k_rw*self.mu_o)/(k_ro*self.mu_w)
        fw = self.Mo/(1+self.Mo)
        self.fw = fw
        return self.fw
    
    def calculate_front_saturation(self, Q_w, A, L, phi, t, dt):
        Sw = sp.Symbol('Sw')
        self.Sw_symbol = Sw
        S = (Sw - self.Swirr) / (1 - self.Swirr - self.Swinit)
        k_rw = 0.2 * S**3
        k_ro = (1 - S)**3

        Mo = (k_rw * self.mu_o) / (k_ro * self.mu_w)
        fw1_expr = Mo / (1 + Mo)

        fw_d_func = self.calculate_fwd(Sw_symbol=Sw, fw=fw1_expr, Sw=self.Sw)
        fw_d_expr = fw_d_func[0]
        fw2_expr = (Sw - self.Swirr) * fw_d_expr
        fw_d = fw_d_func[1]
        
        # Find intersection points by solving fw1 = fw2
        intersection_points = sp.solve(sp.Eq(fw1_expr, fw2_expr), Sw)

        self.fw1_expr = fw1_expr
        self.fw2_expr = fw2_expr

        self.td = Q_w*t*dt/(A*phi*L)
        self.Swf = intersection_points[1]
        self.xdf = self.td*self.calculate_fwd(Sw_symbol=Sw, fw=fw1_expr, Sw=self.Swf)[1]
        self.intersection_points = intersection_points
        self.xd = fw_d*self.td

        return self.Swf, self.xdf, self.xd, fw_d, self.td
    
    def calculate_tbt(self):
        self.tbt = 1/self.calculate_fwd(fw=self.fw1_expr, Sw=self.Swf, Sw_symbol=self.Sw_symbol)[1]
        return self.tbt

    def Sw_xd_profile(self, Swf, xdf, Sw, xd):
        Sw = np.append(Sw, Swf)
        Sw = np.sort(Sw) 
        Sw = np.flip(Sw)
        xd = np.flip(xd)
        idx = np.where(Sw == Swf)[0][0]
        xd = np.insert(xd, idx, xdf)
        xd = np.delete(xd, np.s_[idx+1:])
        Sw = np.delete(Sw, np.s_[idx+1:])
        Sw = np.append(Sw, self.Swinit)
        xd = np.append(xd, xdf)
        Sw = np.append(Sw, self.Swinit)
        xd = np.append(xd, 1)
        return Sw, xd
    
Buckley = BuckleyLeverett_analytical(Swirr=Swcon, Swinit=Swcrit, Sor=Sorw, mu_o=muo, mu_w=muw, N=N)
fw = Buckley.calculate_fractional_flow(Buckley.Sw)

x_analytic = []
Sw_analytic = []

for t in range(1, iter):
    if t % 100 == 0:
        print(t)
    Swf, xdf, xd, fw_d, td = Buckley.calculate_front_saturation(Q_w=Qwi, A=area, L=L, phi=phi, t=t, dt=dt)
    Sw, xd = Buckley.Sw_xd_profile(Swf, xdf, Buckley.Sw, xd)
    Sw_analytic.append(Sw)
    x_analytic.append(xd)

x_analytic = list(map(lambda x: x * L, x_analytic))

# print('Breakthrough time: --> ', Buckley.calculate_tbt())

# Benchmarking analytical solution with numerical solution
for day in range(0, iter, 100):
    plt.plot(x_analytic[day], Sw_analytic[day], label=f't ={day} Analytical', color='orange', linewidth=1.5)
    plt.scatter(center_distances, Sw_list[day], label=f't={day} Numerical', s=16)

plt.ylabel('Water Saturation, Sw')
plt.xlabel('x distance, ft')
# plt.legend()
plt.xlim(0, max(center_distances))
plt.ylim(0,1)
plt.title(f'Sw vs distance with grid numbers = {N}')
plt.show()

#Animated plot
fig, ax = plt.subplots()

x1 = copy.deepcopy(x_analytic)
y1 = copy.deepcopy(Sw_analytic)

x2 = copy.deepcopy(center_distances)
y2 = copy.deepcopy(Sw_list)

ax.set_xlim([0, max(center_distances)])
ax.set_ylim([0, 1])
ax.set_xlabel('Reservoir Length, ft')
ax.set_ylabel('Sw')
ax.set_title(f'Water Saturation vs distance')

myplot1, = ax.plot([], [], color='orange', linewidth=1.5)
myplot2, = ax.plot([], [], '.', markersize=5)
delay = 5
breaking = 1 

def update_plot(i, x1, y1, x2, y2, myplot1, myplot2):
    myplot1.set_data(x1[i], y1[i])
    myplot2.set_data(x2, y2[i])
    myplot2.set_color("blue")

    return myplot1, myplot2

ani = animation.FuncAnimation(fig, update_plot, frames=len(x1),
                              fargs=(x1, y1, x2, y2, myplot1, myplot2), interval=delay,
                              blit=True, repeat=True)
ani.save(filename="C:\\Users\\MahammadOjagzada\\Desktop\\Reservoir Simulation\\BuckleyLeverett_analytical.gif", writer="pillow")
plt.show()
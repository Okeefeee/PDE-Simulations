import numpy as np
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt
import matplotlib.animation as animation

######## Parameters ########

dimension = 1

sigma = 3

x_count = 100;
x_step = 1/10;

t_count = 2000;
t_step = 1/100;

######## End Parameters ########


x = np.linspace(0.0, x_step * x_count, x_count, endpoint = True)
t = np.linspace(0.0, t_step * t_count, t_count, endpoint = True)

k = fftfreq(x_count, x_step)

u = np.zeros((t_count,x_count), dtype = 'complex_')


# Initial data

for i in range(x_count):
    #u[0][i] = np.exp(-4*np.abs(x[i]-5)**2) + np.exp(-2*np.abs(x[i]-7))
    u[0][i] = 1* np.exp(-1*(x[i]-5)**2) 
    #u[0][i] = np.sin(2*np.pi*x[i])

######## Propogating #########

# iut + uxx + u2sigma u = 0 -> ut = -iuxx - i u 2sigma u
# u n = u n-1 + deltaT 

def propogate(u):
    uf = fft(u) # Take the fast fourier transform of u
    ui = u
    for time in range(1,t_count):
        nonlinear_terms = np.abs(ui[time-1])**(2*sigma) * ui[time-1]
        for x in range(x_count):
            uf[time][x] = uf[time-1][x] - t_step * complex(0,1) *(k[x]**2 * uf[time-1][x] - fft(nonlinear_terms)[x])
        ui = ifft(uf)
    return ui

ui = propogate(u)

fig = plt.figure()
ax = plt.axes(xlim = (0, x_count*x_step), ylim = (-0.00000005, 0.00000005))
ax.autoscale_view()
line, = ax.plot([], [], lw=2)
ttl = ax.text(.5, 1.05, '', transform = ax.transAxes, va='center')
ttl.set_text('t = 0 to ' + str(t_step*t_count))
skip_rate = 5

def init():
    line.set_data([], [])
    return line,

def animate(i):
    line.set_data(x, ui[i])
    #max_value = max(ui[i])
    #min_value = min(ui[i])
    #plt.axes(xlim = (0, x_count*x_step), ylim = (min_value, max_value))
    return line,


anim = animation.FuncAnimation(fig, animate, init_func=init,
                           frames=range(0, t_count, skip_rate), interval=10, blit=False)
plt.show()

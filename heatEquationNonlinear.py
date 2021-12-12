import numpy as np
from scipy.fft import fft, ifft, fftfreq
import matplotlib.pyplot as plt
import matplotlib.animation as animation

######## Parameters ########

dimension = 1

sigma = 1

x_count = 100;
x_step = 1/10;

t_count = 2000;
t_step = 1/100;


x = np.linspace(0.0, x_step * x_count, x_count, endpoint = True)
t = np.linspace(0.0, t_step * t_count, t_count, endpoint = True)

k = fftfreq(x_count, x_step)

u = np.zeros((t_count,x_count))


# Initial data

for i in range(x_count):
    u[0][i] = 1+ np.exp(-4*np.abs(x[i]-5)**2) + np.exp(-2*np.abs(x[i]-7))
    #u[0][i] = np.sin(2*np.pi*x[i])

######## Propogating #########

c = 1/100

def propogate(u):
    uf = fft(u) # Take the fast fourier transform of u
    ui = u
    for time in range(1,t_count):
        nonlinear_terms = ui[time-1]**(2*sigma + 1)
        for x in range(x_count):
            uf[time][x] = uf[time-1][x] - k[x]**2 * t_step * uf[time-1][x] + t_step * c * fft(nonlinear_terms)[x]
        ui = ifft(uf)
    return np.real(ui)

ui = propogate(u)

fig = plt.figure()
ax = plt.axes(xlim = (0, x_count*x_step), ylim = (-3, 3))
line, = ax.plot([], [], lw=2)
ttl = ax.text(.5, 1.05, '', transform = ax.transAxes, va='center')
ttl.set_text('t = 0 to ' + str(t_step*t_count))
skip_rate = 5

def init():
    line.set_data([], [])
    return line,

def animate(i):
    line.set_data(x, ui[i])
    return line,

anim = animation.FuncAnimation(fig, animate, init_func=init,
                           frames=range(0, t_count, skip_rate), interval=5, blit=True)

plt.show()

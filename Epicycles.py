import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from math import tau
from scipy.integrate import quad_vec
import matplotlib.animation as anim
from tqdm import tqdm

# Taking user input for the complexity of the fourier series
order = int(input('Enter the order : '))
name = input('Enter the file name : ')
print('Calculating coefficients...')

# reading the image and finding contours
img = cv.imread(r"")
img = cv.resize(img, (500,500))
grey_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
_, thresh = cv.threshold(grey_img, 120, 255, 0)
contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

# The contour should be a continuous list and the information which is detecting the boundary of the image itself is removed
contours = contours[1:]
contours = np.vstack(contours)

# Drawing contours
cv.drawContours(img, contours, -1, (0,0,255), 2)
# cv.imshow('img', img)
# cv.waitKey(0)

# Grabbing the x and y co-ordinates and adjusing them for plotting
x_points, y_points = contours[:,:,0].reshape(-1), -contours[:,:,1].reshape(-1)
x_points = x_points - np.mean(x_points)
y_points = y_points - np.mean(y_points)


xlim_data = plt.xlim()
ylim_data = plt.ylim()

# fig, ax = plt.subplots()
# ax.plot(x_points, y_points)
# plt.show()

# values for t
t_values = np.linspace(0, tau, len(x_points))

# We will be using an interpolated function with real and imaginary or complex parts
def f(t, t_values, x_points, y_points):
    return np.interp(t, t_values, x_points + 1j*y_points)

# Grabbing the Fourier-serie coefficients
coeffs = []
for n in range(-order, order +1):
    coef = (1/tau)*quad_vec(lambda t: f(t, t_values, x_points, y_points)*np.exp(-n*t*1j), 0, tau, limit = 100)[0]
    coeffs.append(coef)

# Creating empty plot variable which will be updated
fig, ax = plt.subplots()

circles = [ax.plot([], [], color = 'yellow')[0] for i in range(-order, order +1)]
circular_lines = [ax.plot([], [], color = 'blue')[0] for i in range(-order, order +1)]
drawing, = ax.plot([], [], color = 'black')

ax.set_xlim(xlim_data[0]-200, xlim_data[1]+200)
ax.set_ylim(ylim_data[0]-200, ylim_data[1]+200)

ax.set_axis_off()
ax.set_aspect('equal')

draw_x, draw_y = [], []

# Defining the maths for the motion of epicycles
def make_anim(i, time, coeffs):
    
    center_x, center_y = 0,0

    t = time[i]

    exp_terms = np.array([np.exp(-n*t*1j) for n in range(-order, order + 1)])

    fourier_values = coeffs*exp_terms

    x_coeffs = np.real(fourier_values)
    y_coeffs = np.imag(fourier_values)

    for i, (x_coeff, y_coeff) in enumerate(zip(x_coeffs, y_coeffs)):
        radius = np.linalg.norm([x_coeff, y_coeff])

        theta = np.linspace(0, tau, num=50)
        x, y = center_x + radius * np.cos(theta), center_y + radius * np.sin(theta)
        circles[i].set_data(x, y)

        x, y = [center_x, center_x + x_coeff], [center_y, center_y + y_coeff]
        circular_lines[i].set_data(x, y)

        center_x, center_y = center_x + x_coeff, center_y + y_coeff

    draw_x.append(center_x)
    draw_y.append(center_y)

    drawing.set_data(draw_x, draw_y)

# Creating and saving the animation
Writer = anim.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

frames = 500
time = np.linspace(0, 2.2*tau, num=frames)

anim = anim.FuncAnimation(fig, make_anim, frames= tqdm(range(frames), initial=0, position=0), fargs=(time, coeffs),interval=20)
anim.save(f'D:\software\{name}.mp4', writer=writer)

# (b) State Transition Matricies
import numpy as np
from copy import copy
import pylab
from matplotlib.patches import Ellipse

A = np.array([[1, 1], [0, 1]])
B = np.array([[0], [0]])
# Noise for acceleration directly varies the velocity, but only affects the position with a factor of 1/2
# v = a*t
# d = v*t + 1/2 a*t**2
# variance (V) = [0.5, 1]. R = V * V.T
R = np.array([[0.25, 0.5], [0.5, 1.0]])


#
# p(xt|ut,xt_1) = det(2*pi*Rt)**-0.5 * exp(-0.5*(xt-A*xt_1 - B*ut).T*inv(R)*(xt-A*xt_1*B*ut)
#
# (c) State Prediction
def state_prediction(mu_t_1, sigma_t_1, u_t):
    mu_bar_t = A.dot(mu_t_1) + B.dot(u_t)
    sigma_bar_t = A.dot(sigma_t_1).dot(A.T) + R

    return (mu_bar_t, sigma_bar_t)


mu = np.array([[0], [0]])
sigma = np.array([[0, 0], [0, 0]])
u = np.array([[0]])

for i in range(6):
    print('t = %d' % (i))
    print('\t' + 'mu = \n\t\t%s' % str(mu).replace('\n', '\n\t\t'))
    print('\t' + 'sigma = \n\t\t%s' % str(sigma).replace('\n', '\n\t\t'))
    print('')
    mu, sigma = state_prediction(mu, sigma, u)

# (d) Plot the posterior


def generate_ellipse(mu, sigma):
    eigen_values, eigen_vectors = np.linalg.eig(sigma)
    max_e_index = eigen_values.argmax()
    min_e_index = eigen_values.argmin()
    max_e_vect = eigen_vectors[:, max_e_index]
    angle = np.arctan2(max_e_vect[1], max_e_vect[0])
    return Ellipse(xy=mu.flatten(),
                   width=np.sqrt(eigen_values[max_e_index]),
                   height=np.sqrt(eigen_values[min_e_index]),
                   angle=angle * 180.0 / np.pi,
                   fill=False)


mu = np.array([[0], [0]])
sigma = np.array([[0, 0], [0, 0]])

ellipses = []
for i in range(6):
    ellipses.append(generate_ellipse(mu, sigma))
    mu, sigma = state_prediction(mu, sigma, [0])

fig = pylab.figure()
ax = fig.add_subplot(111, aspect='equal')
for e in ellipses:
    ax.add_artist(e)
    e.set_clip_box(ax.bbox)
    # e.set_alpha(rand())
    # e.set_facecolor(rand(3))

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_xlabel('$x$')
ax.set_ylabel('$\dot{x}$')
fig.suptitle('1 $\sigma$ uncertianty ellipse')

pylab.show()


c = np.array([[1, 0]])
q = np.array([[10]])


def measurement_update(mu_bar_t, sigma_bar_t, z_t):
    k_t = sigma_bar_t.dot(c.T).dot(np.linalg.inv(c.dot(sigma_bar_t).dot(c.T) + q))
    mu_t = mu_bar_t + k_t.dot(z_t - c.dot(mu_bar_t))
    sigma_t = (np.eye(k_t.shape[0]) - k_t.dot(c)).dot(sigma_bar_t)

    return mu_t, sigma_t

mu = np.array([[0], [0]])
sigma = np.array([[0, 0], [0, 0]])
u = np.array([[0]])

for i in range(5):
    mu, sigma = state_prediction(mu, sigma, u)

before_ellipse = generate_ellipse(mu, sigma)

mu, sigma = measurement_update(mu, sigma, np.array([[5]]))

after_ellipse = generate_ellipse(mu, sigma)

fig = pylab.figure()
ax = fig.add_subplot(111, aspect='equal')
for e in (before_ellipse, after_ellipse):
    ax.add_artist(e)
e.set_clip_box(ax.bbox)
# e.set_alpha(rand())
# e.set_facecolor(rand(3))

ax.set_xlim(-5.5, 5.5)
ax.set_ylim(-5.5, 5.5)
ax.set_xlabel('$x$')
ax.set_ylabel('$\dot{x}$')
fig.suptitle('1 $\sigma$ uncertianty ellipse')
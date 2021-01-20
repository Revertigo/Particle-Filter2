import matplotlib.pyplot as plt
import numpy as np

delay = 0.000001


def displaytrajectory(particles, bestP, gt, plot_all=True, disp=True):
    prts = np.array([[p.x, p.y, p.weight] for p in particles])
    prts[:, 2] /= prts[:, 2].sum()

    if plot_all:
        plt.scatter(prts[:, 0], prts[:, 1], c=prts[:, 2], cmap='gray', s=10, edgecolors='grey')

    plt.plot(gt[0], gt[1], 'g*')
    plt.plot(bestP.x, bestP.y, 'c*')
    # locate the legend at top left corner
    plt.legend((['GT', 'Estimation']), loc=2)

    if disp:
        plt.pause(delay)

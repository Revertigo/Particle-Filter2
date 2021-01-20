import time
import matplotlib.pyplot as plt

from loader import loadData
from visualization import displaytrajectory

from particleFilter import ParticleFilter, getError

numberOfParticles = 200
sigmaY = 0.6
interval = 20  # Every how many times to show the paritcles cloud


def main():
    observation, control_data, gt_data, landmarks = loadData()
    particleFilter = ParticleFilter(0, 0, sigmaY, numOfParticles=numberOfParticles)

    for i in range(len(observation)):
        # prediction
        if i != 0:
            velocity = control_data.iloc[i - 1][0]
            yaw_rate = control_data.iloc[i - 1][1]
            particleFilter.moveParticles(velocity, yaw_rate)
        a = observation[i].copy()
        particleFilter.UpdateWeight(a)  # Sense/measurement update
        particleFilter.Resample()
        bestP = particleFilter.getBestParticle()
        error = getError(gt_data.iloc[i], bestP)
        # Print error of each iteration
        print(i, error)

        # Visualization, every 20 observation prints the particles cloud
        displaytrajectory(particleFilter.particles, bestP, gt_data.iloc[i], i % interval == 0, True)

    plt.savefig('../results .png')
    plt.show()


# Sometimes, the filter won't converge but diverge. It will happen with small probability
if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end - start)

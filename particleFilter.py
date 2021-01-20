import numpy as np
import pandas as pd
from dataclasses import dataclass
import random

np.random.seed(14)

@dataclass
class Particle:
    x: float
    y: float
    theta: float
    weight: float


@dataclass
class LandMark:
    x: float
    y: float
    index: int


map_data = pd.read_csv('data/map_data.txt', names=['X', 'Y', '# landmark'])
result = [(x, y, landmark) for x, y, landmark in zip(map_data['X'], map_data['Y'], map_data['# landmark'])]
landMarkList = []
for res in result:
    l = LandMark(res[0], res[1], res[2])
    landMarkList.append(l)


def calculateDistance(landmark1, landmark2):
    a = np.sqrt((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2)
    return a


def findClosestLandmark(map_landmarks, singleObs):
    closest_landmark = (map_landmarks[0])  # initialization
    min_distance = calculateDistance(map_landmarks[0], singleObs)

    for i in range(1, len(map_landmarks)):
        dist = calculateDistance(map_landmarks[i], singleObs)
        if dist < min_distance:
            min_distance = dist
            closest_landmark = map_landmarks[i]

    return closest_landmark


def getError(gt_data, bestParticle):
    error1 = np.abs(gt_data[0] - bestParticle.x)
    error2 = np.abs(gt_data[1] - bestParticle.y)
    error3 = np.abs(gt_data[2] - bestParticle.theta)
    if (error3 > 2 * np.pi):
        error3 = 2 * np.pi - error3
    return (error1, error2, error3)


def findObservationProbability(closest_landmark, map_coordinates, sigmaX, sigmaY):
    mew_x = closest_landmark.x
    mew_y = closest_landmark.y

    x = map_coordinates.x
    y = map_coordinates.y
    frac = (1 / (2 * np.pi * sigmaX * sigmaY))
    weight1 = (x - mew_x) ** 2 / ((sigmaX) ** 2) + (y - mew_y) ** 2 / (sigmaY ** 2)
    ans = np.exp(-0.5 * weight1) * frac
    # In case weight is zero, we still want to return very small weight
    return max(ans, np.finfo('float').eps)


def mapObservationToMapCoordinates(observation, particle):
    x = observation.x
    y = observation.y

    xt = particle.x
    yt = particle.y
    theta = particle.theta

    MapX = x * np.cos(theta) - y * np.sin(theta) + xt
    MapY = x * np.sin(theta) + y * np.cos(theta) + yt

    return MapX, MapY


# Convert all observations for single particle into global map coordinates
def mapObservationsToMapCordinatesList(observations, particle):
    convertedObservations = []
    i = 0
    for obs in observations.iterrows():
        singleObs = LandMark(obs[1][0], obs[1][1], 1)
        mapX, mapY = mapObservationToMapCoordinates(singleObs, particle)
        tmpLandmark = LandMark(x=mapX, y=mapY, index=i)
        i += 1
        convertedObservations.append(tmpLandmark)
    return convertedObservations


class ParticleFilter:
    particles = []

    def __init__(self, intialX, initialY, std, numOfParticles):
        self.number_of_particles = numOfParticles
        self.sigma = std
        for i in range(self.number_of_particles):
            x = random.gauss(intialX, std)
            y = random.gauss(initialY, std)
            theta = random.uniform(0, 2 * np.pi)
            tmpParticle = Particle(x, y, theta, 1)

            self.particles.append(tmpParticle)

    def moveParticles(self, velocity, yaw_rate, delta_t=0.1):

        for i in range(self.number_of_particles):
            if (yaw_rate != 0):
                theta = self.particles[i].theta
                newTheta = theta + delta_t * yaw_rate
                newX = self.particles[i].x + (velocity / yaw_rate) * (np.sin(newTheta) - np.sin(theta))
                newY = self.particles[i].y + (velocity / yaw_rate) * (np.cos(theta) - np.cos(newTheta))

                self.particles[i].x = newX + random.gauss(0, 0.3)
                self.particles[i].y = newY + random.gauss(0, 0.3)
                self.particles[i].theta = newTheta + random.gauss(0, 0.01)

    def UpdateWeight(self, observations):
        for i in range(self.number_of_particles):
            global_obs = mapObservationsToMapCordinatesList(observations, self.particles[i])
            elaborate_weight = 1

            for j in range(len(global_obs)):
                # Association
                closest_landmark = findClosestLandmark(landMarkList, global_obs[j])
                new_weight = findObservationProbability(closest_landmark, global_obs[j], self.sigma, self.sigma)
                elaborate_weight *= new_weight

            self.particles[i].weight = elaborate_weight

    def getBestParticle(self):
        best_particle = max(self.particles, key=lambda particle: particle.weight)
        return best_particle

    def getBestParticleOut(self):
        x = 0
        y = 0
        theta = 0
        for i in range(self.number_of_particles):
            x += self.particles[i].x
            y += self.particles[i].y
            theta += self.particles[i].theta
        x = x / self.number_of_particles
        y = y / self.number_of_particles
        theta = theta / self.number_of_particles
        best_particle = Particle(x, y, theta, weight=1)
        return best_particle

    def PrintWeights(self):
        for i in range(self.number_of_particles):
            print("Weight:", self.particles[i].weight, self.particles[i].x, self.particles[i].y)

    def Resample(self):
        w = np.array([x.weight for x in self.particles])
        w = w / w.sum()
        new_particles = []
        index = random.randint(0, self.number_of_particles-1)
        for i in range(self.number_of_particles):

            B = np.random.random() * (2 * w.max())

            while w[index] < B:
                B -= w[index]
                index = ((index + 1) % self.number_of_particles)

            # Create new particle
            particle = Particle(
                self.particles[index].x,
                self.particles[index].y,
                self.particles[index].theta,
                self.particles[index].weight
            )
            new_particles.append(particle)

        # save the resampled group
        self.particles = list(new_particles)

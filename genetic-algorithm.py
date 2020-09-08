import random
import math
import matplotlib.pyplot as plt
import numpy as np


class Population:
    def __init__(self, n_indiv, n_traits, r_traits, m_rate, individuals=[]):
        self.n_indiv = n_indiv
        self.n_traits = n_traits
        self.r_traits = r_traits
        self.m_rate = m_rate
        self.max_fitness = -1
        self.mean_fitness = -1
        if individuals == []:
            self.individuals = []
            for i in range(n_indiv):
                self.individuals.append(Individual(n_traits=n_traits, r_traits=r_traits))
        else:
            self.individuals = individuals

    def calc_fitness(self, optimum):
        squared_sums = 0
        for t in range(self.n_traits):
            squared_sums += (self.r_traits[0]-self.r_traits[1])**2
        worst_fitness = math.sqrt(squared_sums)
        fitnesses = []
        for i in self.individuals:
            fitnesses.append(i.calc_fitness(optimum=optimum, worst_fitness=worst_fitness))
        self.max_fitness = max(fitnesses)
        self.mean_fitness = sum(fitnesses) / len(fitnesses)

    def breed(self):
        parent_pool = []
        for i in self.individuals:
            for p in range(round(i.fitness*10)*round(i.fitness*10)):
                parent_pool.append(i)
        new_pop = []
        for n in range(self.n_indiv):
            parent_1 = parent_pool[random.randint(0, len(parent_pool)-1)]
            parent_2 = parent_pool[random.randint(0, len(parent_pool)-1)]
            traits = []
            for t in range(self.n_traits):
                if random.uniform(0, 1) < self.m_rate:
                    traits.append(random.uniform(self.r_traits[0], self.r_traits[1]))
                else:
                    if random.uniform(0, 1) < 0.5:
                        traits.append(parent_1.traits[t])
                    else:
                        traits.append(parent_2.traits[t])
            new_pop.append(Individual(n_traits=self.n_traits, r_traits=self.r_traits, traits=traits))
        return Population(n_indiv=self.n_indiv, n_traits=self.n_traits, r_traits=self.r_traits, m_rate=self.m_rate, individuals=new_pop)

    def view(self, optimum, generation):
        if self.n_traits==2:
            points = [[] for t in range(self.n_traits)]
            for i in self.individuals:
                for t in range(self.n_traits):
                    points[t].append(i.traits[t])
            optimum_converted_x = ((optimum[0]-self.r_traits[0]*2)/(self.r_traits[1]*2-self.r_traits[0]*2))*100
            optimum_converted_y = ((-1*optimum[1]-self.r_traits[0]*2)/(self.r_traits[1]*2-self.r_traits[0]*2))*100
            arr = np.zeros((100, 100, 3), dtype=np.uint8)
            imgsize = arr.shape[:2]
            innerColor = (0, 255, 0)
            outerColor = (0, 0, 0)
            for y in range(imgsize[1]):
                for x in range(imgsize[0]):
                    distanceToCenter = np.sqrt((x - optimum_converted_x) ** 2 + (y - optimum_converted_y) ** 2)
                    distanceToCenter = distanceToCenter / (np.sqrt(2) * imgsize[0] / 2)
                    r = outerColor[0] * distanceToCenter + innerColor[0] * (1 - distanceToCenter)
                    g = ((outerColor[1] * distanceToCenter + innerColor[1] * (1 - distanceToCenter))-(255/2))*2
                    if g < 0:
                        g = 0
                    b = outerColor[2] * distanceToCenter + innerColor[2] * (1 - distanceToCenter)
                    arr[y, x] = (int(r), int(g), int(b))
            plt.imshow(arr, extent=[self.r_traits[0]*2, self.r_traits[1]*2, self.r_traits[0]*2, self.r_traits[1]*2])
            plt.scatter(points[0], points[1])
            plt.scatter([optimum[0]], [optimum[1]], color="red")
            plt.ylim(self.r_traits[0], self.r_traits[1])
            plt.xlim(self.r_traits[0], self.r_traits[1])
            plt.savefig(str(generation) + ".png")
            plt.show()
        else:
            print("View not supported for number of traits")

class Individual:
    def __init__(self, n_traits, r_traits, traits=[]):
        self.n_traits = n_traits
        self.r_traits = r_traits
        self.fitness = -1
        if traits == []:
            self.traits = []
            for t in range(n_traits):
                self.traits.append(random.uniform(r_traits[0], r_traits[1]))
        else:
            self.traits = traits

    def calc_fitness(self, optimum, worst_fitness):
        squared_sums = 0
        for t in range(self.n_traits):
            squared_sums += (self.traits[t]-optimum[t])**2
        self.fitness = (worst_fitness - math.sqrt(squared_sums)) / worst_fitness
        return self.fitness


optimum = [0, 0]
generations = 25

pop = Population(n_indiv=1000, n_traits=len(optimum), r_traits=(-10, 10), m_rate=0.1)
for g in range(generations):
    pop.view(optimum=optimum, generation=g)
    pop.calc_fitness(optimum=optimum)
    print(pop.max_fitness, pop.mean_fitness)
    pop = pop.breed()
pop.calc_fitness(optimum=optimum)
print(pop.max_fitness, pop.mean_fitness)
pop.view(optimum=optimum, generation=g)
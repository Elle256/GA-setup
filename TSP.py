import numpy as np 
import random
import copy
import matplotlib.pyplot as plt 
import time


class Problem:
    def __init__(self, matrix):
        self.matrix = matrix
    def get_size(self):
        return self.matrix.shape[0]
    

def decode(chromosome):
    indices = np.argsort(chromosome)
    solution = np.empty_like(chromosome)
    rank = np.arange(1, len(chromosome) + 1)
    solution[indices] = rank
    return solution.astype(int)

def get_fitness(solution, problem : Problem):
    city = decode(solution)
    cost = problem.matrix
    N = city.size
    total_cost = cost[0][city[0]]
    for i in range(1, N):
        total_cost += cost[city[i - 1]][city[i]]
    total_cost += cost[city[-1]][0]
    return -total_cost


class Individual:
    def __init__(self):
        self.chromosome = None
        self.fitness = None

    def genIndi(self, problem : Problem):
        self.chromosome = np.random.uniform(0.0, 1.0, problem.get_size() - 1)
    
    def cal_fitness(self, problem):
        self.fitness = get_fitness(self.chromosome, problem)

    def clone(self):
        return copy.deepcopy(self)
    
    def __repr__(self):
        return f"chromosome={self.chromosome}, fitness={self.fitness}" 
    

# Simulated binary crossover - SBX
def crossover(parent1, parent2, problem : Problem, eta = 2.0):
    off1 = Individual()
    off2 = Individual()
    r = np.random.rand()
    if (r <= 0.5):
        beta = (2*r)**(1.0/(eta + 1))
    else:
        beta = (1.0/(2*(1 - r)))**(1.0/(eta + 1))
    p1 = parent1.chromosome
    p2 = parent2.chromosome
    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    c1 = np.clip(c1, 0.0, 1.0)
    c2 = np.clip(c2, 0.0, 1.0)
    off1.chromosome = c1
    off2.chromosome = c2
    return off1.clone(), off2.clone()


# Polynomial mutaion - PM
def mutation(indi : Individual, eta = 20.0):
    chr = indi.chromosome
    for i in range(chr.size):
        mu = np.random.rand()
        if (mu <= 0.5):
            delta = (2 * mu)**(1.0/(1 + eta)) - 1
            chr[i] = chr[i] + delta * chr[i]
        else:
            delta = 1 - (2 - 2*mu)**(1.0/(1 + eta))
            chr[i] = chr[i] + delta * (1 - chr[i])
            
    chr = np.clip(chr, 0.0, 1.0)
    indi.chromosome = chr
    return indi.clone()


class Population:
    def __init__(self, pop_size, problem : Problem):
        self.pop_size = pop_size
        self.list_indi = []
        self.problem = problem
    
    def genPop(self):
        for i in range(self.pop_size):
            indi = Individual()
            indi.genIndi(self.problem)
            indi.cal_fitness(self.problem)
            self.list_indi.append(indi)

    def __repr__(self):
        pass
def selection(list, k = 2):
    tour1 = random.sample(list, k)
    tour2 = random.sample(list, k)
    x = max(tour1, key=lambda indi: indi.fitness)
    y = max(tour2, key=lambda indi: indi.fitness)
    return x.clone(), y.clone() 
def survival_selection(list, pop_size):
    list = sorted(list, key=lambda indi: indi.fitness, reverse=True)
    list = list[0: pop_size]
    return list
def GA(problem, pop_size, max_gen, p_c, p_m):
    pop = Population(pop_size, problem)
    pop.genPop()
    history = []
    for i in range(max_gen):
        child = []
        while (len(child) < pop_size):
            p1, p2 = selection(pop.list_indi)
            if np.random.rand() <= p_c:
                c1, c2 = crossover(p1, p2, problem)
                c1.cal_fitness(problem)
                c2.cal_fitness(problem)
                child.append(c1)
                child.append(c2)
            if np.random.rand() <= p_m:
                p1 = mutation(p1)
                p2 = mutation(p2)
                p1.cal_fitness(problem)
                p2.cal_fitness(problem)
                child.append(p1)
                child.append(p2)
        pop.list_indi = survival_selection(pop.list_indi + child, pop_size)
        history.append(pop.list_indi[0].fitness)
    solution = pop.list_indi[0]
    return history, solution
# setup
n = 10
matrix = []
for i in range(n):
    arr = np.random.uniform(0.0, 20.0, n)
    arr[i] = 0.0
    matrix.append(arr)
matrix = np.array(matrix)

problem = Problem(matrix)
pop_size = 10
max_gen = 100
Pc = 0.5
Pm = 0.5
start_time = time.time()
# start
fitness_history, solution = GA(problem, pop_size, max_gen, Pc, Pm)
end_time = time.time()
execution_time = end_time - start_time
#show
for i in range(len(fitness_history)):
    print(f"Generation {i}, bestfitness = {-fitness_history[i]:.2f}")

np.set_printoptions(precision=2, suppress=True)
print("problem:")
print(matrix)
print()
print("solution:")
print(decode(solution.chromosome))
print(f"total_cost =  {-solution.fitness:.2f}")
print(f"\n Thời gian chạy chương trình: {execution_time:.4f} giây")
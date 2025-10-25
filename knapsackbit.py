import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import time

# ----- Định nghĩa bài toán Knapsack -----
class Problem:
    def __init__(self, profit, weight, capacity):
        self.profit = profit
        self.weight = weight
        self.capacity = capacity
#Hàm mã hóa (Bit encoding)
def decode(chromosome):
    return chromosome.astype(int)

# Hàm fitness
def get_fitness(chromosome, problem: Problem):
    solution = decode(chromosome)
    total_profit = np.sum(solution * problem.profit)
    total_weight = np.sum(solution * problem.weight)
    if total_weight > problem.capacity:
        return 0.0
    else:
        return float(total_profit)

class Individual:
    def __init__(self):
        self.chromosome = None  # numpy array of 0/1
        self.fitness = None

    def gen_indi(self, problem: Problem):
        num_item = problem.profit.size
        # random bits 0/1
        self.chromosome = np.random.randint(0, 2, num_item, dtype=int)

    def cal_fitness(self, problem: Problem):
        self.fitness = get_fitness(self.chromosome, problem)

    def clone(self):
        return copy.deepcopy(self)

    def __repr__(self):
        return f"chromosome={self.chromosome}, fitness={self.fitness}"

# One-point crossover for bitstrings
def crossover(parent1, parent2):
    off1 = Individual()
    off2 = Individual()
    p1 = parent1.chromosome
    p2 = parent2.chromosome
    n = p1.size
    if n <= 1:
        # no crossover possible
        off1.chromosome = p1.copy()
        off2.chromosome = p2.copy()
        return off1.clone(), off2.clone()

    point = np.random.randint(1, n)  # crossover point in [1, n-1]
    c1 = np.concatenate([p1[:point], p2[point:]]).astype(int)
    c2 = np.concatenate([p2[:point], p1[point:]]).astype(int)
    off1.chromosome = c1
    off2.chromosome = c2
    return off1.clone(), off2.clone()

# Bit-flip mutation
def mutation(indi, pm_gene=0.01):
    chr = indi.chromosome.copy()
    for i in range(chr.size):
        if np.random.rand() < pm_gene:
            chr[i] = 1 - chr[i]
    indi.chromosome = chr
    return indi.clone()

class Population:
    def __init__(self, pop_size, problem: Problem):
        self.pop_size = pop_size
        self.list_indi = []
        self.problem = problem

    def genPop(self):
        self.list_indi = []
        for i in range(self.pop_size):
            indi = Individual()
            indi.gen_indi(self.problem)
            indi.cal_fitness(self.problem)
            self.list_indi.append(indi)

    def show(self):
        for i in range(len(self.list_indi)):
            print(f"Individual {i}: {self.list_indi[i]}")

# Tournament selection (k-way)
def selection(list_indi, k=2):
    tour1 = random.sample(list_indi, k)
    tour2 = random.sample(list_indi, k)
    x = max(tour1, key=lambda indi: indi.fitness)
    y = max(tour2, key=lambda indi: indi.fitness)
    return x.clone(), y.clone()

def survival_selection(list_all, pop_size):
    list_sorted = sorted(list_all, key=lambda indi: indi.fitness, reverse=True)
    return [indi.clone() for indi in list_sorted[:pop_size]]

def GA(problem, pop_size, max_gen, p_c, p_m, pm_gene=0.02, tournament_k=3):
    pop = Population(pop_size, problem)
    pop.genPop()
    history = []
    for gen in range(max_gen):
        child = []
        while len(child) < pop_size:
            p1, p2 = selection(pop.list_indi, k=tournament_k)
            if np.random.rand() <= p_c:
                c1, c2 = crossover(p1, p2)
            else:
                c1 = p1.clone()
                c2 = p2.clone()
            if np.random.rand() <= p_m:
                c1 = mutation(c1, pm_gene=pm_gene)
            if np.random.rand() <= p_m:
                c2 = mutation(c2, pm_gene=pm_gene)
            c1.cal_fitness(problem)
            c2.cal_fitness(problem)
            child.append(c1)
            if len(child) < pop_size:
                child.append(c2)
        pop.list_indi = survival_selection(pop.list_indi + child, pop_size)
        history.append(pop.list_indi[0].fitness)
    solution = pop.list_indi[0]
    return history, solution

# -------------------- setup--------------------

profit = np.random.uniform(5.0, 20.0, 5)   
weight = np.random.uniform(2.0, 20.0, 5)
capacity = np.random.uniform(0.0, np.sum(weight) * 0.5) 
problem = Problem(profit, weight, capacity)

pop_size = 10
max_gen = 100
Pc = 0.5
Pm = 0.5
pm_gene = 0.02  

start_time = time.time()
fitness_history, solution = GA(problem, pop_size, max_gen, Pc, Pm, pm_gene=pm_gene, tournament_k=3)
end_time = time.time()
execution_time = end_time - start_time

# show results
for i in range(len(fitness_history)):
    print(f"Generation {i}, bestfitness = {fitness_history[i]:.2f}")

np.set_printoptions(precision=2, suppress=True)
print("\nproblem:")
print(f"profit:  {profit}")
print(f"weight:  {weight}")
print(f"capacity = {capacity:.2f}\n")

print("solution (bitstring):")
sol_bits = decode(solution.chromosome)
print(sol_bits)
print(f"total_profit =  {solution.fitness:.2f}")
print(f"total_weight = {np.sum(sol_bits * weight):.2f}")
print(f"\n⏱️ Thời gian chạy chương trình: {execution_time:.4f} giây")

# plot fitness progression
generations = list(range(len(fitness_history)))
plt.figure(figsize=(12, 4))
plt.plot(generations, fitness_history, marker='o', linestyle='-', label='Best Fitness')
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("Fitness Progress Over Generations")
plt.legend()
plt.grid(True)
plt.show()

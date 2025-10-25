import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import time

# ----- Định nghĩa bài toán -----
class Problem:
    def __init__(self, matrix):
        self.matrix = matrix
    def get_size(self):
        return self.matrix.shape[0]

# ----- Hàm tính fitness -----
def get_fitness(chromosome, problem: Problem):
    cost = problem.matrix
    N = len(chromosome)
    total_cost = 0
    current = 0
    visited = set()

    for _ in range(N):
        nxt = chromosome[current]
        total_cost += cost[current][nxt]
        visited.add(current)
        current = nxt

    total_cost += cost[current][0]  # quay lại điểm đầu

    # phạt nếu chưa đi hết thành phố
    if len(visited) < N:
        total_cost += 1e6

    return -total_cost

def repair(chromosome):
    n = len(chromosome)
    used = set()
    for i in range(n):
        nxt = chromosome[i]
        if nxt == i or nxt in used or nxt >= n:
            candidates = [x for x in range(n) if x not in used and x != i]
            if not candidates:
                candidates = [x for x in range(n) if x != i]
            chromosome[i] = random.choice(candidates)
        used.add(chromosome[i])
    return chromosome

# ----- Lớp cá thể -----
class Individual:
    def __init__(self):
        self.chromosome = None
        self.fitness = None

    def genIndi(self, problem: Problem):
        n = problem.get_size()
        cities = list(range(n))
        random.shuffle(cities)
        chromosome = [-1] * n
        for i in range(n - 1):
            chromosome[cities[i]] = cities[i + 1]
        chromosome[cities[-1]] = cities[0]
        self.chromosome = chromosome

    def cal_fitness(self, problem):
        self.fitness = get_fitness(self.chromosome, problem)

    def clone(self):
        return copy.deepcopy(self)

# ----- Toán tử lai ghép -----
def crossover(parent1, parent2, problem: Problem):
    off1 = parent1.clone()
    off2 = parent2.clone()
    n = len(parent1.chromosome)

    for i in range(n):
        if random.random() < 0.5:
            off1.chromosome[i] = parent2.chromosome[i]
        if random.random() < 0.5:
            off2.chromosome[i] = parent1.chromosome[i]

    off1.chromosome = repair(off1.chromosome)
    off2.chromosome = repair(off2.chromosome)
    off1.cal_fitness(problem)
    off2.cal_fitness(problem)
    return off1, off2

# ----- Toán tử đột biến -----
def mutation(indi: Individual, problem: Problem):
    n = len(indi.chromosome)
    a, b = random.sample(range(n), 2)
    indi.chromosome[a], indi.chromosome[b] = indi.chromosome[b], indi.chromosome[a]
    indi.chromosome = repair(indi.chromosome)
    indi.cal_fitness(problem)
    return indi

# ----- Quần thể -----
class Population:
    def __init__(self, pop_size, problem: Problem):
        self.pop_size = pop_size
        self.list_indi = []
        self.problem = problem

    def genPop(self):
        for _ in range(self.pop_size):
            indi = Individual()
            indi.genIndi(self.problem)
            indi.cal_fitness(self.problem)
            self.list_indi.append(indi)

# ----- Chọn lọc -----
def selection(list, k=2):
    tour1 = random.sample(list, k)
    tour2 = random.sample(list, k)
    x = max(tour1, key=lambda indi: indi.fitness)
    y = max(tour2, key=lambda indi: indi.fitness)
    return x.clone(), y.clone()

# ----- Chọn lọc sống sót -----
def survival_selection(list, pop_size):
    list = sorted(list, key=lambda indi: indi.fitness, reverse=True)
    return list[:pop_size]

# ----- Hàm chính GA -----
def GA(problem, pop_size, max_gen, p_c, p_m):
    pop = Population(pop_size, problem)
    pop.genPop()
    history = []

    for i in range(max_gen):
        child = []
        while len(child) < pop_size:
            p1, p2 = selection(pop.list_indi)
            if np.random.rand() <= p_c:
                c1, c2 = crossover(p1, p2, problem)
                child.append(c1)
                child.append(c2)
            if np.random.rand() <= p_m:
                p1 = mutation(p1, problem)
                p2 = mutation(p2, problem)
                child.append(p1)
                child.append(p2)

        pop.list_indi = survival_selection(pop.list_indi + child, pop_size)
        history.append(-pop.list_indi[0].fitness)
    solution = pop.list_indi[0]
    return history, solution

# ----- Thiết lập -----
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
fitness_history, solution = GA(problem, pop_size, max_gen, Pc, Pm)
end_time = time.time()
execution_time = end_time - start_time

# ----- Kết quả -----
for i in range(len(fitness_history)):
    print(f"Generation {i}, best_cost = {fitness_history[i]:.2f}")

print("\nProblem matrix:")
print(matrix)
print("\nBest chromosome (adjacency list):")
print(solution.chromosome)
print(f"Total cost = {fitness_history[-1]:.2f}")
print(f"\nThời gian chạy chương trình: {execution_time:.4f} giây")


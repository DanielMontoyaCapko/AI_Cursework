import random as rd

tournament_size = 3
mutation_rate = 0.01
num_generations = 100
population_size = 10
chromosome_length = 32
#fgoal = [1] * chromosome_length
best_individual = 0
input_chromosome = [1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0]

def initial_population(size, chromosome, input_chromosome):
    # Initialize the population with the given input chromosome and random chromosomes
    population = [input_chromosome]
    for _ in range(size - 1):
        population.append(rd.choices(range(2), k=chromosome))
    return population

def fitness_function(chromosome):
    # Calculate the fitness by counting the number of differences between the chromosome and the goal
    return chromosome.count(1)


def print_fpop(f_pop):
    # Print the population or any list of individuals
    for indexp in f_pop:
        print(indexp)

def mating_crossover(parent_a, parent_b):
    # Perform crossover (single-point) between two parents to create offspring
    offspring = []
    cut_point = rd.randint(1, len(parent_a) - 1)
    offspring.append(parent_a[:cut_point] + parent_b[cut_point:])
    offspring.append(parent_b[:cut_point] + parent_a[cut_point:])
    return offspring

def mutate(chromosome, mutation_rate):
    # Apply mutation to a chromosome by flipping a bit with a small probability
    for idx in range(len(chromosome)):
        if rd.random() < mutation_rate:
            chromosome = chromosome[:idx] + [1 - chromosome[idx]] + chromosome[idx + 1:]
    return chromosome

def tournament_selection(population, fitness, tournament_size):
    selected = []
    for _ in range(len(population)):
        participants = rd.sample(list(zip(population, fitness)), tournament_size)
        winner = max(participants, key=lambda x: x[1])
        selected.append(winner[0])
    return selected

print('###########################')
# Initialize the population
population = initial_population(population_size, chromosome_length, input_chromosome)
print('population')
print('________________________')
print_fpop(population)
print('________________________')

#while generation < num_generations:
for generation in range(num_generations):

    # Calculate fitness for each individual in the population
    fitall = [fitness_function(indi) for indi in population]
    print(f"Generation number: {generation}")

    if max(fitall) == chromosome_length:
        print(f"Goal reached at generation {generation}")
        break

    # Perform tournament selection to choose parents for crossover
    parents = tournament_selection(population, fitall, tournament_size)

    # Perform crossover 
    off = mating_crossover(parents[1], parents[2])

    # Generate offspring
    offspring = []
    for i in range(0, len(parents), 2):
        if i + 1 < len(parents):
            offspring.extend(mating_crossover(parents[i], parents[i+1]))

####3

 # Apply mutation
    for i in range(len(offspring)):
        offspring[i] = mutate(offspring[i],mutation_rate)

    # Elite Selection - Keep some of the best individuals
    elite_size = 2
    sorted_population = sorted(zip(population, fitall), key=lambda x: x[1], reverse=True) #key is sorting the fitness, you reverse it to do it from smaller to bigger
    elites = [x[0] for x in sorted_population[:elite_size]]

    # Create new population
    population = elites + offspring

    #Final population
    print('Final population:')
    print_fpop(population)

    #Calculate fitness for the final population
    final_fitness = [fitness_function(indi) for indi in population]

    #Find and print the best chromosome from the final population
    best_final_fitness = max(final_fitness)
    best_final_chromosome = population[final_fitness.index(best_final_fitness)]
    print(f"Best Chromosome in Final Population: {best_final_chromosome}, Fitness: {best_final_fitness}")

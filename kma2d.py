import numpy as np
import random


def get_function(function_id, dimension):
    """
    :param function_id: function id
    :param dimension: dimension of the problem
    :return: function name, upper bound, lower bound, number of constraints
    """
    n_var = dimension

    # switch
    match function_id:
        case 1:
            cons_ub = 100
            cons_lb = -100
            mv = 0

    return n_var, cons_ub, cons_lb, mv


def pop_cons_initialization(population_size, n_var, cons_ub, cons_lb):
    """
    Generate a population of individuals with constraints
    :param population_size: number of individuals
    :param n_var: number of variables
    :param cons_ub: upper bound of constraints
    :param cons_lb: lower bound of constraints
    :return: population of individuals
    """

    # initialize f1 and f2 with constraints
    f1 = [0.01, 0.01, 0.99, 0.99]
    f2 = [0.01, 0.99, 0.01, 0.99]

    # initialize population
    population = np.zeros((population_size, n_var))

    IndX = 0
    for nn in range(1, population_size + 1, 4):
        # check for the last population
        if population_size - nn >= 4:
            number_locations = 4
        else:
            number_locations = population_size - nn + 1

        ss = 0
        while ss < number_locations:
            Temp = np.zeros(n_var)
            for i in range(n_var // 2):
                Temp[i] = cons_lb[i] + (cons_ub[i] - cons_lb[i]) * (
                    f1[ss] + (random.random() * 2 - 1) * 0.01
                )
            for i in range(n_var // 2, n_var):
                Temp[i] = cons_lb[i] + (cons_ub[i] - cons_lb[i]) * (
                    f2[ss] + (random.random() * 2 - 1) * 0.01
                )

            population[IndX, :] = Temp
            IndX += 1
            ss += 1

    return population


# runner
if __name__ == "__main__":
    function_id = 1
    dimension = 50
    pop_size = 5

    n_var, cons_ub, cons_lb, mv = get_function(function_id, dimension)

    cons_ub = np.ones(n_var) * cons_ub
    cons_lb = np.ones(n_var) * cons_lb

    population = pop_cons_initialization(pop_size, n_var, cons_ub, cons_lb)
    print(population)

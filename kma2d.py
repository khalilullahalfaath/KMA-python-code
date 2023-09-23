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


def evaluation(X, function_id):
    """
    :param X: population
    :return: fitness value
    """
    dim = X.shape[0]

    # switch
    match function_id:
        case 1:
            fx = np.sum(X**2)

    return fx


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


def move_big_males_female_first_stage(big_males, big_malesFX, female, femaleFX, n_var):
    HQ = big_males.copy()
    HQFX = big_malesFX.copy()

    TempSM = big_males.copy()
    TempSMFX = big_malesFX.copy()

    for ss in range(TempSM.shape[0]):
        MaxFolHQ = np.random.randint(1, 3)
        VM = np.zeros(n_var)  # Velocity of a strong male
        RHQ = np.random.permutation(HQ.shape[0])
        FolHQ = 0

        for fs in range(RHQ.shape[0]):
            ind = RHQ[fs]
            if ind != ss:
                # Semi-randomly select an individual to define attraction or distraction
                if HQFX[ind] < TempSMFX[ss] or np.random.rand() < 0.5:
                    VM += np.random.rand() * (HQ[ind, :] - TempSM[ss, :])
                else:
                    VM += np.random.rand() * (TempSM[ss, :] - HQ[ind, :])
            FolHQ += 1
            if FolHQ >= MaxFolHQ:
                break

        NewBM = TempSM[ss, :] + VM  # New Big Males
        NewBM = trimr(NewBM)  # Limit the values into the given dimensional boundaries
        TempSM[ss, :] = NewBM
        TempSMFX[ss] = evaluation(NewBM)

    # Replace the Big Males with the best ones
    big_males, big_malesFX = replacement(big_males, big_malesFX, TempSM, TempSMFX)

    winnerBM = big_males[0, :]
    winnerFX = big_malesFX[0]

    if winnerFX < femaleFX or np.random.rand() < 0.5:  # Sexual reproduction
        OffSprings = crossover(winnerBM, female)
        fx1 = evaluation(OffSprings[0, :])
        fx2 = evaluation(OffSprings[1, :])

        # Keep the best position of female
        if fx1 < fx2:
            if fx1 < femaleFX:
                female = OffSprings[0, :]
                femaleFX = fx1
        else:
            if fx2 < femaleFX:
                female = OffSprings[1, :]
                femaleFX = fx2
    else:  # Asexual reproduction
        newFemale = mutation()
        fx = evaluation(newFemale)

        # Keep the best position of female
        if fx < femaleFX:
            female = newFemale
            femaleFX = fx
    return big_males, big_malesFX, female, femaleFX

def crossover(n_var,parent1, parent2):
    """
    Crossover operator
    :param n_var: number of variables
    :param parent1: parent 1
    :param parent2: parent 2
    :return: offsprings
    """
    Offsprings = np.zeros((2, n_var))  # Initialize Offsprings
    
    for ii in range(n_var):
        rval = np.random.rand()  # Generate a random value in each dimension
        Offsprings[0, ii] = rval * parent1[ii] + (1 - rval) * parent2[ii]
        Offsprings[1, ii] = rval * parent2[ii] + (1 - rval) * parent1[ii]
    
    # Limit the values into the given dimensional boundaries
    Offsprings[0, :] = trimr(Offsprings[0, :])
    Offsprings[1, :] = trimr(Offsprings[1, :])
    
    return Offsprings


def mutation(female, n_var, cons_ub, cons_lb, mut_rate, mut_radius):
    new_female = female.copy()  # Initialize a new Female
    max_step = mut_radius * (cons_ub - cons_lb)  # Maximum step of the Female mutation
    
    for ii in range(n_var):
        if np.random.rand() < mut_rate:  # Check if a random value is lower than the Mutation Rate
            new_female[ii] = female[ii] + (2 * np.random.rand() - 1) * max_step[ii]
    
    # Limit the values into the given dimensional boundaries
    new_female = trimr(new_female)
    
    return new_female

def replacement(X, FX, Y, FY):
    """
    Replace the worst individuals with the best ones
    :param X: population
    :param XFX: fitness value
    :param Y: population
    :param YFX: fitness value
    :return: population
    """
    LX = X.shape[0]  # Number of individuals in the old population
    XY = np.vstack((X, Y))  # Joint individuals of the old and new population
    FXFY = np.hstack((FX, FY))  # Joint fitness values of the old and new population

    SortedInd = np.argsort(FXFY)  # Sort all fitness values
    Z = XY[SortedInd[:LX], :]  # Select the best individuals
    FZ = FXFY[SortedInd[:LX]]  # Select the best fitness

    return Z, FZ


def trimr(X, n_var, cons_ub, cons_lb):
    """
    Limit the values into the given dimensional boundaries
    :param X: population
    :param n_var: number of variables
    :param cons_ub: upper bound of constraints
    :param cons_lb: lower bound of constraints
    :return: population
    """
    for ii in range(n_var):
        X[X[:, ii] < cons_lb[ii], ii] = cons_lb[ii]
        X[X[:, ii] > cons_ub[ii], ii] = cons_ub[ii]
    Z = X.copy()
    return Z


# runner
if __name__ == "__main__":
    function_id = 1
    dimension = 50
    pop_size = 5

    n_var, cons_ub, cons_lb, mv = get_function(function_id, dimension)

    cons_ub = np.ones(n_var) * cons_ub
    cons_lb = np.ones(n_var) * cons_lb

    population = pop_cons_initialization(pop_size, n_var, cons_ub, cons_lb)
    # print(population)
    fx = np.zeros(pop_size)
    for i in range(pop_size):
        fx[i] = evaluation(population[i, :], function_id)
    # print(fx)

    # sort the individual
    sorted_fx = fx.sort(axis=1)
    ind_fx = fx.argsort(axis=1)

    fx = sorted_fx
    population = population[ind_fx, :]

    one_elit_fx = fx[0]

    # setting the parameters
    max_adaptive_population = pop_size * 40
    num_big_males = np.floor(pop_size / 2)
    mlipir_rate = (n_var - 1) / n_var
    mutation_rate = 0.5
    mutation_radius = 0.5




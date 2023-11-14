import numpy as np
import random
import math


def get_function(function_id, dimension):
    """
    :param function_id: function id
    :param dimension: dimension of the problem
    :return: upper bound, lower bound, number of constraints
    """
    n_var = dimension

    # switch
    match function_id:
        case 1:
            cons_ub = 100
            cons_lb = -100
            mv = 0
        case 2:
            cons_ub = 10
            cons_lb = -10
            mv = 0
        case 3:
            cons_ub = 100
            cons_lb = -100
            mv = 0
        case 4:
            cons_ub = 100
            cons_lb = -100
            mv = 0
        case 5:
            cons_ub = 30
            cons_lb = -30
            mv = 0
        case 6:
            cons_ub = 100
            cons_lb = -100
            mv = 0
        case 7:
            cons_ub = 1.28
            cons_lb = -1.28
            mv = 0
        case 8:
            cons_ub = 500
            cons_lb = -500
            mv = -418.9829 * n_var
        case 9:
            cons_ub = 5.12
            cons_lb = -5.12
            mv = 0
        case 10:
            cons_ub = 32
            cons_lb = -32
            mv = 0
        case 11:
            cons_ub = 600
            cons_lb = -600
            mv = 0
        case 12:
            cons_ub = 50
            cons_lb = -50
            mv = 0
        case 13:
            cons_ub = 50
            cons_lb = -50
            mv = 0
        case 14:
            n_var = 2
            cons_ub = 65
            cons_lb = -65
            mv = 0.998
        case 15:
            n_var = 4
            cons_ub = 5
            cons_lb = -5
            mv = 0.0003
        case 16:
            n_var = 2
            cons_ub = 5
            cons_lb = -5
            mv = -1.0316

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
            # sphere
            fx = np.sum(X**2)
        case 2:
            # schwefel 2.22
            fx = np.sum(np.abs(X)) + np.prod(np.abs(X))
        case 3:
            # schwefel 1.2
            fx = 0
            for ii in range(dim):
                fx = fx + np.sum(X[:ii]) ** 2
        case 4:
            # schewefel 2.21
            fx = np.max(np.abs(X))
        case 5:
            # rosenbrock function
            # fx=sum(100*(x(2:Dim)-(x(1:Dim-1).^2)).^2+(x(1:Dim-1)-1).^2);
            # TODO: check the formula
            fx = np.sum(
                100 * (X[1:dim] - (X[: dim - 1] ** 2)) ** 2 + (X[: dim - 1] - 1) ** 2
            )

        case 6:
            # step function
            fx = np.sum(np.floor(X + 0.5) ** 2)
        case 7:
            # quartic function
            # TODO: check the formula
            fx = np.sum(np.arange(1, dim + 1) * X**4) + np.random.rand()
        case 8:
            # schwefel function
            fx = np.sum(-X * np.sin(np.sqrt(np.abs(X))))
        case 9:
            # rastrigin function
            fx = np.sum(X**2 - 10 * np.cos(2 * np.pi * X) + 10 * dim)
        case 10:
            # ackley function
            # TODO: check the formula
            fx = (
                -20 * np.exp(-0.2 * np.sqrt(np.sum(X**2) / dim))
                - np.exp(np.sum(np.cos(2 * np.pi * X)) / dim)
                + 20
                + np.exp(1)
            )
        case 11:
            # fx=sum(x.^2)/4000-prod(cos(x./sqrt([1:Dim])))+1;
            # TODO: check the formula
            fx = (
                np.sum(X**2) / 4000
                - np.prod(np.cos(X / np.sqrt(np.arange(1, dim + 1))))
                + 1
            )
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
            Temp = np.zeros((1, n_var))
            for i in range(int(np.floor(n_var / 2))):
                Temp[:, i] = cons_lb[:, i] + (cons_ub[:, i] - cons_lb[:, i]) * (
                    f1[ss] + (np.random.rand() * 2 - 1) * 0.01
                )
            for i in range(int(np.floor(n_var / 2)) + 1, n_var):
                Temp[:, i] = cons_lb[:, i] + (cons_ub[:, i] - cons_lb[:, i]) * (
                    f2[ss] + (np.random.rand() * 2 - 1) * 0.01
                )

            population[IndX, :] = Temp
            IndX += 1
            ss += 1

    return population


def move_big_males_female_first_stage(big_males, big_malesFX, female, femaleFX, n_var):
    """
    move big males closer to the optimum value and then the female will reproduce or mutate (asexual reproduction)
    :param big_males, big_males_fx: big_males value and it's fitness value
    :param female, female_fx: female value and it's fitness value
    :param n_var: number of values in given problem
    :return the current position after moving for big_males and female
    """
    HQ = big_males.copy()
    HQFX = big_malesFX.copy()

    TempSM = big_males.copy()
    TempSMFX = big_malesFX.copy()

    for ss in range(TempSM.shape[0]):
        MaxFolHQ = np.random.randint(1, 3)
        VM = np.zeros((1, n_var))  # Velocity of a strong male
        RHQ = np.random.permutation(HQ.shape[0])
        FolHQ = 0

        for fs in range(RHQ.shape[0]):
            ind = RHQ[fs]
            if ind != ss:
                # Semi-randomly select an individual to define attraction or distraction
                if HQFX[ind] < TempSMFX[ss] or np.random.rand() < 0.5:
                    VM = VM + np.random.rand() * (HQ[ind, :] - TempSM[ss, :])
                else:
                    VM = VM + np.random.rand() * (TempSM[ss, :] - HQ[ind, :])
            FolHQ += 1
            if FolHQ >= MaxFolHQ:
                break
        NewBM = TempSM[ss, :] + VM  # New Big Males
        NewBM = trimr(
            NewBM, n_var, cons_ub, cons_lb
        )  # Limit the values into the given dimensional boundaries
        TempSM[ss, :] = NewBM
        TempSMFX[ss] = evaluation(NewBM, function_id)

    # Replace the Big Males with the best ones
    big_males, big_malesFX = replacement(big_males, big_malesFX, TempSM, TempSMFX)

    winnerBM = big_males[0, :].reshape(1, -1)
    winnerFX = big_malesFX[0]

    if winnerFX < femaleFX or np.random.rand() < 0.5:  # Sexual reproduction
        OffSprings = crossover(n_var, winnerBM, female, cons_ub, cons_lb)
        fx1 = evaluation(OffSprings[0, :], function_id)
        fx2 = evaluation(OffSprings[1, :], function_id)

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
        newFemale = mutation(
            female, n_var, cons_ub, cons_lb, mutation_rate, mutation_radius
        )
        fx = evaluation(newFemale, function_id)

        # Keep the best position of female
        if fx < femaleFX:
            female = newFemale
            femaleFX = fx

    return big_males, big_malesFX, female, femaleFX


def move_small_males_first_stage(
    mlipir_rate, big_males, small_males, small_males_fx, n_var, function_id
):
    """
    move the small males based on the dimensional size of MLIPIR
    :param mlipir_rate: rate of MLIPIR
    :param big_males: big males value so far as the reference point
    :param small_males, small_males_fx: small males and it's fitness value
    :return: moved small males and it's fitness value
    """
    HQ = big_males.copy()
    temp_weak_males = small_males.copy()
    temp_weak_males_fx = small_males_fx.copy()
    max_fol_hq = 1

    for ww in range(small_males.shape[0]):
        vector_mlipir_velocity = np.zeros((1, n_var))  # vector of MLIPIR velocity
        rhq = np.random.permutation(HQ.shape[0])  # random permutation of HQ
        fol_hq = 0  # number of following HQ

        for fs in range(len(rhq)):  # for each HQ
            ind = rhq[fs]  # index of HQ
            movement_attribute = np.random.permutation(
                n_var
            )  # random permutation of movement attribute
            dimensional_size_mlipir = round(
                mlipir_rate * n_var
            )  # number of dimensional size of MLIPIR
            if dimensional_size_mlipir >= n_var:
                dimensional_size_mlipir = n_var - 1
            if dimensional_size_mlipir < 1:
                dimensional_size_mlipir = 1

            # move the weak males based on the dimensional size of MLIPIR
            move_weak_males = movement_attribute[:dimensional_size_mlipir].reshape(
                1, -1
            )
            binary_pattern = np.zeros((1, n_var))
            binary_pattern[:, move_weak_males] = 1

            vector_mlipir_velocity = (
                vector_mlipir_velocity
                + np.random.rand(n_var) * (HQ[ind, :] * binary_pattern)
                - (small_males[ww, :] * binary_pattern)
            )

            fol_hq += 1
            if fol_hq >= max_fol_hq:
                break

        new_small_males = small_males[ww, :] + vector_mlipir_velocity
        new_small_males = trimr(new_small_males, n_var, cons_ub, cons_lb)
        temp_weak_males[ww, :] = new_small_males
        temp_weak_males_fx[ww] = evaluation(new_small_males, function_id)

    small_males = temp_weak_males
    small_males_fx = temp_weak_males_fx

    return small_males, small_males_fx


def move_small_males_second_stage(
    mlipir_rate, big_males, all_hq, small_males, small_males_fx, n_var
):
    if len(all_hq) != 0:
        hq = np.vstack((big_males, all_hq))
    else:
        hq = big_males.copy()

    temp_weak_males = small_males.copy()
    temp_weak_males_fx = small_males_fx.copy()

    for ww in range(small_males.shape[0]):
        max_fol_hq = np.random.randint(1, 3)
        vector_mlipir_velocity = np.zeros((1, n_var))
        RHQ = np.random.permutation(hq.shape[0])
        fol_hq = 0
        for fs in range(len(RHQ)):
            individual = RHQ[fs]
            attributes_movement = np.random.permutation(n_var)
            dimensional_size_mlipir = round(mlipir_rate * n_var)
            if dimensional_size_mlipir >= n_var:
                dimensional_size_mlipir = n_var - 1
            if dimensional_size_mlipir < 1:
                dimensional_size_mlipir = 1
            movement = attributes_movement[:dimensional_size_mlipir].reshape(1, -1)
            binary_pattern = np.zeros((1, n_var))
            binary_pattern[:, movement] = 1
            vector_mlipir_velocity = (
                vector_mlipir_velocity
                + np.random.rand(n_var) * (hq[individual, :] * binary_pattern)
                - (small_males[ww, :] * binary_pattern)
            )
            fol_hq += 1
            if fol_hq >= max_fol_hq:
                break
        new_small_males = small_males[ww, :] + vector_mlipir_velocity
        new_small_males = trimr(new_small_males, n_var, cons_ub, cons_lb)
        temp_weak_males[ww, :] = new_small_males
        temp_weak_males_fx[ww] = evaluation(new_small_males, function_id)
    small_males = temp_weak_males
    small_males_fx = temp_weak_males_fx
    return small_males, small_males_fx


def move_big_males_female_second_stage(
    big_males, big_males_fx, all_hq, all_hq_fx, female, female_fx, n_var, function_id
):
    """ """
    if len(all_hq) != 0:
        global_hq = np.vstack((big_males, all_high_quality_big_males))
        global_hq_fx = np.hstack((big_males_fx, all_hq_fx))
    else:
        global_hq = big_males.copy()
        global_hq_fx = big_males_fx.copy()

    temp_sm = big_males.copy()
    temp_sm_fx = big_males_fx.copy()

    for ss in range(temp_sm.shape[0]):
        VM = np.zeros((1, n_var))
        RHQ = np.random.permutation(global_hq.shape[0])
        max_fol_hq = np.random.randint(1, 3)
        fol_hq = 0

        for fs in range(RHQ.shape[0]):
            individual = RHQ[fs]
            if individual != ss:
                # select randomly individual to define attraction or distraction
                if global_hq_fx[individual] < temp_sm_fx[ss] or np.random.rand() < 0.5:
                    VM = VM + np.random.rand() * (
                        global_hq[individual, :] - temp_sm[ss, :]
                    )
                else:
                    VM = VM + np.random.rand() * (
                        temp_sm[ss, :] - global_hq[individual, :]
                    )
            fol_hq += 1
            if fol_hq >= max_fol_hq:
                break
        new_big_males = temp_sm[ss, :] + VM
        new_big_males = trimr(new_big_males, n_var, cons_ub, cons_lb)
        temp_sm[ss, :] = new_big_males
        temp_sm_fx[ss] = evaluation(new_big_males, function_id)

    big_males, big_males_fx = replacement(big_males, big_males_fx, temp_sm, temp_sm_fx)

    winnerBM = big_males[0, :].reshape(1, -1)
    winnerFX = big_males_fx[0]

    if winnerFX < female_fx or np.random.rand() < 0.5:  # sexual reproduction
        Offsprings = crossover(n_var, winnerBM, female, cons_ub, cons_lb)
        fx1 = evaluation(Offsprings[0, :], function_id)
        fx2 = evaluation(Offsprings[1, :], function_id)

        # keep the best population of female
        if fx1 < fx2:
            if fx1 < female_fx:
                female = Offsprings[0, :]
                female_fx = fx1
        else:
            if fx2 < female_fx:
                female = Offsprings[1, :]
                female_fx = fx2
    else:  # asexual reproduction
        newFemale = mutation(
            female, n_var, cons_ub, cons_lb, mutation_rate, mutation_radius
        )
        fx = evaluation(newFemale, function_id)

        # keep the best position of female
        if fx < female_fx:
            female = newFemale
            female_fx = fx

    return big_males, big_males_fx, female, female_fx


def crossover(n_var, parent1, parent2, cons_ub, cons_lb):
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
        Offsprings[0, ii] = rval * parent1[0, ii] + (1 - rval) * parent2[0, ii]
        Offsprings[1, ii] = rval * parent2[0, ii] + (1 - rval) * parent1[0, ii]

    # Limit the values into the given dimensional boundaries
    Offsprings[0, :] = trimr(Offsprings[0, :], n_var, cons_ub, cons_lb)
    Offsprings[1, :] = trimr(Offsprings[1, :], n_var, cons_ub, cons_lb)

    return Offsprings


def mutation(female, n_var, cons_ub, cons_lb, mut_rate, mut_radius):
    new_female = female.copy()  # Initialize a new Female
    max_step = mut_radius * (cons_ub - cons_lb)  # Maximum step of the Female mutation

    for ii in range(n_var):
        if (
            np.random.rand() < mut_rate
        ):  # Check if a random value is lower than the Mutation Rate
            new_female[:, ii] = (
                female[:, ii] + (2 * np.random.rand() - 1) * max_step[:, ii]
            )

    # Limit the values into the given dimensional boundaries
    new_female = trimr(new_female, n_var, cons_ub, cons_lb)

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

    SortedVal, SortedInd = sorted(FXFY), np.argsort(FXFY)  # Sort all fitness values
    Z = XY[SortedInd[:LX], :]  # Select the best individuals
    FZ = SortedVal[:LX]  # Select the best fitness

    return Z, FZ


def reposition(
    X, FX, n_var, cons_ub, cons_lb, function_id, mutation_rate, mutation_radius
):
    tempX = X.copy()
    tempX = tempX.reshape(1, -1)
    X = X.reshape(1, -1)
    max_step = mutation_radius * (cons_ub - cons_lb)
    for ii in range(n_var):
        if np.random.rand() < mutation_rate:
            tempX[:, ii] = (
                X[:, ii]
                + (2 * np.random.rand() - 1) * mutation_radius * max_step[:, ii]
            )
    tempX = trimr(tempX, n_var, cons_ub, cons_lb)
    tempFX = evaluation(tempX, function_id)

    if tempFX < FX:
        newX = tempX
        newFX = tempFX
    else:
        newX = X
        newFX = FX
    return newX, newFX


def trimr(X, n_var, cons_ub, cons_lb):
    """
    Limit the values into the given dimensional boundaries
    :param X: population
    :param n_var: number of variables
    :param cons_ub: upper bound of constraints
    :param cons_lb: lower bound of constraints
    :return: population
    """
    X = X.reshape(1, -1)
    for ii in range(n_var):
        X[X[:, ii] < cons_lb[:, ii], ii] = cons_lb[:, ii]
        X[X[:, ii] > cons_ub[:, ii], ii] = cons_ub[:, ii]
    Z = X.copy()
    return Z


def levy(n, m, beta):
    num = math.gamma(1 + beta) * np.sin(np.pi * beta / 2)

    den = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)

    sigma_u = (num / den) ** (1 / beta)

    # random normal
    u = np.random.normal(0, sigma_u, size=(n, m))
    v = np.random.normal(0, 1, size=(n, m))

    z = u / np.power(np.abs(v), 1 / beta)

    return z


def adding_population(population, n_var, cons_ub, cons_lb, function_id):
    new_population = population + (0.05 * levy(1, n_var, 1.5)) * np.abs(
        cons_ub - cons_lb
    )
    new_population = trimr(new_population, n_var, cons_ub, cons_lb)
    new_population_fx = evaluation(new_population, function_id)
    return new_population, new_population_fx


# runner
if __name__ == "__main__":
    function_id = 6  # identity of the benchmark function
    dimension = 50  # dimension can scaled up to thousands for the functions f1-f13, but it is fixed for f14-f23
    max_num_evaluation = 25000  # maximum number of evaluations
    pop_size = 5  # population size (number of komodo individuals)
    min_adaptive_population = pop_size * 4  # minimum adaptive population size
    max_adaptive_population = pop_size * 40  # maximum adaptive population size

    n_var, cons_ub, cons_lb, f_treshold_fx = get_function(function_id, dimension)

    cons_ub = np.ones(n_var) * cons_ub
    cons_lb = np.ones(n_var) * cons_lb

    # reshape to (1,50)
    cons_ub = cons_ub.reshape(1, -1)
    cons_lb = cons_lb.reshape(1, -1)

    population = pop_cons_initialization(pop_size, n_var, cons_ub, cons_lb)

    # print(population)
    fx = np.zeros((1, pop_size))
    for ii in range(pop_size):
        fx[0, ii] = evaluation(population[ii, :], function_id)

    # sort the individual
    sorted_fx = np.sort(fx).flatten()
    ind_fx = np.argsort(fx).flatten()

    fx = sorted_fx
    population = population[ind_fx, :]

    one_elit_fx = fx[0]

    # setting the parameters
    max_adaptive_population = pop_size * 40
    num_big_males = int(np.floor(pop_size / 2))
    mlipir_rate = (n_var - 1) / n_var
    mutation_rate = 0.5
    mutation_radius = 0.5

    # first stage
    # examining if the benchmark function is simple or complex

    is_global = 0  # boolean to check if the global optimum is found
    improve_rate = 0  # improvement rate to examine the benchmark function
    num_evaluation = 0  # number of evaluation
    generation = 0  # number of generation
    max_generation_exam_1 = 100  # maximum number of generation of the first examination
    max_generation_exam_2 = (
        1000  # maximum number of generation of the second examination
    )

    f_opt = []  # optimal fitness value each generation
    f_mean = []  # mean fitness value each generation
    evo_population_size = []  # population size each generation
    gen_improve = 0  # generation counter to check the improvement rate condition

    while generation < max_generation_exam_2:
        generation += 1  # increase the generation counter
        num_evaluation += pop_size  # increase the number of evaluation

        big_males = population[:num_big_males, :]
        big_males_fx = fx[0:num_big_males]

        female = population[num_big_males, :].reshape(1, -1)
        female_fx = fx[num_big_males]

        small_males = population[num_big_males + 1 :, :]
        small_males_fx = fx[num_big_males + 1 :]

        # Move the BigMales and Female as well in the first stage
        big_males, big_males_fx, female, female_fx = move_big_males_female_first_stage(
            big_males, big_males_fx, female, female_fx, n_var
        )

        # Move (Mlipir) SmallMales in the first stage
        small_males, small_males_fx = move_small_males_first_stage(
            mlipir_rate, big_males, small_males, small_males_fx, n_var, function_id
        )

        population = np.vstack((big_males, female, small_males))
        fx = np.hstack((big_males_fx, female_fx, small_males_fx))

        sorted_fx, ind_fx = sorted(fx), fx.argsort(axis=0)

        fx = sorted_fx
        population = population[ind_fx, :]

        best_individual = population[0, :]
        optimum_value = fx[0]

        f_opt.append(optimum_value)
        f_mean.append(np.mean(fx))
        evo_population_size.append(population.shape[0])

        if optimum_value < one_elit_fx:
            one_elit_fx = optimum_value
            gen_improve += 1
            improve_rate = gen_improve / generation
        if optimum_value <= f_treshold_fx:
            is_global = 1
            break
        if generation == max_generation_exam_1:
            if improve_rate < 0.5:
                is_global = 0
                break

    # second stage
    if (not is_global) and num_evaluation <= max_num_evaluation:
        first_stage_population = (
            population.copy()
        )  # keep five best individuals from the first stage
        first_stage_population_fx = (
            fx.copy()
        )  # keep five fitness value from the first stage
        first_stage_population_fx = np.array(first_stage_population_fx).reshape(1, -1)
        swarm_size = population.shape[0]  # swarm size
        num_big_males = int(np.floor(swarm_size / 2))

        increment_adaptive_population = swarm_size  # increment adaptive population
        decrement_adaptive_population = swarm_size  # decrement adaptive population

        mlipir_rate = 0.5
        max_generation_improve = 2
        max_generation_stagnan = 2

        gen_improve = 0
        gen_stagnan = 0

        constrained_population = pop_cons_initialization(
            max_adaptive_population - swarm_size, n_var, cons_ub, cons_lb
        )
        constrained_population_fx = np.zeros((1, constrained_population.shape[0]))
        for ii in range(constrained_population.shape[0]):
            constrained_population_fx[0, ii] = evaluation(
                constrained_population[ii, :], function_id
            )

        # combine the population of the first stage and the constrained population
        population = np.vstack((population, constrained_population))
        pop_size = population.shape[0]
        fx = np.hstack((first_stage_population_fx, constrained_population_fx))
        fx = fx.reshape(-1)
        one_elit_fx = min(fx)

        while num_evaluation < max_num_evaluation:
            adaptive_pop_size = population.shape[0]
            all_high_quality_big_males = np.empty((0, 50))
            all_high_quality_big_males_fx = []

            for individu in range(0, adaptive_pop_size, swarm_size):
                # MicroSwarm    = Pop(ind:ind+SwarmSize-1,:);
                microswarm = population[individu : individu + swarm_size, :]
                microswarm_fx = fx[individu : individu + swarm_size]
                ind_fx = microswarm_fx.argsort(axis=0)
                microswarm = microswarm[ind_fx, :]
                microswarm_fx = microswarm_fx[ind_fx]

                # # handling error for first iteration
                # if all_high_quality_big_males.size == 0:
                #     continue

                all_high_quality_big_males = np.vstack(
                    (all_high_quality_big_males, microswarm[:num_big_males, :])
                )
                all_high_quality_big_males_fx = np.hstack(
                    (all_high_quality_big_males_fx, microswarm_fx)
                )

            for individu in range(0, adaptive_pop_size, swarm_size):
                microwarm = population[individu : individu + swarm_size, :]
                microswarm_fx = fx[individu : individu + swarm_size]
                ind_fx = microswarm_fx.argsort(axis=0)
                microswarm = microswarm[ind_fx, :]
                microswarm_fx = microswarm_fx[ind_fx]

                big_males = microswarm[:num_big_males]
                big_males_fx = microswarm_fx[0:num_big_males]

                female = microswarm[num_big_males, :].reshape(1, -1)
                female_fx = microswarm_fx[num_big_males]

                small_males = microswarm[num_big_males + 1 :, :]
                small_males_fx = microswarm_fx[num_big_males + 1 :]

                (
                    big_males,
                    big_males_fx,
                    female,
                    female_fx,
                ) = move_big_males_female_second_stage(
                    big_males,
                    big_males_fx,
                    all_high_quality_big_males,
                    all_high_quality_big_males_fx,
                    female,
                    female_fx,
                    n_var,
                    function_id,
                )

                small_males, small_males_fx = move_small_males_second_stage(
                    mlipir_rate,
                    big_males,
                    all_high_quality_big_males,
                    small_males,
                    small_males_fx,
                    n_var,
                )

                # Ensure the slice size fits within the existing array or expand if needed
                if individu + num_big_males > all_high_quality_big_males.shape[0]:
                    new_size = (
                        individu + num_big_males - all_high_quality_big_males.shape[0]
                    )
                    padding = np.empty(
                        (new_size, 50)
                    )  # Create an empty array to expand
                    all_high_quality_big_males = np.vstack(
                        (all_high_quality_big_males, padding)
                    )

                # Assign the BigMales to the slice in the array
                all_high_quality_big_males[
                    individu : individu + num_big_males, :
                ] = big_males
                all_high_quality_big_males_fx[
                    individu : individu + num_big_males
                ] = big_males_fx

                # new population
                population[individu : individu + swarm_size] = np.vstack(
                    (big_males, female, small_males)
                )
                fx[individu : individu + swarm_size] = np.hstack(
                    (big_males_fx, female_fx, small_males_fx)
                )

                num_evaluation += swarm_size
                ind_minimum = np.argmin(fx)
                optimum_value = fx[ind_minimum]
                if optimum_value <= f_treshold_fx:
                    break

            # random population
            individu = np.random.permutation(len(fx))
            population = population[individu, :]
            fx = fx[individu]

            # sort the population
            individu_min = np.argmin(fx)
            best_individual = population[individu_min, :]
            optimum_value = fx[individu_min]
            f_opt.append(optimum_value)
            f_mean.append(np.mean(fx))

            # if round the optimum value is equal to the threshold value
            if optimum_value <= f_treshold_fx:
                break

            # self adaptation of population size
            if optimum_value < one_elit_fx:
                gen_improve += 1
                gen_stagnan = 0
                one_elit_fx = optimum_value
            else:
                gen_stagnan += 1
                gen_improve = 0

            # if consecutive generation is improved
            if gen_improve > max_generation_improve:
                adaptive_pop_size -= decrement_adaptive_population
                if adaptive_pop_size < min_adaptive_population:
                    adaptive_pop_size = min_adaptive_population
                sorted_fx, ind_fx = sorted(fx), np.argsort(fx)
                sorted_population = population[ind_fx, :]
                population = sorted_population[:adaptive_pop_size]
                fx = sorted_fx[:adaptive_pop_size]
                gen_improve = 0

            # if consecutive generation is stagnan
            if gen_stagnan > max_generation_stagnan:
                adaptive_pop_size_old = population.shape[0]
                adaptive_pop_size += increment_adaptive_population
                num_add_population = adaptive_pop_size - adaptive_pop_size_old
                if adaptive_pop_size > max_adaptive_population:
                    adaptive_pop_size = adaptive_pop_size_old
                    num_add_population = 0
                if adaptive_pop_size > adaptive_pop_size_old:
                    new_population = np.zeros((num_add_population, n_var))
                    new_population_fx = np.zeros(num_add_population)
                    for nn in range(num_add_population):
                        (
                            new_population[nn, :],
                            new_population_fx[nn],
                        ) = adding_population(
                            best_individual, n_var, cons_ub, cons_lb, function_id
                        )
                    # Pop(end+1:end+size(NewPop,1),:) = NewPop;
                    population[-1 : -1 + new_population.shape[0], :] = new_population
                    fx[-1 : -1 + new_population_fx.shape[0]] = new_population_fx
                    num_evaluation += num_add_population
                else:
                    for nn in range(population.shape[0]):
                        population[nn, :], fx[nn] = reposition(
                            population[nn, :],
                            fx[nn],
                            n_var,
                            cons_ub,
                            cons_lb,
                            function_id,
                            mutation_rate,
                            mutation_radius,
                        )
                    num_evaluation += population.shape[0]
                gen_stagnan = 0
            random_ind = np.random.permutation(population.shape[0])
            fx = fx[random_ind]
            population = population[random_ind, :]

            evo_population_size = np.hstack((evo_population_size, adaptive_pop_size))
            generation += 1
    print("Function ID\t= ", f"F{function_id}")
    print("Dimension\t= ", dimension)
    print("Number of evaluation\t= ", num_evaluation)
    print("Global optimum\t= ", f_treshold_fx)
    print("Actual solution\t= ", optimum_value)
    print("Best solution\t= ", best_individual)

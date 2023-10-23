import numpy as np
import random


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
        # case 11:
        #     cons_ub = 600
        #     cons_lb = -600
        #     mv = 0
        # case 12:
        #     cons_ub = 50
        #     cons_lb = -50
        #     mv = 0
        # case 13:
        #     cons_ub = 50
        #     cons_lb = -50
        #     mv = 0
        # case 14:
        #     n_var = 2
        #     cons_ub = 65
        #     cons_lb = -65
        #     mv = 0.998
        # case 15:
        #     n_var = 4
        #     cons_ub = 5
        #     cons_lb = -5
        #     mv = 0.0003
        # case 16:
        #     n_var = 2
        #     cons_ub = 5
        #     cons_lb = -5
        #     mv = -1.0316

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
                fx += np.sum(X[:ii]) ** 2
        case 4:
            # schewefel 2.21
            fx = np.max(np.abs(X))
        case 5:
            # rosenbrock function
            fx = (
                np.sum(100 * X[1:dim] - X[: dim - 1] ** 2) ** 2
                + (X[: dim - 1] - 1) ** 2
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
            fx = (
                -20 * np.exp(-0.2 * np.sqrt(np.sum(X**2) / dim))
                - np.exp(np.sum(np.cos(2 * np.pi * X)) / dim)
                + 20
                + np.exp(1)
            )
        case 11:
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
    """
    move big males closer to the optimum value and then the female will reproduce or mutate (asexual reproduction)
    :param big_males, big_males_fx: big_males value and it's fitness value
    :param female, female_fx: female value and it's fitness value
    :return the current position after moving
    """
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
        NewBM = trimr(
            NewBM, n_var, cons_ub, cons_lb
        )  # Limit the values into the given dimensional boundaries
        TempSM[ss, :] = NewBM
        TempSMFX[ss] = evaluation(NewBM, function_id)

    # Replace the Big Males with the best ones
    big_males, big_malesFX = replacement(big_males, big_malesFX, TempSM, TempSMFX)

    winnerBM = big_males[0, :]
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
    HQ = big_males
    temp_weak_males = small_males.copy()
    temp_weak_males_fx = small_males_fx.copy()
    max_fol_hq = 1

    for ww in range(small_males.shape[0]):
        vector_mlipir_velocity = np.zeros(n_var)  # vector of MLIPIR velocity
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
            move_weak_maes = movement_attribute[:dimensional_size_mlipir]
            binary_pattern = np.zeros(n_var)
            binary_pattern[move_weak_maes] = 1

            vector_mlipir_velocity = vector_mlipir_velocity + np.random.rand(n_var) * (
                HQ[ind, :] * binary_pattern - small_males[ww, :] * binary_pattern
            )
            fol_hq += 1
            if fol_hq >= max_fol_hq:
                break

        new_small_males = small_males[ww, :] + vector_mlipir_velocity
        new_small_males = trimr(new_small_males)
        temp_weak_males[ww, :] = new_small_males
        temp_weak_males_fx[ww] = evaluation(new_small_males, function_id)

    small_males = temp_weak_males
    small_males_fx = temp_weak_males_fx

    return small_males, small_males_fx


def move_small_males_second_stage(
    mlipir_rate, big_males, all_hq, small_males, small_males_fx, n_var
):
    if len(all_hq) != 0:
        hq = [[big_males], [all_hq]]
    else:
        hq = big_males
    temp_weak_males = small_males
    temp_weak_males_fx = small_males_fx

    for ww in range(1, small_males.shape[0] + 1):
        max_fol_hq = np.random.randint(1, 3)
        vector_mlipir_rate = np.zeros(n_var)
        r_hq = np.random.permutation(hq.shape[0])
        fol_hq = 0
        for fs in range(1, r_hq.size[0] + 1):
            individual = r_hq[fs]
            attributes_movement = np.random.permutation(n_var)
            dimensional_size_mlipir = round(mlipir_rate * n_var)
            if dimensional_size_mlipir >= n_var:
                dimensional_size_mlipir = n_var - 1
            if dimensional_size_mlipir < 1:
                dimensional_size_mlipir = 1
            movement = attributes_movement[:dimensional_size_mlipir]
            binary_pattern = np.zeros(n_var)
            binary_pattern[movement] = 1
            vector_mlipir_rate = (
                vector_mlipir_rate
                + np.random.rand(n_var) * (hq[individual, :] * binary_pattern)
                - small_males[ww, :] * binary_pattern
            )
            fol_hq += 1
            if fol_hq >= max_fol_hq:
                break
        new_small_males = small_males[ww, :] + vector_mlipir_rate
        new_small_males = trimr(new_small_males)
        temp_weak_males[ww, :] = new_small_males
        temp_weak_males_fx[ww] = evaluation(new_small_males)
    small_males = temp_weak_males
    small_males_fx = temp_weak_males_fx
    return small_males, small_males_fx


def move_big_males_second_stage(
    big_males, big_males_fx, all_hq, all_hq_fx, female, female_fx, n_var, function_id
):
    """ """
    if len(all_hq) != 0:
        global_hq = [[big_males], [all_hq]]
        global_hq_fx = np.concatenate((big_males, all_hq_fx))
    else:
        global_hq = big_males
        global_hq_fx = big_males_fx

    temp_sm = big_males
    temp_sm_fx = big_males_fx

    for ss in range(1, temp_sm.shape[0]):
        velocity_big_male = np.zeros(n_var)
        r_hq = np.random.permutation(global_hq.shape[0])
        max_fol_hq = np.random.randint(1, 3)
        fol_hq = 0

        for fs in range(1, len(r_hq)):
            individual = r_hq[fs]
            if individual != ss:
                # select randomly individual to define attraction or distraction
                if global_hq_fx[individual] < temp_sm_fx[ss] or np.random.rand() < 0.5:
                    velocity_big_male = velocity_big_male + np.random.rand() * (
                        global_hq[individual, :] - temp_sm[ss, :]
                    )
                else:
                    velocity_big_male = velocity_big_male + np.random.rand() * (
                        temp_sm[ss, :] - global_hq[individual, :]
                    )
            fol_hq += 1
            if fol_hq >= max_fol_hq:
                break
        new_big_males = temp_sm[ss, :] + velocity_big_male
        new_big_males = trimr(new_big_males)
        temp_sm[ss, :] = new_big_males
        temp_sm_fx[ss] = evaluation(new_big_males)
    return female, female_fx


def crossover(n_var, parent1, parent2, cons_ub, cons_lb):
    """
    Crossover operator
    :param n_var: number of variables
    :param parent1: parent 1
    :param parent2: parent 2
    :return: offsprings
    """
    # fix size of parent1 and parent2
    parent1 = np.ravel(parent1)
    parent2 = np.ravel(parent2)

    Offsprings = np.zeros((2, n_var))  # Initialize Offsprings

    for ii in range(n_var):
        rval = np.random.rand()  # Generate a random value in each dimension
        Offsprings[0, ii] = rval * parent1[ii] + (1 - rval) * parent2[ii]
        Offsprings[1, ii] = rval * parent2[ii] + (1 - rval) * parent1[ii]

    # Limit the values into the given dimensional boundaries
    Offsprings[0, :] = trimr(Offsprings[0, :], n_var, cons_ub, cons_lb)
    Offsprings[1, :] = trimr(Offsprings[1, :], n_var, cons_ub, cons_lb)

    return Offsprings


def mutation(female, n_var, cons_ub, cons_lb, mut_rate, mut_radius):
    female = np.ravel(female)
    new_female = female.copy()  # Initialize a new Female
    new_female = np.ravel(new_female)
    max_step = mut_radius * (cons_ub - cons_lb)  # Maximum step of the Female mutation

    for ii in range(n_var):
        if (
            np.random.rand() < mut_rate
        ):  # Check if a random value is lower than the Mutation Rate
            new_female[ii] = female[ii] + (2 * np.random.rand() - 1) * max_step[ii]

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
        if X[ii] > cons_ub[ii]:
            X[ii] = cons_ub[ii]
        if X[ii] < cons_lb[ii]:
            X[ii] = cons_lb[ii]
    Z = X.copy()
    return Z


# runner
if __name__ == "__main__":
    function_id = 1
    dimension = 50
    pop_size = 5

    n_var, cons_ub, cons_lb, f_treshold_fx = get_function(function_id, dimension)

    cons_ub = np.ones(n_var) * cons_ub
    cons_lb = np.ones(n_var) * cons_lb

    population = pop_cons_initialization(pop_size, n_var, cons_ub, cons_lb)
    # print(population)
    fx = np.zeros(pop_size)
    for i in range(pop_size):
        fx[i] = evaluation(population[i, :], function_id)

    # sort the individual
    sorted_fx = sorted(fx)
    ind_fx = fx.argsort(axis=0)

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
        100  # maximum number of generation of the second examination
    )

    f_opt = []  # optimal fitness value each generation
    f_mean = []  # mean fitness value each generation
    evo_population_size = []  # population size each generation
    gen_improve = 0  # generation counter to check the improvement rate condition

    while generation < max_generation_exam_2:
        generation += 1  # increase the generation counter
        num_evaluation += pop_size  # increase the number of evaluation

        big_males = population[:num_big_males]
        big_males_fx = fx[:num_big_males]

        female = population[num_big_males : num_big_males + 1, :]
        female_fx = fx[num_big_males]

        small_males = population[num_big_males + 1 :]
        small_males_fx = fx[num_big_males + 2 :]

        big_males, big_males_fx, female, female_fx = move_big_males_female_first_stage(
            big_males, big_males_fx, female, female_fx, n_var
        )
        # print(big_males, big_males_fx)

        population = np.vstack((big_males, female, small_males))
        fx = np.concatenate((big_males_fx, [female_fx], small_males_fx))

        # sorted_fx, ind_fx = np.sort(fx), fx.argsort(fx)
        # fx = sorted_fx
        # population = population[ind_fx, :]
        # best_individual = population[0, :]
        # optimum_value = fx[0]

        # f_opt.append(optimum_value)
        # f_mean.append(np.mean(fx))
        # evo_population_size.append(population.shape[0])

        # if optimum_value < one_elit_fx:
        #     one_elit_fx = optimum_value
        #     gen_improve += 1
        #     improve_rate = gen_improve / generation
        # if optimum_value <= f_treshold_fx:
        #     is_global = 1
        #     break
        # if generation == max_generation_exam_1:
        #     if improve_rate < 0.5:
        #         is_global = 0
        #         break

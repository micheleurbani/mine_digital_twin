from Mine import *
import simpy, csv, json
from datetime import datetime, timedelta
from statistics import mean
from multiprocessing import Pool

def std(param, time_parameters=None, output=True, for_internal_use=False):
    """
    The function asks for a dictionary where we specify the parameters required by the simulation. Parameters are listed below:

    - preventiveMaintenancePolicy: it can be one of the follwing values:
        -- *condinal_probability* to use the conditional probability rule
        -- *age_based* to use the age-based maintenance rule
    - nTrucks:        the number of trucks simulated
    - nShovels:       the number of shovels simulated
    - nDumpSites:     the number of dumpsites simulated
    - nWorkShops:     the number of workshops simulated
    - shovelPolicy:   a list containing the thresholds for maintenance of shovels
    - truckPolicy:    a list containing the thresholds for maintenance of trucks
    - thresholdsPM:   a dictionary with PM thresholds for trucks and shovels
    - simTime:        the simulation time in minutes
    - seed:           a value for the seed (not relevant if replication of results is not required)
    - initialTime:    the absolute time value from which the experiment is restarted [in minutes]

    In order to continue an experiment previously stopped, it is also possible to specify the maintenance parameters from the last execution of the experiment using the variable *time_parameters*. For each truck and shovel, the dictionary contains:

    - LastMaintenance: the absolute time of the last maintenance interventions [in minutes]
    - NextFault: the absolute time of the next fault [in minutes]

    The function returns two dictionaries.
    The first contains the following statistics:
    - Number of maintenance interventions per each truck and shovel, subdivided in:
        -- preventive interventions
        -- corrective interventions
    - Maintenance history of trucks and shovels
    - More statistics on usage of trucks and shovels
    - Stockpiles levels as a function of time for dumpsites

    A second dictionary contains the information to restore the experiment from the point where it was interrupted. For each truck and shovel the dictionary contains:

    - the absolute time of the last maintenance interventions [in minutes]
    - the absolute time of the next fault [in minutes]

    """
    if type(param['shovelPolicy']) is list:
        assert len(param['shovelPolicy']) == int(param['nShovels'])
    elif type(param['shovelPolicy']) is float:
        param['shovelPolicy'] = [param['shovelPolicy']]
    if type(param['truckPolicy']) is list:
        assert len(param['truckPolicy']) == int(param['nTrucks'])
    elif type(param['truckPolicy']) is float:
        param['truckPolicy'] = [param['truckPolicy']]

    # Create the simulation environment
    env = simpy.Environment(initial_time=param['initialTime'])
    env.statistics = dict()

    with open("data/workshops_data.csv","r",newline="\n") as f:
        doc = csv.reader(f,delimiter=",",quoting=csv.QUOTE_NONNUMERIC)
        workshopsData = [x for x in doc]
    workshops = [
        WorkShop(env,i,(workshopsData[i][0],workshopsData[i][1]))
        for i in range(int(param['nWorkShops']))]

    with open("data/shovels_data.csv","r",newline="\n") as f:
        doc = csv.reader(f,delimiter=",",quoting=csv.QUOTE_NONNUMERIC)
        shovelsData = [x for x in doc]
    shovels = [
        Shovel(
            env,
            i,
            coordinates=(shovelsData[i][0],shovelsData[i][1]),
            mu=shovelsData[i][2],
            sigma=shovelsData[i][3],
            muPreventive=shovelsData[i][4],
            sigmaPreventive=shovelsData[i][5],
            muCorrective=shovelsData[i][6],
            sigmaCorrective=shovelsData[i][7],
            alpha=shovelsData[i][8],
            beta=shovelsData[i][9],
            Cc=shovelsData[i][10],
            Cp=shovelsData[i][11],
            p=param['shovelPolicy'][i],
            workshops=workshops)
        for i in range(int(param['nShovels']))]
    if param['initialTime'] > 0:
        for shovel in shovels:
            shovel.lastMaintenance = time_parameters["Shovel%d"%shovel.id]["LastMaintenance"]
            shovel.nextFault = time_parameters["Shovel%d"%shovel.id]["NextFault"]

    with open("data/dumpsites_data.csv","r",newline="\n") as f:
        doc = csv.reader(f,delimiter=",",quoting=csv.QUOTE_NONNUMERIC)
        dumpsiteData = [x for x in doc]
    dumpsites = [
        DumpSite(
            env,
            i,
            coordinates=(dumpsiteData[i][2],dumpsiteData[i][3]),
            mu=dumpsiteData[i][0],
            sigma=dumpsiteData[i][1],
            maxCapacity=param['maxCapacity']/param['nDumpSites'],
            millRate=param['millRate']/param['nDumpSites'])
        for i in range(int(param['nDumpSites']))]

    with open("data/truck_data.csv","r",newline="\n") as f:
        doc = csv.reader(f,delimiter=",",quoting=csv.QUOTE_NONNUMERIC)
        truckData = [x for x in doc]
    trucks = [Truck(
        alpha=truckData[i][0],
        beta=truckData[i][1],
        muCorrective=truckData[i][6],
        sigmaCorrective=truckData[i][7],
        muPreventive=truckData[i][4],
        sigmaPreventive=truckData[i][5],
        Cc=truckData[i][2],
        Cp=truckData[i][3],
        p=param['truckPolicy'][i],
        env=env,
        id=i,
        shovels=shovels,
        dumpsites=dumpsites,
        workshops=workshops,
        muCapacity=truckData[i][8],
        sigmaCapacity=truckData[i][9])
        for i in range(int(param["nTrucks"]))]
    if param['initialTime'] > 0:
        for truck in trucks:
            truck.lastMaintenance = time_parameters["Truck%d"%truck.id]["LastMaintenance"]
            truck.nextFault = time_parameters["Truck%d"%truck.id]["NextFault"]
    try:
        random.seed(param["seed"])
    except:
        pass
    begin = datetime.now()
    # Stick the policy to the classes Shovel and Trucks
    Shovel.preventiveMaintenanceRule = param['PMRule']
    Truck.preventiveMaintenanceRule = param['PMRule']

    env.run(until=param["initialTime"] + param["simTime"])
    if output:
        print('End')
    processingTime = datetime.now() - begin
    if output:
        print("Processing time ", processingTime)
    time_parameters = dict()
    # Results and statistics update
    for i in range(len(trucks)):
        if output:
            print("Truck%d: \tFailures =" %i, env.statistics["Truck%d" %i]["Failure"], "\t Preventive =", env.statistics["Truck%d" %i]["PreventiveInterventions"])
        s = env.statistics["Truck%d"%i]["Statistics"]
        s["Availability"] = 1 - (s["Failed"] + s["PMRepair"] + s["CMRepair"]) / (param["simTime"])
        time_parameters["Truck%d"%i] = dict()
        time_parameters["Truck%d"%i]["LastMaintenance"] = trucks[i].lastMaintenance
        time_parameters["Truck%d"%i]["NextFault"] = trucks[i].nextFault
    for i in range(len(shovels)):
        if output:
            print("Shovel%d:\tFailures =" %i, env.statistics["Shovel%d" %i]["Failure"], "\t Preventive =", env.statistics["Shovel%d" %i]["PreventiveInterventions"])
        s = env.statistics["Shovel%d"%i]["Statistics"]
        s["Availability"] = 1 - (s["Failed"] + s["PMRepair"] + s["CMRepair"] + s['TravelTime']) / (param["simTime"])
        s["IdleTime"] = param["simTime"] - s["WorkingTime"] - s["TravelTime"] - s["TimeInQueue"] - s["Failed"] - s["PMRepair"] - s["CMRepair"]
        time_parameters["Shovel%d"%i] = dict()
        time_parameters["Shovel%d"%i]["LastMaintenance"] = shovels[i].lastMaintenance
        time_parameters["Shovel%d"%i]["NextFault"] = shovels[i].nextFault

    if for_internal_use:
        return env.statistics
    return json.dumps(env.statistics), json.dumps(time_parameters)

def test(SIM_TIME,seed):
    """The function run a single instance of the simulation experiment.
    Input parameters:

    :param int SIM_TIME: the value of the simulation horizon expressed in minutes
    :param int seed: a value for the seed. It is used to reproduce the experiment with the same results.
    :return:

    * Number of maintenance interventions per each truck and shovel, subdivided in:

        * preventive interventions
        * corrective interventions

    * Stockpiles levels in time

    :rtype: dict
    """
    env = simpy.Environment()
    env.statistics = dict()
    # WORKSHOPS DECLARATION
    with open("data/workshops_data.csv","r",newline="\n") as f:
        doc = csv.reader(f,delimiter=",",quoting=csv.QUOTE_NONNUMERIC)
        workshopsData = [x for x in doc]
    workshops = [
        WorkShop(env,i,(workshopsData[i][0],workshopsData[i][1]))
        for i in range(2)]

    # SHOVEL DECLARATION
    with open("data/shovels_data.csv","r",newline="\n") as f:
        doc = csv.reader(f,delimiter=",",quoting=csv.QUOTE_NONNUMERIC)
        shovelsData = [x for x in doc]
    shovels = [
        Shovel(
            env,
            i,
            coordinates=(shovelsData[i][0],shovelsData[i][1]),
            mu=shovelsData[i][2],
            sigma=shovelsData[i][3],
            muPreventive=shovelsData[i][4],
            sigmaPreventive=shovelsData[i][5],
            muCorrective=shovelsData[i][6],
            sigmaCorrective=shovelsData[i][7],
            alpha=shovelsData[i][8],
            beta=shovelsData[i][9],
            Cc=shovelsData[i][10],
            Cp=shovelsData[i][11],
            p=0.004,
            workshops=workshops)
        for i in range(2)]

    # DUMPSITES DECLARATION
    with open("data/dumpsites_data.csv","r",newline="\n") as f:
        doc = csv.reader(f,delimiter=",",quoting=csv.QUOTE_NONNUMERIC)
        dumpsiteData = [x for x in doc]
    dumpsites = [
        DumpSite(env,i,coordinates=(dumpsiteData[i][2],dumpsiteData[i][3]),mu=dumpsiteData[i][0],sigma=dumpsiteData[i][1])
        for i in range(2)]

    # TRUCK DECLARATION
    with open("data/truck_data.csv","r",newline="\n") as f:
        doc = csv.reader(f,delimiter=",",quoting=csv.QUOTE_NONNUMERIC)
        truckData = [x for x in doc]
    trucks = [Truck(
        alpha=truckData[i][0],
        beta=truckData[i][1],
        muCorrective=truckData[i][6],
        sigmaCorrective=truckData[i][7],
        muPreventive=truckData[i][4],
        sigmaPreventive=truckData[i][5],
        Cc=truckData[i][2],
        Cp=truckData[i][3],
        p=0.03,
        env=env,
        id=i,
        shovels=shovels,
        dumpsites=dumpsites,
        workshops=workshops,
        muCapacity=truckData[i][8],
        sigmaCapacity=truckData[i][9]) for i in range(10)]

    random.seed(seed)
    begin = datetime.now()
    env.run(until=SIM_TIME)
    print('End')
    processingTime = datetime.now() - begin
    print("Processing time ", processingTime)
    for i in range(len(trucks)):
        print("Truck%d: Failures =" %i, env.statistics["Truck%d" %i]["Failure"], ",\t Preventive =", env.statistics["Truck%d" %i]["PreventiveInterventions"])
    for i in range(len(shovels)):
        print("Shovel%d: Failures =" %i, env.statistics["Shovel%d" %i]["Failure"], ",\t Preventive =", env.statistics["Shovel%d" %i]["PreventiveInterventions"])

    return env.statistics

def multiTest(nTrial,sim_time):
    """
    The function is for development purposes only. It runs *nTrial* attempts of the simulation with a different seed but using the same simulation parameters.
    """
    nExceptions = 0
    for i in range(nTrial):
        random.seed(sim_time,i)
        try:
            std(i)
        except Exception as e:
            print("Trial %d:" % i, e, "\n")
            nExceptions += 1
    print("There where %d exceptions." % nExceptions)

def fitness(SIM_TIME, seed, thresholds):
    """
    The function run a single instance of the simulation experiment with the given maintenance thresholds and returns the total cost of maintenance.
    By returning the total cost of maintenane we are weighting more corrective interventions.
    The function simulates a system with as many trucks and shovels as the length of the relative thresholds' arrays.
    Input parameters:

    :param int SIM_TIME: the value of the simulation horizon expressed in minutes
    :param int seed: a value for the seed. It is used to reproduce the experiment with the same results.
    :param dict thresholds: a dictionary containing two arrays of thresholds, one for trucks the other for shovels.
    :return: the total cost of maintenance
    :rtype: int

    """
    env = simpy.Environment()
    env.statistics = dict()
    DEBUG = False
    # WORKSHOPS DECLARATION
    with open("data/workshops_data.csv","r",newline="\n") as f:
        doc = csv.reader(f,delimiter=",",quoting=csv.QUOTE_NONNUMERIC)
        workshopsData = [x for x in doc]
    workshops = [
        WorkShop(env,i,(workshopsData[i][0],workshopsData[i][1]))
        for i in range(2)]

    # SHOVEL DECLARATION
    with open("data/shovels_data.csv","r",newline="\n") as f:
        doc = csv.reader(f,delimiter=",",quoting=csv.QUOTE_NONNUMERIC)
        shovelsData = [x for x in doc]
    shovels = [
        Shovel(
            env,
            i,
            coordinates=(shovelsData[i][0],shovelsData[i][1]),
            mu=shovelsData[i][2],
            sigma=shovelsData[i][3],
            muPreventive=shovelsData[i][4],
            sigmaPreventive=shovelsData[i][5],
            muCorrective=shovelsData[i][6],
            sigmaCorrective=shovelsData[i][7],
            alpha=shovelsData[i][8],
            beta=shovelsData[i][9],
            Cc=shovelsData[i][10],
            Cp=shovelsData[i][11],
            p=thresholds['Shovels'][i],
            workshops=workshops)
        for i in range(len(thresholds['Shovels']))]

    # DUMPSITES DECLARATION
    with open("data/dumpsites_data.csv","r",newline="\n") as f:
        doc = csv.reader(f,delimiter=",",quoting=csv.QUOTE_NONNUMERIC)
        dumpsiteData = [x for x in doc]
    dumpsites = [
        DumpSite(env,i,coordinates=(dumpsiteData[i][2],dumpsiteData[i][3]),mu=dumpsiteData[i][0],sigma=dumpsiteData[i][1])
        for i in range(2)]

    # TRUCK DECLARATION
    with open("data/truck_data.csv","r",newline="\n") as f:
        doc = csv.reader(f,delimiter=",",quoting=csv.QUOTE_NONNUMERIC)
        truckData = [x for x in doc]
    trucks = [Truck(
        alpha=truckData[i][0],
        beta=truckData[i][1],
        muCorrective=truckData[i][6],
        sigmaCorrective=truckData[i][7],
        muPreventive=truckData[i][4],
        sigmaPreventive=truckData[i][5],
        Cc=truckData[i][2],
        Cp=truckData[i][3],
        p=thresholds['Trucks'][i],
        env=env,
        id=i,
        shovels=shovels,
        dumpsites=dumpsites,
        workshops=workshops,
        muCapacity=truckData[i][8],
        sigmaCapacity=truckData[i][9])
        for i in range(len(thresholds['Trucks']))]

    if seed is not None:
        random.seed(seed)
    begin = datetime.now()
    # Stick the policy to the classes Shovel and Trucks
    Shovel.preventiveMaintenanceRule = 'age_based'
    Truck.preventiveMaintenanceRule = 'age_based'

    env.run(until=SIM_TIME)
    if DEBUG:
        print('End')
        processingTime = datetime.now() - begin
        print("Processing time ", processingTime)
        for i in range(len(trucks)):
            print("Truck%d: Failures =" %i, env.statistics["Truck%d" %i]["Failure"], ",\t Preventive =", env.statistics["Truck%d" %i]["PreventiveInterventions"])
        for i in range(len(shovels)):
            print("Shovel%d: Failures =" %i, env.statistics["Shovel%d" %i]["Failure"], ",\t Preventive =", env.statistics["Shovel%d" %i]["PreventiveInterventions"])

    sumShovels = sum([env.statistics[item[1]]['Failure'] * shovels[item[0]].Cc +
                    env.statistics[item[1]]['PreventiveInterventions'] * shovels[item[0]].Cp
                    for item in enumerate(env.statistics.keys())
                    if type(env.statistics[item[1]]) is dict and item[1][:-1] == "Shovel"])

    sumTrucks = sum([env.statistics[item[1]]['Failure'] * trucks[int(item[1][-1:])].Cc +
                    env.statistics[item[1]]['PreventiveInterventions'] * trucks[int(item[1][-1:])].Cp
                    for item in enumerate(env.statistics.keys())
                    if type(env.statistics[item[1]]) is dict and item[1][:5] == "Truck"])

    return sumShovels + sumTrucks

def multiFitness(nTrial, SIM_TIME, thresholds):
    scores = list()
    for i in range(nTrial):
        scores.append(fitness(SIM_TIME,seed=None,thresholds=thresholds))
    return mean(scores)

def GA(initialPopSize, items, simTime):
    """
    A genetic algorithm which aim is to optimize maintenance thresholds.

    :param int initialPopSize: the size of the initial population
    :param int items: the total number of items present in the system
    :param int simTime: the length of the simulation horizon in [minutes]
    :return: a touple containing the dictionary with thresholds and the value of the average cost of the maintenance obtained using such thresholds
    :rtype: touple

    """
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    def generateIndividuals(popSize, nGenes):
        """
        Generate a population of new *nIndividuals*.

        :param int popSize: the number of individuals in the population
        :param int nGenes: the number of genes in each individuals
        :return: the new population of size *popSize* as a list of lists
        :rtype: list

        """
        population = list()
        for individual in range(popSize):
            population.append([random.random()*3000 for _ in range(nGenes)])
        return population

    def wheelOfFortune(n, population, scores):
        pdf = [x/sum(scores) for x in scores]
        cdf = list()
        for i in range(len(pdf)):
            if i == 0:
                cdf.append(pdf[i])
            else:
                cdf.append(cdf[-1] + pdf[i])
        parents = list()
        for _ in range(n):
            x = random.random()
            for i in range(len(cdf)):
                if x < cdf[i]:
                    parents.append(population[i])
                    break
        return parents

    def crossover(n, parents):
            """
            The function perform crossover on couples of individuals at a random point. Individuals are randomply selected from a pool of parent individuals.
            """
            x = list()
            for _ in range(int(n/2+1)):
                y = random.sample(parents,k=2)
                q = random.randint(0,len(parents[0])-1)
                if q == 0:
                    x.append([y[0][0]] + y[1][1:])
                    x.append([y[1][0]] + y[0][1:])
                elif q == len(parents[0]):
                    x.append(y[0][q-1] + [y[1][q]])
                    x.append(y[1][q-1] + [y[0][q]])
                else:
                    x.append(y[0][:q] + y[1][q:])
                    x.append(y[1][:q] + y[0][q:])
            return x[:n]

    def mutation(n, parents):
            """
            The function performs a single point mutation with probability 1.
            """
            x = list()
            for _ in range(n):
                # Select a random individual from parents
                y = parents[random.randint(0,len(parents)-1)]
                # Two-points mutation
                for _ in range(2):
                    # Select the mutation point
                    q = random.randint(0,len(y)-1)
                    y[q] += random.normalvariate(mu=0,sigma=0.1)
                x.append(y)
            return x[:n]

    population = generateIndividuals(initialPopSize,items)
    max_generations = 100
    stats = dict()
    stats['best'] = list()
    stats['average'] = list()

    for _ in tqdm(range(max_generations)):

        # Wrap parameters for the simulation
        data = [(20, simTime, {"Shovels": ind[:3], "Trucks": ind[3:]}) for ind in population]

        with Pool(processes=60) as p:
            scores = list(p.starmap(multiFitness, data))

        # Selection of parents
        parents = wheelOfFortune(population=population, scores=scores, n=15)
        # Crossover
        crossovered = crossover(n=20, parents=parents)
        # Mutation
        mutated = mutation(n=40,parents=parents)
        # Order the population and the scores
        zipped_sorted = sorted(zip(population, scores), key=lambda x: x[1])
        population, scores = list(), list()
        for i in range(len(zipped_sorted)):
            population.append(zipped_sorted[i][0])
            scores.append(zipped_sorted[i][1])
        # Collect stats
        stats['best'].append(scores[0])
        stats['average'].append(mean(scores))
        # New population
        population = crossovered + mutated + list(population[:10])

    stats['thresholds'] = population[0]
    stats['score'] = scores[0]

    with open("results.json", "w") as f:
        json.dump(stats, f)

    plt.plot(range(max_generations),stats['best'])
    plt.plot(range(max_generations),stats['average'])
    plt.legend(['Best', 'Average'])
    plt.show()

    return {"Shovels": population[0][:2], "Trucks": population[0][2:]}, scores[0]

def mineMap(thresholds):
    """The function plots the position of sites within the mine."""
    env = simpy.Environment()
    env.statistics = dict()
    DEBUG = False
    # WORKSHOPS DECLARATION
    with open("data/workshops_data.csv","r",newline="\n") as f:
        doc = csv.reader(f,delimiter=",",quoting=csv.QUOTE_NONNUMERIC)
        workshopsData = [x for x in doc]
    workshops = [
        WorkShop(env,i,(workshopsData[i][0],workshopsData[i][1]))
        for i in range(2)]

    # SHOVEL DECLARATION
    with open("data/shovels_data.csv","r",newline="\n") as f:
        doc = csv.reader(f,delimiter=",",quoting=csv.QUOTE_NONNUMERIC)
        shovelsData = [x for x in doc]
    shovels = [
        Shovel(
            env,
            i,
            coordinates=(shovelsData[i][0],shovelsData[i][1]),
            mu=shovelsData[i][2],
            sigma=shovelsData[i][3],
            muPreventive=shovelsData[i][4],
            sigmaPreventive=shovelsData[i][5],
            muCorrective=shovelsData[i][6],
            sigmaCorrective=shovelsData[i][7],
            alpha=shovelsData[i][8],
            beta=shovelsData[i][9],
            Cc=shovelsData[i][10],
            Cp=shovelsData[i][11],
            p=None,
            workshops=workshops)
        for i in range(3)]

    # DUMPSITES DECLARATION
    with open("data/dumpsites_data.csv","r",newline="\n") as f:
        doc = csv.reader(f,delimiter=",",quoting=csv.QUOTE_NONNUMERIC)
        dumpsiteData = [x for x in doc]
    dumpsites = [
        DumpSite(env,i,coordinates=(dumpsiteData[i][2],dumpsiteData[i][3]),mu=dumpsiteData[i][0],sigma=dumpsiteData[i][1])
        for i in range(2)]

    colors = ['orange', 'blue', 'green']
    for site in enumerate([shovels, workshops, dumpsites]):
        for i in enumerate(site[1]):
            plt.scatter(i[1].coordinates[0],i[1].coordinates[1],label=i[1].__class__.__name__+str(i[0]),c=colors[site[0]])
            plt.annotate(i[1].__class__.__name__+str(i[0]),(i[1].coordinates[0],i[1].coordinates[1]))
    plt.title("Sites location on the map")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.legend(loc=9)
    plt.savefig("figures/mine_map.eps")

def output_amount(param, time_parameters=None):
    results = std(param, time_parameters=time_parameters, output=False, for_internal_use=True)
    throughput = 0
    for i in range(2):
        if len(results['DumpSite%d'%i]) > 0:
            throughput += sum([x[1] for x in results['DumpSite%d'%i]])
    return throughput

def calculate_output(n, attempt_param, time_parameters=None):

    def percentile(data, P):
        from math import ceil
        n = int(ceil(P * len(data)))
        return data[n-1]

    with Pool() as p:
        production_outputs = list(p.starmap(output_amount, [(attempt_param,) for _ in range(int(n))]))

    # production_outputs = [output_amount(attempt_param, time_parameters=time_parameters) for _ in range(int(n))]
    return percentile(production_outputs, P=0.95)

def change_configuration(nshovels, ntrucks, param):
    attempt_param = dict(param)
    attempt_param['shovelPolicy'] = param['shovelPolicy'][:nshovels]
    attempt_param['nShovels'] = nshovels
    attempt_param['truckPolicy'] = param['truckPolicy'][:ntrucks]
    attempt_param['nTrucks'] = ntrucks
    return attempt_param

def optimize_configuration(target, n, param, shovels_ub=3, trucks_ub=10, time_parameters=None):
    """
    The function search for the configuration that, using the minimum amount of trucks and shovels, satisfies with the 95% of probability the production target.

    :param float target: the production target in hundreds of kilograms
    :param int n: the number of runs used to perform the 95% probability test
    :param dict param: the dictionary containing the parameters to run the simulation (maintenance thresholds, simulation horizon, etc.)
    :param int shovels_ub: the maximum number of shovels allowed, by default is set equal to three
    :param int trucks_ub: the maximum number of trucks allowed, by default is set equal to ten
    :return: a tuple containing the minimum number of trucks and shovels that guarantees to reach the target throughput with 95% probability
    :rtype: tuple

    """
    shovels_lb, trucks_lb = 1, 1
    nshovels = min([shovels_lb, shovels_ub])
    ntrucks = max([trucks_lb, trucks_ub])

    test_shovels, test = False, False
    i = 0

    attempt_param = change_configuration(nshovels, ntrucks, param)
    # Estimate the production output for the initial configuration.
    guaranteed_output = calculate_output(n, attempt_param)
    print(f"Iteration {i}: ntrucks = {ntrucks}, nshovels = {nshovels}. \t Guaranteed throughput {guaranteed_output} [ton]")
    i += 1

    # Optimize the number of shovels
    while not test_shovels:
        if guaranteed_output > target:
            test_shovels = True
        else:
            if nshovels + 1 <= shovels_ub: nshovels += 1
            else: test_shovels = True
            attempt_param = change_configuration(nshovels, ntrucks, param)
            guaranteed_output = calculate_output(n, attempt_param)

            print(f"Iteration {i}: ntrucks = {ntrucks}, nshovels = {nshovels}. \t Guaranteed throughput {guaranteed_output} [ton]")
            i += 1
    # Optimize the number of trucks
    while not test:
        if guaranteed_output < target:
            if ntrucks + 1 > trucks_ub:
                print(f"Impossible to reach the target with {shovels_ub} shovels and {trucks_ub} trucks.")
                break
            else:
                ntrucks += 1
                attempt_param = change_configuration(nshovels, ntrucks, param)
                guaranteed_output = calculate_output(n, attempt_param)
                test = True
                print(f"Iteration {i}: ntrucks = {ntrucks}, nshovels = {nshovels}. \t Guaranteed throughput {guaranteed_output} [ton]")
                i += 1
        elif guaranteed_output > target:
            if ntrucks - 1 >= trucks_lb:
                ntrucks -= 1
                attempt_param = change_configuration(nshovels, ntrucks, param)
                guaranteed_output = calculate_output(n, attempt_param)
                print(f"Iteration {i}: ntrucks = {ntrucks}, nshovels = {nshovels}. \t Guaranteed throughput {guaranteed_output} [ton]")
                i += 1
            else:
                test = True

    return ntrucks, nshovels

def parametrizedP(a, sim_time, output=False):

    from scipy.special import gamma

    # Create the simulation environment
    env = simpy.Environment()
    env.statistics = dict()

    with open("data/workshops_data.csv","r",newline="\n") as f:
        doc = csv.reader(f,delimiter=",",quoting=csv.QUOTE_NONNUMERIC)
        workshopsData = [x for x in doc]
    workshops = [
        WorkShop(env,i,(workshopsData[i][0],workshopsData[i][1]))
        for i in range(2)]

    with open("data/shovels_data.csv","r",newline="\n") as f:
        doc = csv.reader(f,delimiter=",",quoting=csv.QUOTE_NONNUMERIC)
        shovelsData = [x for x in doc]
    shovels = [
        Shovel(
            env,
            i,
            coordinates=(shovelsData[i][0],shovelsData[i][1]),
            mu=shovelsData[i][2],
            sigma=shovelsData[i][3],
            muPreventive=shovelsData[i][4],
            sigmaPreventive=shovelsData[i][5],
            muCorrective=shovelsData[i][6],
            sigmaCorrective=shovelsData[i][7],
            alpha=shovelsData[i][8],
            beta=shovelsData[i][9],
            Cc=shovelsData[i][10],
            Cp=shovelsData[i][11],
            p=a * shovelsData[i][8] * gamma(1 + 1/shovelsData[i][9]),
            workshops=workshops)
        for i in range(3)]

    with open("data/dumpsites_data.csv","r",newline="\n") as f:
        doc = csv.reader(f,delimiter=",",quoting=csv.QUOTE_NONNUMERIC)
        dumpsiteData = [x for x in doc]
    dumpsites = [
        DumpSite(env,i,coordinates=(dumpsiteData[i][2],dumpsiteData[i][3]),mu=dumpsiteData[i][0],sigma=dumpsiteData[i][1])
        for i in range(2)]

    with open("data/truck_data.csv","r",newline="\n") as f:
        doc = csv.reader(f,delimiter=",",quoting=csv.QUOTE_NONNUMERIC)
        truckData = [x for x in doc]
    trucks = [Truck(
        alpha=truckData[i][0],
        beta=truckData[i][1],
        muCorrective=truckData[i][6],
        sigmaCorrective=truckData[i][7],
        muPreventive=truckData[i][4],
        sigmaPreventive=truckData[i][5],
        Cc=truckData[i][2],
        Cp=truckData[i][3],
        p=a * truckData[i][0] * gamma(1 + 1/truckData[i][1]),
        env=env,
        id=i,
        shovels=shovels,
        dumpsites=dumpsites,
        workshops=workshops,
        muCapacity=truckData[i][8],
        sigmaCapacity=truckData[i][9])
        for i in range(10)]

    begin = datetime.now()
    # Stick the policy to the classes Shovel and Trucks
    Shovel.preventiveMaintenanceRule = "age_based"
    Truck.preventiveMaintenanceRule = "age_based"

    env.run(until=sim_time)

    time_parameters = dict()
    CMcost = 0
    PMcost = 0

    # Results and statistics update
    for i in range(len(trucks)):
        if output:
            print("Truck%d: \tFailures =" %i, env.statistics["Truck%d" %i]["Failure"], "\t Preventive =", env.statistics["Truck%d" %i]["PreventiveInterventions"])
        s = env.statistics["Truck%d"%i]["Statistics"]
        s["Availability"] = 1 - (s["Failed"] + s["PMRepair"] + s["CMRepair"]) / (sim_time)
        del env.statistics["Truck%d"%i]["PreventiveMaintenanceHistory"]
        del env.statistics["Truck%d"%i]["FailureHistory"]
        del env.statistics["Truck%d"%i]["History"]
        CMcost += trucks[i].Cc * env.statistics["Truck%d"%i]["Failure"]
        PMcost += trucks[i].Cp * env.statistics["Truck%d"%i]["PreventiveInterventions"]

    for i in range(len(shovels)):
        if output:
            print("Shovel%d:\tFailures =" %i, env.statistics["Shovel%d" %i]["Failure"], "\t Preventive =", env.statistics["Shovel%d" %i]["PreventiveInterventions"])
        s = env.statistics["Shovel%d"%i]["Statistics"]
        s["Availability"] = 1 - (s["Failed"] + s["PMRepair"] + s["CMRepair"] + s['TravelTime']) / (sim_time)
        s["IdleTime"] = sim_time - s["WorkingTime"] - s["TravelTime"] - s["TimeInQueue"] - s["Failed"] - s["PMRepair"] - s["CMRepair"]
        del env.statistics["Shovel%d"%i]["PreventiveMaintenanceHistory"]
        del env.statistics["Shovel%d"%i]["FailureHistory"]
        del env.statistics["Shovel%d"%i]["History"]
        CMcost += shovels[i].Cc * env.statistics["Shovel%d"%i]["Failure"]
        PMcost += shovels[i].Cp * env.statistics["Shovel%d"%i]["PreventiveInterventions"]

    for i in range(2):
        del env.statistics["DumpSite%d"%i]

    return CMcost, PMcost

def parametrizedCost(a, b, sim_time, output=False):

    from scipy.special import gamma

    # Create the simulation environment
    env = simpy.Environment()
    env.statistics = dict()

    with open("data/workshops_data.csv","r",newline="\n") as f:
        doc = csv.reader(f,delimiter=",",quoting=csv.QUOTE_NONNUMERIC)
        workshopsData = [x for x in doc]
    workshops = [
        WorkShop(env,i,(workshopsData[i][0],workshopsData[i][1]))
        for i in range(2)]

    with open("data/shovels_data.csv","r",newline="\n") as f:
        doc = csv.reader(f,delimiter=",",quoting=csv.QUOTE_NONNUMERIC)
        shovelsData = [x for x in doc]
    shovels = [
        Shovel(
            env,
            i,
            coordinates=(shovelsData[i][0],shovelsData[i][1]),
            mu=shovelsData[i][2],
            sigma=shovelsData[i][3],
            muPreventive=shovelsData[i][4],
            sigmaPreventive=shovelsData[i][5],
            muCorrective=shovelsData[i][6],
            sigmaCorrective=shovelsData[i][7],
            alpha=shovelsData[i][8],
            beta=shovelsData[i][9],
            Cc=shovelsData[i][10],
            Cp=b * shovelsData[i][10],
            p=a * shovelsData[i][8] * gamma(1 + 1/shovelsData[i][9]),
            workshops=workshops)
        for i in range(3)]

    with open("data/dumpsites_data.csv","r",newline="\n") as f:
        doc = csv.reader(f,delimiter=",",quoting=csv.QUOTE_NONNUMERIC)
        dumpsiteData = [x for x in doc]
    dumpsites = [
        DumpSite(env,i,coordinates=(dumpsiteData[i][2],dumpsiteData[i][3]),mu=dumpsiteData[i][0],sigma=dumpsiteData[i][1])
        for i in range(2)]

    with open("data/truck_data.csv","r",newline="\n") as f:
        doc = csv.reader(f,delimiter=",",quoting=csv.QUOTE_NONNUMERIC)
        truckData = [x for x in doc]
    trucks = [Truck(
        alpha=truckData[i][0],
        beta=truckData[i][1],
        muCorrective=truckData[i][6],
        sigmaCorrective=truckData[i][7],
        muPreventive=truckData[i][4],
        sigmaPreventive=truckData[i][5],
        Cc=truckData[i][2],
        Cp=b * truckData[i][2],
        p=a * truckData[i][0] * gamma(1 + 1/truckData[i][1]),
        env=env,
        id=i,
        shovels=shovels,
        dumpsites=dumpsites,
        workshops=workshops,
        muCapacity=truckData[i][8],
        sigmaCapacity=truckData[i][9])
        for i in range(10)]

    begin = datetime.now()
    # Stick the policy to the classes Shovel and Trucks
    Shovel.preventiveMaintenanceRule = "age_based"
    Truck.preventiveMaintenanceRule = "age_based"

    env.run(until=sim_time)

    time_parameters = dict()
    CMcost = 0
    PMcost = 0

    # Results and statistics update
    for i in range(len(trucks)):
        if output:
            print("Truck%d: \tFailures =" %i, env.statistics["Truck%d" %i]["Failure"], "\t Preventive =", env.statistics["Truck%d" %i]["PreventiveInterventions"])
        s = env.statistics["Truck%d"%i]["Statistics"]
        s["Availability"] = 1 - (s["Failed"] + s["PMRepair"] + s["CMRepair"]) / (sim_time)
        del env.statistics["Truck%d"%i]["PreventiveMaintenanceHistory"]
        del env.statistics["Truck%d"%i]["FailureHistory"]
        del env.statistics["Truck%d"%i]["History"]
        CMcost += trucks[i].Cc * env.statistics["Truck%d"%i]["Failure"]
        PMcost += trucks[i].Cp * env.statistics["Truck%d"%i]["PreventiveInterventions"]

    for i in range(len(shovels)):
        if output:
            print("Shovel%d:\tFailures =" %i, env.statistics["Shovel%d" %i]["Failure"], "\t Preventive =", env.statistics["Shovel%d" %i]["PreventiveInterventions"])
        s = env.statistics["Shovel%d"%i]["Statistics"]
        s["Availability"] = 1 - (s["Failed"] + s["PMRepair"] + s["CMRepair"] + s['TravelTime']) / (sim_time)
        s["IdleTime"] = sim_time - s["WorkingTime"] - s["TravelTime"] - s["TimeInQueue"] - s["Failed"] - s["PMRepair"] - s["CMRepair"]
        del env.statistics["Shovel%d"%i]["PreventiveMaintenanceHistory"]
        del env.statistics["Shovel%d"%i]["FailureHistory"]
        del env.statistics["Shovel%d"%i]["History"]
        CMcost += shovels[i].Cc * env.statistics["Shovel%d"%i]["Failure"]
        PMcost += shovels[i].Cp * env.statistics["Shovel%d"%i]["PreventiveInterventions"]

    for i in range(2):
        del env.statistics["DumpSite%d"%i]

    return env.statistics

def mtbf_vs_cost_downtime(values):
    from tqdm import tqdm
    import numpy as np

    random.seed(42)
    sim_time = 1e5
    N = 5
    results = np.ndarray((len(values), N, 2))

    for i in tqdm(range(len(values))):
        for j in range(N):
            results[i,j,0], results[i,j,1] = parametrizedP(values[i], sim_time)

    np.save("costs", results)

def plot_costs(results, values):
    import numpy as np

    CM = np.mean(results[:,:,0], axis=1)/1000
    PM = np.mean(results[:,:,1], axis=1)/1000
    plt.plot(values, CM, values, PM, values, CM+PM)
    plt.fill_between(values, CM, PM, where=CM >= PM, facecolor='#F7C8C1')
    plt.fill_between(values, CM, PM, where=PM >= CM, facecolor='#C1D9F7')
    plt.xlabel("a")
    plt.title("Total Cost of CM/PM")
    plt.ylabel("Cost")
    plt.legend(["CM cost", "PM cost", "Total cost"])
    plt.savefig("figures/costs1.png")
    plt.show()

def variable_costs():

    import numpy as np
    import pprint
    import matplotlib.pyplot as plt

    pp = pprint.PrettyPrinter(indent=0)

    onlyCM = 50
    balanced = 20
    onlyPM = 2

    maintenance = [onlyCM, balanced, onlyPM]

    values = [0.2, 0.5, 1, 2, 3]

    N = 10
    sim_time = 1e5

    results = np.ndarray((len(maintenance), len(values), N, 2))

    for i in range(len(maintenance)):
        for j in range(len(values)):
            for k in range(N):
                stats = parametrizedCost(maintenance[i], values[j], sim_time=sim_time)
                results[i,j,k,0] += sum((stats["Truck%d"%t]["Statistics"]["CMRepair"] for t in range(10)))
                results[i,j,k,0] += sum((stats["Truck%d"%t]["Statistics"]["PMRepair"] for t in range(10)))
                results[i,j,k,1] += sum((stats["Shovel%d"%t]["Statistics"]["CMRepair"] for t in range(3)))
                results[i,j,k,1] += sum((stats["Shovel%d"%t]["Statistics"]["PMRepair"] for t in range(3)))

    results = np.mean(results, axis=2)
    print(results)
    np.save("downtime", results)
    for i in range(len(maintenance)):
        plt.plot(values, results[i,:,0])
    for i in range(len(maintenance)):
        plt.plot(values, results[i,:,1])
    plt.xlabel("b (PM cost as '%' of CM cost)")
    plt.ylabel("cost")
    plt.legend(["Downtime - OnlyCM", "Downtime - Balanced", "Downtime - OnlyPM", "Preventive - OnlyCM", "Preventive - Balanced", "Preventive - OnlyPM"])
    plt.show()

def stockpiles_level(stats):
    import matplotlib.pyplot as plt
    import numpy as np
    d0 = np.array(stats["DumpSite0_stockpileLevel"])
    d1 = np.array(stats["DumpSite1_stockpileLevel"])

    plt.subplot(2,1,1)
    plt.step(d0[:,0], d0[:,1])
    plt.subplot(2,1,2)
    plt.step(d1[:,0], d1[:,1])
    plt.show()

if __name__ == "__main__":
    # EXP 1
    # values = [1, 2, 3, 5, 10, 18, 20, 30, 50, 70]
    # mtbf_vs_cost_downtime(values)
    # import numpy as np
    # results = np.load("costs.npy")
    # plot_costs(results, values)

    # EXP 2
    # with open('param.json', 'r') as f:
    #     param = json.load(f)

    # results = std(param, for_internal_use=True)
    # with open('stats.json', 'w') as f:
    #     json.dump(results, f)
    # stockpiles_level(results)

    # EXP 3
    with open('param.json', 'r') as f:
        param = json.load(f)
    optimize_configuration(4*1e5, 30, param)

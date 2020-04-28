from Mine import *
import simpy, csv, json
from datetime import datetime, timedelta
from statistics import mean
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from multiprocessing import Pool

def std(param, time_parameters=None):
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

    assert len(param['shovelPolicy']) == param['nShovels']
    assert len(param['truckPolicy']) == param['nTrucks']

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
        DumpSite(env,i,coordinates=(dumpsiteData[i][2],dumpsiteData[i][3]),mu=dumpsiteData[i][0],sigma=dumpsiteData[i][1])
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
    print('End')
    processingTime = datetime.now() - begin
    print("Processing time ", processingTime)
    time_parameters = dict()
    # Results and statistics update
    for i in range(len(trucks)):
        print("Truck%d: \tFailures =" %i, env.statistics["Truck%d" %i]["Failure"], "\t Preventive =", env.statistics["Truck%d" %i]["PreventiveInterventions"])
        s = env.statistics["Truck%d"%i]["Statistics"]
        s["Availability"] = 1 - (s["Failed"] + s["PMRepair"] + s["CMRepair"]) / (param["simTime"])
        time_parameters["Truck%d"%i] = dict()
        time_parameters["Truck%d"%i]["LastMaintenance"] = trucks[i].lastMaintenance
        time_parameters["Truck%d"%i]["NextFault"] = trucks[i].nextFault
    for i in range(len(shovels)):
        print("Shovel%d:\tFailures =" %i, env.statistics["Shovel%d" %i]["Failure"], "\t Preventive =", env.statistics["Shovel%d" %i]["PreventiveInterventions"])
        s = env.statistics["Shovel%d"%i]["Statistics"]
        s["Availability"] = 1 - (s["Failed"] + s["PMRepair"] + s["CMRepair"] + s['TravelTime']) / (param["simTime"])
        s["IdleTime"] = param["simTime"] - s["WorkingTime"] - s["TravelTime"] - s["TimeInQueue"] - s["Failed"] - s["PMRepair"] - s["CMRepair"]
        time_parameters["Shovel%d"%i] = dict()
        time_parameters["Shovel%d"%i]["LastMaintenance"] = shovels[i].lastMaintenance
        time_parameters["Shovel%d"%i]["NextFault"] = shovels[i].nextFault

    # return env.statistics
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
            population.append([random.random()*200 for _ in range(nGenes)])
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
                # Select the mutation point
                q = random.randint(0,len(y)-1)
                y[q] += random.normalvariate(mu=0,sigma=0.001)
                x.append(y)
            return x[:n]

    population = generateIndividuals(initialPopSize,items)
    max_generations = 10
    stats = dict()
    stats['best'] = list()
    stats['average'] = list()

    for _ in tqdm(range(max_generations)):

        # Wrap parameters for the simulation
        data = [(20, simTime, {"Shovels": ind[:2], "Trucks": ind[2:]}) for ind in population]

        with Pool() as p:
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
        population = crossovered + mutated + list(population[:10]) + generateIndividuals(30,items)

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
        for i in range(4)]

    colors = ['orange', 'blue', 'green']
    for site in enumerate([shovels, workshops, dumpsites]):
        for i in enumerate(site[1]):
            plt.scatter(i[1].coordinates[0],i[1].coordinates[1],label=i[1].__class__.__name__+str(i[0]),c=colors[site[0]])
            plt.annotate(i[1].__class__.__name__+str(i[0]),(i[1].coordinates[0],i[1].coordinates[1]))
    plt.title("Sites location on the map")
    plt.grid()
    plt.legend()
    plt.savefig("figures/mine_map.png")

if __name__ == "__main__":
    with open('param.json', 'r') as f:
        param = json.load(f)
    with open('results_4000.json', 'r') as f:
        thresholds = json.load(f)
    param['shovelPolicy'] = thresholds['Shovels']
    param['truckPolicy'] =  thresholds['Trucks']
    param['nShovels'] = 2
    param['PMRule'] = 'age_based'
    stats = std(param)

    # best, score = GA(50, 12, 1e5)

    # with open("results.json", "w") as f:
    #     json.dump(best, f)

import simpy,random
from math import exp, sqrt

DEBUG = False

class Truck(object):
    """
    The class replicates the operations and maintenance process of a Truck. The following parameters must be passed to initialize the object.

    :param float alpha: Scale parameter of Weibull distribution
    :param float beta: Shape parameter of Weibull distribution
    :param float muCorrective: Mean time to repair
    :param float sigmaCorrective: Time to repair std. dev.
    :param float muPreventive: Mean maintenance time
    :param float sigmaPreventive: Maintenance time std. dev.
    :param float Cc: Cost of corrective maintenance
    :param float Cp: Cost of preventive maintenance
    :param float p: Probability threshold for preventive maintenance
    :param obj env: A :class:`simpy.Environment` object
    :param int id: Identification number
    :param list shovels: The list of shovels in the system
    :param list dumpsites: The list of dump sites in the system
    :param list workshops: The list of workshops in the system
    """

    def __init__(
        self,
        alpha,
        beta,
        muCorrective,
        sigmaCorrective,
        muPreventive,
        sigmaPreventive,
        Cc,
        Cp,
        p,
        env,
        id,
        shovels,
        dumpsites,
        workshops,
        muCapacity,
        sigmaCapacity
    ):
        self.id = id
        self.env = env
        self.process = env.process(self.run(shovels,dumpsites,workshops))
        self.alpha = alpha
        self.beta = beta
        self.capacity = random.normalvariate(mu=muCapacity,sigma=sigmaCapacity)
        self.muCorrective = muCorrective
        self.sigmaCorrective = sigmaCorrective
        self.muPreventive = muPreventive
        self.sigmaPreventive = sigmaPreventive
        self.Cc = Cc
        self.Cp = Cp
        self.lastMaintenance = 0
        self.p = p
        self.nextFault = 0
        self.coordinates = (0,0)
        self.broken = False
        self.priority = 3
        self.failure = env.process(self.fault())
        self.env.statistics["Truck%d" %self.id] = {
            "Failure": 0,
            "FailureHistory": list(),
            "PreventiveInterventions": 0,
            "PreventiveMaintenanceHistory": list()}

    def run(self,shovels,dumpsites,workshops):
        """
        The method is a generator function which returns the event linked with the operations of a truck.
        """
        while True:
            try:
                # ASSIGN A SHOVEL
                shovel = self.assignment(shovels)
                # TRAVEL
                if DEBUG:
                    print("Truck%d \t is traveling towards Shovel%d \tat %.2f." %(self.id,shovel.id,self.env.now))
                yield self.env.process(self.travel(shovel.coordinates))
                if DEBUG:
                    print("Truck%d \t arrived at Shovel%d \t\tat %.2f." %(self.id,shovel.id,self.env.now))

                # LOAD
                loadingTime = shovel.servingTime()
                priority, flag = self.priority, 0
                while loadingTime:
                    with shovel.machine.request(priority=priority) as req:
                        yield req
                        if DEBUG:
                            print("Truck%d \t start loading at Shovel%d\tat %.2f." %(self.id,shovel.id,self.env.now))
                        try:
                            start = self.env.now
                            yield self.env.process(shovel.load())
                            if DEBUG:
                                print("Truck%d \t loaded by Shovel%d\t\tat %.2f." %(self.id,shovel.id,self.env.now))
                            loadingTime = 0
                        except simpy.Interrupt as interruption:
                            if type(interruption.cause) is simpy.resources.resource.Preempted:
                                if DEBUG:
                                    print("Truck%d\t interrupted loading at Shovel%d\tat %.2f." %(self.id,shovel.id,self.env.now))
                                loadingTime -= self.env.now - start
                                priority = 2
                            elif interruption.cause[0] == 2:
                                flag = 1
                                loadingTime = 0
                if flag == 1:
                    raise simpy.Interrupt(cause="Truck%d" % self.id)
                # CHECK SHOVEL FOR PREVENTIVE MAINTENANCE
                workshop = shovel.assignment(workshops)
                expTaskTime = shovel.distance(workshop.coordinates) + shovel.waitingTime()
                if shovel.doPreventiveMaintenance(expTaskTime):
                    # Interrupt failure process
                    shovel.failure.interrupt()
                    self.env.process(shovel.preventiveMaintenance(workshop))

                # ASSIGN A DUMPSITE
                dumpsite = self.assignment(dumpsites)

                # TRAVEL
                if DEBUG:
                    print("Truck%d \t is traveling towards DumpSite%d at %.2f." %(self.id,dumpsite.id,self.env.now))
                yield self.env.process(self.travel(dumpsite.coordinates))
                if DEBUG:
                    print("Truck%d \t arrived at DumpSite%d \t\tat %.2f." %(self.id,dumpsite.id,self.env.now))

                # UNLOAD
                with dumpsite.machine.request() as req:
                    yield req
                    if DEBUG:
                        print("Truck%d \t under unloading at DumpSite%d\tat %.2f." %(self.id,dumpsite.id,self.env.now))
                    yield self.env.process(dumpsite.unload(self,amount=self.capacity))
                    if DEBUG:
                        print("Truck%d \t unloaded at DumpSite%d\t\tat %.2f." %(self.id,dumpsite.id,self.env.now))

                # PREVENTIVE MAINTENANCE
                shovel = self.assignment(shovels)
                expTaskTime = self.estimateExpTaskTime(shovel)
                if self.doPreventiveMaintenance(expTaskTime=expTaskTime):
                    self.env.statistics["Truck%d" % self.id]["PreventiveInterventions"] += 1
                    self.env.statistics["Truck%d" % self.id]["PreventiveMaintenanceHistory"].append([self.env.now, self.id, self.Cp])
                    # ASSIGN A WORKSHOP
                    workshop = self.assignment(workshops)
                    # DELETE OLD FAULT EVENT
                    self.failure.interrupt(cause="PM%d" % self.id)
                    # TRAVEL
                    if DEBUG:
                        print("Truck%d \t is traveling to Workshop%d \tat %.2f." %(self.id,workshop.id,self.env.now))
                    yield self.env.process(self.travel(workshop.coordinates))
                    if DEBUG:
                            print("Truck%d \t arrived at WorkShop%d \t\tat %.2f." %(self.id,workshop.id,self.env.now))
                    # REPAIR
                    with workshop.machine.request(priority=2) as req:
                        yield req
                        if DEBUG:
                            print("Truck%d \t under repair at Workshop%d\tat %.2f." %(self.id,workshop.id,self.env.now))
                        yield self.env.process(workshop.preventiveMaintenance(self))
                        if DEBUG:
                            print("Truck%d \t repaired by Workshop%d \t\tat %.2f." %(self.id,workshop.id,self.env.now))
                    # REGENERATE THE FAULT EVENT
                    self.failure = self.env.process(self.fault())

            except simpy.Interrupt:
                if DEBUG:
                    print("Truck%d \t failed \t\t\tat %.2f." % (self.id, self.env.now))
                self.env.statistics["Truck%d" %self.id]["Failure"] += 1
                self.env.statistics["Truck%d" %self.id]["FailureHistory"].append([self.env.now, self.id, self.Cc])
                # ASSIGN A WORKSHOP
                workshop = self.assignment(workshops)
                # TRAVEL
                if DEBUG:
                    print("Truck%d \t is traveling to Workshop%d \tat %.2f." %(self.id,workshop.id,self.env.now))
                yield self.env.process(self.travel(workshop.coordinates))
                if DEBUG:
                        print("Truck%d \t arrived at WorkShop%d \t\tat %.2f." %(self.id,workshop.id,self.env.now))
                # REPAIR
                with workshop.machine.request(priority=2) as req:
                    yield req
                    if DEBUG:
                        print("Truck%d \t under repair at Workshop%d\tat %.2f." %(self.id,workshop.id,self.env.now))
                    yield self.env.process(workshop.correctiveRepair(self))
                    if DEBUG:
                        print("Truck%d \t repaired by Workshop%d \t\tat %.2f." %(self.id,workshop.id,self.env.now))
                # REGENERATE THE FAULT EVENT
                self.failure = self.env.process(self.fault())

    def fault(self):
        """ The method generates a fault every now and then."""
        try:
            ttf = 60 * random.weibullvariate(alpha=self.alpha,beta=self.beta)
            self.nextFault = self.env.now + ttf
            # print(self.nextFault,ttf)
            yield self.env.timeout(ttf)
            self.process.interrupt(cause=(2,"Truck%d" % self.id))
        except simpy.Interrupt:
            pass

    def travel(self,destination):
        travelTime = self.distance(destination)
        yield self.env.timeout(travelTime)
        self.coordinates = destination

    def distance(self,destination):
        return 0.5 * random.lognormvariate(
            mu=sqrt((self.coordinates[0] - destination[0])**2 + (self.coordinates[1] - destination[1])**2),
            sigma=0.1)

    def assignment(self,items):
        """
        The function returns the item :math:`i` (shovel, dumpsite, workshop) to which the Truck is assigned among a set of items :math:`I`, which is provided as a list.
        The value :math:`i` is returned by the following equation.

        .. math::

            \\text{argmin}_{i \\in I} = \\mathbb{E} [ \\text{travel distance to $i$} ] + \\mathbb{E} [ \\text{waiting time at $i$} ]

        """
        x, candidate = 1e6, None
        for item in items:
            time = self.distance(item.coordinates) + item.waitingTime()
            if time < x:
                x, candidate = time, item
        if candidate is None:
            raise ValueError
        else:
            return candidate

    def estimateExpTaskTime(self,item):
        return self.distance(item.coordinates) + item.waitingTime()

    def doPreventiveMaintenance(self,expTaskTime):
        """
        The decision rule to trigger a preventive maintenance intervention is enclosed in the method. The rule used is the following.

        .. math::

            P(t < T < \\Delta t | T > t) > \\text{threshold}

        """
        def weibull(x):
            return 1 - exp(-x/self.alpha)**self.beta

        pFailure = (weibull(self.env.now/60 - self.lastMaintenance/60 + expTaskTime/60) - weibull(self.env.now/60 - self.lastMaintenance/60)) / (1 - weibull(self.env.now/60 - self.lastMaintenance/60))

        if pFailure > self.p: return True
        else: return False

    def timeToRepairCorrective(self):
        return random.lognormvariate(mu=self.muCorrective,sigma=self.sigmaCorrective)

    def timeToRepairPreventive(self):
        return random.lognormvariate(mu=self.muPreventive,sigma=self.sigmaPreventive)


class Server(object):
    """
    This is the base class with the common attributes and methods of all server entities within the simulation.
    """

    def __init__(self,env,id,coordinates,mu=None,sigma=None):
        self.env = env
        self.id = id
        self.coordinates = coordinates
        self.mu = mu
        self.sigma = sigma
        self.broken = False

    def servingTime(self):
        return random.lognormvariate(mu=self.mu,sigma=self.sigma)


class Shovel(Server):
    """
    The class inherits the basic attributes from :class:`Server`. It includes the method to replicate the operations in case of preventive maintenance of a shovel.

    :param obj env: A :class:`simpy.Environment` object
    :param int id: Identification number
    :param tuple coordinates: the (*x*, *y*) coordinates of the site on the map
    :param float mu: mean of the (lognormal) serving time distribution
    :param float sigma: st. dev. of the (lognormal) serving time distribution
    :param float alpha: Scale parameter of Weibull distribution
    :param float beta: Shape parameter of Weibull distribution
    :param float muCorrective: Mean time to repair
    :param float sigmaCorrective: Time to repair std. dev.
    :param float muPreventive: Mean maintenance time
    :param float sigmaPreventive: Maintenance time std. dev.
    :param float Cc: Cost of corrective maintenance
    :param float Cp: Cost of preventive maintenance
    :param float p: Probability threshold for preventive maintenance
    :param list workshops: The list of workshops in the system
    """

    def __init__(
        self,
        env,
        id,
        coordinates,
        mu,
        sigma,
        alpha,
        beta,
        muPreventive,
        sigmaPreventive,
        muCorrective,
        sigmaCorrective,
        Cc,
        Cp,
        p,
        workshops
    ):
        super().__init__(env, id, coordinates,mu,sigma)
        self.alpha = alpha
        self.beta = beta
        self.muPreventive = muPreventive
        self.sigmaPreventive = sigmaPreventive
        self.muCorrective = muCorrective
        self.sigmaCorrective = sigmaCorrective
        self.machine = simpy.PreemptiveResource(env,capacity=1)
        self.nextFault = None
        self.Cc = Cc
        self.Cp = Cp
        self.lastMaintenance = 0
        self.p = p
        self.workshops = workshops
        self.failure = env.process(self.fault())

        env.statistics["Shovel%d" %self.id] = {
            "Failure": 0,
            "FailureHistory": list(),
            "PreventiveInterventions": 0,
            "PreventiveMaintenanceHistory": list()}

    def timeToFailure(self):
        return 60 * random.weibullvariate(alpha=self.alpha,beta=self.beta)

    def load(self):
        yield self.env.timeout(self.servingTime())

    def waitingTime(self):
        return (len(self.machine.queue) + self.machine.count) * self.mu

    def timeToRepairCorrective(self):
        return random.lognormvariate(mu=self.muCorrective,sigma=self.sigmaCorrective)

    def timeToRepairPreventive(self):
        return random.lognormvariate(mu=self.muPreventive,sigma=self.sigmaPreventive)

    def assignment(self,items):
        x, candidate = 1e6, None
        for item in items:
            if not item.broken:
                time = self.distance(item.coordinates) + item.waitingTime()
                if time < x:
                    x, candidate = time, item
        if candidate is None:
            raise ValueError
        else:
            return candidate

    def distance(self,destination):
        return 0.5 * random.lognormvariate(
            mu=sqrt((self.coordinates[0] - destination[0])**2 + (self.coordinates[1] - destination[1])**2),
            sigma=0.1)

    def travel(self,destination):
        travelTime = self.distance(destination)
        yield self.env.timeout(travelTime)

    def fault(self):
        while True:
            try:
                ttf = self.timeToFailure()
                self.nextFault = self.env.now + ttf
                yield self.env.timeout(ttf)
                if not self.broken:
                    with self.machine.request(priority=1) as req:
                        self.broken = True
                        if DEBUG:
                            print("Shovel%d\t failed\t\t\t\tat %.2f." % (self.id, self.env.now))
                        self.env.statistics["Shovel%d" %self.id]["Failure"] += 1
                        self.env.statistics["Shovel%d" %self.id]["FailureHistory"].append([self.env.now, self.id, self.Cc])
                        yield req
                        # ASSIGN WORKSHOP
                        workshop = self.assignment(self.workshops)
                        # TRAVEL
                        if DEBUG:
                            print("Shovel%d\t is traveling towards Workshop%d\tat %.2f." %(self.id,workshop.id,self.env.now))
                        yield self.env.process(self.travel(workshop.coordinates))
                        if DEBUG:
                            print("Shovel%d\t arrived at Workshop%d\t\tat %.2f." %(self.id,workshop.id,self.env.now))
                        # REPAIR
                        with workshop.machine.request(priority=1) as req:
                            yield req
                            if DEBUG:
                                print("Shovel%d\t is under repair at Workshop%d\tat %.2f." %(self.id,workshop.id,self.env.now))
                            self.lastMaintenance = self.env.now
                            yield self.env.timeout(self.timeToRepairCorrective())
                            if DEBUG:
                                print("Shovel%d\t repaired at Workshop%d\t\tat %.2f." %(self.id,workshop.id,self.env.now))
                        # TRAVEL
                        if DEBUG:
                            print("Shovel%d\t is traveling towards its site\tat %.2f." %(self.id,self.env.now))
                        yield self.env.process(self.travel(workshop.coordinates))
                        if DEBUG:
                            print("Shovel%d\t arrived at its site\t\tat %.2f." %(self.id,self.env.now))
                        self.broken = False
            except simpy.Interrupt:
                break

    def doPreventiveMaintenance(self,expTaskTime):
        """
        The decision rule to trigger a preventive maintenance intervention is enclosed in the method. The rule used is the following.

        .. math::

            P(t < T < \\Delta t | T > t) > \\text{threshold}

        """
        def weibull(x):
            return 1 - exp(-x/self.alpha)**self.beta

        pFailure = (weibull(self.env.now/60 - self.lastMaintenance/60 + expTaskTime/60) - weibull(self.env.now/60 - self.lastMaintenance/60)) / (1 - weibull(self.env.now/60 - self.lastMaintenance/60))

        if pFailure > self.p: return True
        else: return False

    def preventiveMaintenance(self,workshop):
        """
        The method is a generator function which replicates the maintenance operations for a shovel in case of preventive maintenance.
        """
        with self.machine.request(priority=1) as req:
            if DEBUG:
                print("Shovel%d\t failed\t\t\t\tat %.2f." % (self.id, self.env.now))
            self.env.statistics["Shovel%d" %self.id]["PreventiveInterventions"] += 1
            self.env.statistics["Shovel%d" %self.id]["PreventiveMaintenanceHistory"].append([self.env.now, self.id, self.Cp])
            yield req
            # TRAVEL
            if DEBUG:
                print("Shovel%d\t is traveling towards Workshop%d\tat %.2f." %(self.id,workshop.id,self.env.now))
            yield self.env.process(self.travel(workshop.coordinates))
            if DEBUG:
                print("Shovel%d\t arrived at Workshop%d\t\tat %.2f." %(self.id,workshop.id,self.env.now))
            # REPAIR
            with workshop.machine.request(priority=1) as req:
                yield req
                if DEBUG:
                    print("Shovel%d\t is under repair at Workshop%d\tat %.2f." %(self.id,workshop.id,self.env.now))
                self.lastMaintenance = self.env.now
                yield self.env.timeout(self.timeToRepairCorrective())
                if DEBUG:
                    print("Shovel%d\t repaired at Workshop%d\t\tat %.2f." %(self.id,workshop.id,self.env.now))
            # TRAVEL
            if DEBUG:
                print("Shovel%d\t is traveling towards its site\tat %.2f." %(self.id,self.env.now))
            yield self.env.process(self.travel(workshop.coordinates))
            if DEBUG:
                print("Shovel%d\t arrived at its site\t\tat %.2f." %(self.id,self.env.now))
        self.failure = self.env.process(self.fault())


class DumpSite(Server):
    """
    The class inherits the basic attributes from :class:`Server` and replicates the behavior of a dumpsite.
    """

    def __init__(self, env, id, coordinates,mu,sigma):
        super().__init__(env, id, coordinates,mu,sigma)
        self.machine = simpy.Resource(env,capacity=1)
        env.statistics["DumpSite%d" %self.id] = list()

    def unload(self,truck,amount):
        self.env.statistics["DumpSite%d" %self.id].append([self.env.now, truck.capacity])
        yield self.env.timeout(self.servingTime())

    def waitingTime(self):
        return (len(self.machine.queue) + self.machine.count) * self.mu


class WorkShop(Server):
    """
    The class inherits the basic attributes from :class:`Server` and replicates the behavior of a workshop.
    The serving time depends on the repair time of the item under repair and on the type of maintenance, i.e., corrective or preventive.
    """

    def __init__(self, env, id, coordinates):
        super().__init__(env, id, coordinates)
        self.machine = simpy.PriorityResource(env,capacity=1)

    def correctiveRepair(self,equipment):
        equipment.lastMaintenance = self.env.now
        yield self.env.timeout(equipment.timeToRepairCorrective())

    def preventiveMaintenance(self,equipment):
        equipment.lastMaintenance = self.env.now
        yield self.env.timeout(equipment.timeToRepairPreventive())

    def waitingTime(self):
        return len(self.machine.queue) + self.machine.count

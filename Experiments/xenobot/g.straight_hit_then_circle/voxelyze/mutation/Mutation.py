# This is the base class for any mutation method
#

class Mutation:
    def __init__(self):
        """ Init Mutation with data """
        pass

    def mutate(self, population):
        """ Do mutation """
        from .. import names
        mutants = {}
        anykey = None
        for key in population.keys():
            mutants[key] = []
            anykey = key
        for i in range(len(population[anykey])):
            # copy all properties
            for key in population.keys():
                mutants[key].append(population[key][i])
            # change the first name
            if "firstname" in mutants:
                mutants["firstname"][-1] = names.get_first_name()
            
        return mutants

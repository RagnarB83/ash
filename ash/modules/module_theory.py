
# Basic Theory class

class Theory:
    def __init__(self):
        self.theorytype = None
        self.theorynamelabel = None
        self.label = None
        self.analytic_hessian = False
        self.numcores = None
        self.filename = None
        self.printlevel = None
        self.properties = {}
    def set_numcores(self,numcores):
        self.numcores=numcores
    def cleanup(self):
        print("Cleanup method called but not yet implemented for this theory")
    def run(self):
        print("Run was called but not yet implemented for this theory")


# Basic Theory class

class Theory:
    def __init__(self):
        self.theorytype = None
        self.theorynamelabel = None
        self.label = None
        self.analytic_hessian = False
        self.numcores = 1
        self.filename = None
        self.printlevel = None
        self.properties = {}
    def set_numcores(self,numcores):
        self.numcores=numcores
    def cleanup(self):
        print("Cleanup method called but not yet implemented for this theory")
    def run(self):
        print("Run was called but not yet implemented for this theory")

class QMTheory(Theory):
    def __init__(self):
        super().__init__()
        self.theorytype = "QM"
    def set_numcores(self,numcores):
        self.numcores=numcores
    def cleanup(self):
        print("Cleanup method called but not yet implemented for this theory")
    def run(self):
        print("Run was called but not yet implemented for this theory")


# Numerical gradient class

class NumGrad:
    def __init__(self):
        pass
    def set_numcores(self,numcores):
        self.numcores=numcores
    def cleanup(self):
        print("Cleanup method called but not yet implemented for Numgrad")
    def run(self):
        print("Run was called but not yet implemented for this theory")

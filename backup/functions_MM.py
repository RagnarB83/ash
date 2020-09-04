class AtomMMobject:
    def __init__(self, atomcharge=None, LJparameters=[]):
        sf="dsf"
        self.atomcharge = atomcharge
        self.LJparameters = LJparameters
    def add_charge(self, atomcharge=None):
        self.atomcharge = atomcharge
    def add_LJparameters(self, LJparameters=None):
        self.LJparameters=LJparameters


def coulombcharge(charges):
    sumenergy=0
    for i in charges:
        for j in charges:
            if i != j:
                pairenergy=i*j
                sumenergy+=pairenergy
    return sumenergy
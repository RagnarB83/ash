import geometric
class Test:
    def __init__(self):
        self.sdf='sdf'
        self.prefix='aaa'


args=Test()

geometric.optimize.run_optimizer(**vars(args))

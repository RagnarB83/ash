#Currently disabled Settings class
# Settings class
#class Settings:
#    def __init__(self, scale=1.0, tol=0.0):
#        self.scale = scale
#        self.tol = tol
#    def change_scale(self, new_scale): # note that the first argument is self
#        self.scale = new_scale # access the class attribute with the self keyword
#    def change_tol(self, new_tol): # note that the first argument is self
#        self.tol = new_tol # access the class attribute with the self keyword

#Defining global variables here: called as settings_molcrys.scale etc.
def init():
    global scale
    global tol
    scale=1.0
    tol=0.0
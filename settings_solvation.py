class Sphere:
    def __init__(self, name, path, file, numatoms):
        self.Name = name
        self.path = path
        self.file = file
        self.pathtofile = path + '/' + file
        self.numatoms = numatoms

# Defining global variables here: called as settings_molcrys.scale etc.
def init(progdir,ORCApath,XTBpath,ncores):
    global bulkfile
    global bulksphere
    global scale
    global tol
    global debug
    #Making orcadir and xtbdir (originally read from inputfile) global everywhere
    global orcadir
    global xtbdir
    global NumCores
    global print_time
    orcadir=ORCApath
    xtbdir=XTBpath
    NumCores=ncores
    # Scaling and tolerance settings for connectivity
    scale=1.0
    tol=0.0
    # Defining Bulksphere as a convenient object.
    # Todo: Read coordinates and charges as attributes to object instead???? Change Class from Sphere to something more appropriate
    bulksphere = Sphere('Sphere100-32', progdir, 'tip3pbox-sphere-100hollow32-noheader.pc', 405312)

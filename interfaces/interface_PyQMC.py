
# Possible future PyQMC interface. Interfaces with pyscf but we want to avoid having it there
#PyQMC=False, PyQMC_nconfig=1, PyQMC_method='DMC',

def load_pyqmc():
    try:
    import pyqmc.api as pyq
    except:
    print(BC.FAIL, "Problem importing pyqmc.api. Make sure pyqmc has been installed: pip install pyqmc", BC.END)
    ashexit(code=9)
configs = pyqmc.initial_guess(mol,PyQMC_nconfig)
wf, to_opt = pyqmc.generate_wf(mol,mf)
pgrad_acc = pyqmc.gradient_generator(mol,wf, to_opt)
wf, optimization_data = pyqmc.line_minimization(wf, configs, pgrad_acc)
#DMC, untested
if PyQMC_method == 'DMC':
    configs, dmc_data = pyqmc.rundmc(wf, configs)
    #VMC. untested
elif PyQMC_method == 'VMC':
    df, configs = vmc(wf,configs)

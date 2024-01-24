import parmed as pmd
from parmed.charmm import CharmmParameterSet
from parmed import openmm

#Simple parmed script to create OpenMM XML file from CHARMM top and par files for a few ligands
#Relevant parameters taken from : toppar_all36_moreions.str and par_all36_cgenff.prm
Phosphate topologies taken from : 

#Read CHARMM parameters from CHARMM files
params = CharmmParameterSet('phoshates.str','par.prm')
#Convert CHARMM parameters to OpenMM format
omm_params = openmm.OpenMMParameterSet.from_parameterset(params)
#Write OpenMM parameters to XML file
omm_params.write("phosphates.xml")


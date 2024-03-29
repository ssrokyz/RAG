#!/usr/bin/env python
"""
 <NOTE> 

* POTCAR file including pseudopotentials of Ge and Te is required.
  (POTCAR file is not included due to licenses. Please get one or use calculators other than VASP.)

* To run this script, you should install RAG in advance.
  To get RAG, visit the GitHub repository below.
  https://github.com/ssrokyz/RAG

* Script is using the VASP to calculate the energies and forces.
  You can easily revise the script to use different calculators.

"""

import numpy as np
from ase.io import read, write
from RAG import random_atoms_gen as rag
from subprocess import call
import datetime

# @ Hyperparams
label = 'GeTe-RAG'
backbone = read('backbone-GeTe-24.vasp')
iter = 500
# strain_max = 1.1
# strain_min = 0.95
strain_mean  = 1.02
strain_sigma = 0.03
vacuum_max = 0.

#
NSLOTS = 4
EXEC = 'vasp_std'
OUT = 'vasp.out'
calc_command = 'mpiexec.hydra -np {} {} > {}'.format(NSLOTS, EXEC, OUT)
# You should adjust the RAG parameters below.

# @ Preprocess
from ase.io.trajectory import Trajectory as Traj
traj = Traj(label+'.traj', 'w')

# @ Main
for i in range(iter):
    # Random ratio
    # strain_ratio = np.random.rand(3) * (strain_max - strain_min) + strain_min
    strain_ratio = np.random.normal(strain_mean, strain_sigma, 3)
    vacuum = [0.,0.,np.random.rand() * vacuum_max]

    # Random system generation
    # Descriptions for all parameters are included in the RAG.py file.
    atoms = rag(
        backbone,
        num_spec_dict = {'Ge':12, 'Te':12},
        # fix_ind_dict  = {'Te':[1, 3, 4, 6, 9, 11, 13, 15, 16, 18, 21, 23]},
        # pin_the_fixed = False,
        # cutoff_radi   = 2.864 * 0.75,
        cutoff_frac   = 0.9,
        # random_radi   = None,
        random_frac   = 0.50,
        # strain        = None,
        strain_ratio  = strain_ratio,
        vacuum        = vacuum,
        # vacuum_ratio  = None,
        # max_trial_sec = 5,
        # log           = True,
        )
    write('POSCAR', atoms)
    call(calc_command, shell=True)
    tmp = read('vasprun.xml', ':')
    for j in range(len(tmp)):
        traj.write(tmp[j])

    # @ Log
    now = datetime.datetime.now()
    time = now.strftime('%Y-%m-%d %H:%M:%S')
    log = open("log_"+label+".txt", "a")
    log.write("\n"+str(i)+"    "+time+"    ")
    log.close()

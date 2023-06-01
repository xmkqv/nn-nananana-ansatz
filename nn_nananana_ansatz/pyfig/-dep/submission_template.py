import numpy as np

def dict_to_cmd(d: dict, sep=' ', exclude_false=False, exclude_none=True):

	items = d.items()
	items = [(k, (v.tolist() if isinstance(v, np.ndarray) else v)) for (k,v) in items]
	items = [(str(k).replace(" ", ""), str(v).replace(" ", "")) for (k,v) in items]

	if exclude_false:
		items = [(k, v) for (k,v) in items if not (d[k] is False)]
	if exclude_none:
		items = [(k, v) for (k,v) in items if not (d[k] is None)]
  
	items = [(k, v) for (k,v) in items if v]

	return ' '.join([(f'--{k}' if ((v=='True') or (v is True)) else f'--{k + sep + v}') for k,v in items])

from simple_slurm import Slurm

input_params = dict(
    variable_1 = [0.1, 0.4, 0.1, -1.7], # all possible values for variable_1
    variable_2 = [200, 300], # all possible values for variable_2
    test = ['test', 'tomas', 'sky']
)

# create data folder 
# how to minimise the number of files 
# write a argparser for the input parameters ### WRITE IN YOUR FILE ### look at tutorial online! Think about types 
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    for key, value in input_params.items():
        parser.add_argument(f'--{key}', type=type(value[0]), default=value[0])
    return parser.parse_args()

args = parse_args()
# returns a dictionary with the input parameters


# https://www.hpc.dtu.dk/?page_id=2534
filename = ''
environment = '' 


### Create a list of all possible combinations of the input parameters
from itertools import product

list_of_all_variables = list(input_params.values())
experiments = list(product(*list_of_all_variables))
for i, combination in enumerate(experiments):
    print(combination)



for exp in experiments:
    cmd = dict_to_cmd(dict(zip(input_params.keys(), exp)))

    body = f"""


# python {filename} {cmd}
"""
    print(body)

    # BASH FILE 
    # test.sh
    # bsub < test.sh ### <<< this command runs the bash file
    # Learn about command line arguments 
    filetext = f""" 
    !/bin/sh
    module purge
    source ~/.bashrc
    conda activate {environment}
    module load GCC
    module load OpenMPI
    embedded options to bsub - start with #BSUB
    ### -- set the job Name AND the job array --
    #BSUB -J My_array[1-25]

    ### –- specify queue -- 
    #BSUB -q hpc 

    ### -- ask for number of cores (default: 1) --
    #BSUB -n 1

    ### -- set walltime limit: hh:mm --
    #BSUB -W 00:10:00 

    ### -- specify that we need 4GB of memory per core/slot -- 
    #BSUB -R "rusage[mem=4GB]"

    ### -- set the email address --
    # please uncomment the following line and put in your e-mail address,
    # if you want to receive e-mail notifications on a non-default address
    ##BSUB -u your_email_address
    ### -- send notification at start --
    #BSUB -B

    ### -- send notification at completion--
    #BSUB -N
    ### -- Specify the output and error file. %J is the job-id %I is the job-array index --
    ### -- -o and -e mean append, -oo and -eo mean overwrite -- 
    #BSUB -o Output_%J_%I.out
    #BSUB -e Output_%J_%I.err 

    python run_template.py {cmd} 
    """

    with open('run.sh', 'w') as f:
        f.write(filetext)

    import os
    os.system('chmod +x run.sh')
    os.system('bsub < run.sh')




"""

SLURM 

from pydantic import BaseModel

class slurm_c(BaseModel):
    export			= 'ALL'		# 1 task 1 gpu 8 cpus per task 
    cpu_per_task	= 1 ### TODO: Find out minimum number of cpus per task
    partition       = '' ### TODO: Where do the submissions go?! on the hpc partition the type of cpu etc
    time            = '0-00:10:00'  # D-HH:MM:SS
    nodes           = 1 			# (MIN-MAX) 
    ntasks          = 1
    job_name        = 'test'
    # output          = 'o-%j.out'
    # error           = 'e-%j.err'


slurm = Slurm(**slurm_c.dict())

for exp in experiments:
    cmd = dict_to_cmd(dict(zip(input_params.keys(), exp)))

# python {filename} {cmd}
#
    print(body)
    break

    slurm.sbatch(body, verbose= True)
"""
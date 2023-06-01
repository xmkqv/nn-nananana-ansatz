# Anything that counts as configuration for this project should be in this file.

from pydantic import BaseModel, Field

from nn_nananana_ansatz.appfig import Walkers, Ansatzcfg, Pyfig
from nn_nananana_ansatz.systems import System

class Walkers(Walkers):
	...

class Ansatzcfg(Ansatzcfg):
	...

class Pyfig(Pyfig):
	...
	



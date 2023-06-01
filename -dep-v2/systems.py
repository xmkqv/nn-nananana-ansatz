from pydantic import Field
from typing import Optional

import numpy as np

from .typfig import ModdedModel, ndarray

class System(ModdedModel):

	a: ndarray[float]  	= Field(..., description= 'atomic positions, shape (n_atoms, 3), units: Bohr')
	a_z: ndarray[int] 	= Field(..., description= 'atomic numbers, shape (n_atoms), units: Bohr')

	charge:     int		= Field(0, description= 'total charge of the system')
	spin:       int		= Field(0, description= 'total spin of the system')

	@property
	def name(self):
		return element_name(self)

	@property
	def n_e(self): # number of electrons
		return int(sum(self.a_z))

	@property
	def n_u(self): # number of up electrons
		return (self.spin + self.n_e) // 2

	@property
	def n_d(self): # number of down electrons
		return self.n_e - self.n_u

	@property
	def system_id(self):
		return [[int(a_z_i), a_i.tolist()] for a_z_i, a_i in zip(self.a_z, self.a)]
		# return '_'.join([f'{a_z_i}_{a_i[0]}_{a_i[1]}_{a_i[2]}' for a_z_i, a_i in zip(self.a_z, self.a)])
	
	
_example_system = System(
	name = 'Be2',
	a = np.array(
		[
			[0.0, 0.0, 0.0], 
			[0.0, 0.0, 0.0],
		]),
	a_z = np.array([4, 4]),
	spin = 0,
	charge = 0,
)

elements = {
	1: 'H',          # - Hydrogen
	2: 'He',         # - Helium
	3: 'Li',         # - Lithium
	4: 'Be',         # - Beryllium
	5: 'B',          # - Boron
	6: 'C',          # - Carbon
	7: 'N',          # - Nitrogen
	8: 'O',          # - Oxygen
	9: 'F',          # - Fluorine
	10: 'Ne',            # - Neon
	11: 'Na',            # - Sodium
	12: 'Mg',            # - Magnesium
	13: 'Al',            # - Aluminum
	14: 'Si',            # - Silicon
	15: 'P',         # - Phosphorus
	16: 'S',         # - Sulfur
	17: 'Cl',            # - Chlorine
	18: 'Ar',            # - Argon
	19: 'K',         # - Potassium
	20: 'Ca',            # - Calcium
}


def element_name(system: System):
	""" naming convention for system
	
	{atom_0}{n_atom_0}{atom_1}{n_atom_1}..._{charge_sign}{spin}{spin_symbol}_{charge_sign}{charge} 
	
	- nuclei arranged by atomic number """

	sorted_a_z = np.sort(system.a_z)
	unique_a_z, count = np.unique(sorted_a_z, return_counts= True)
	atoms = [elements.get(int(a_z), 'N/A') for a_z in unique_a_z]
	name = [f'{atom}{n_atom}' for atom, n_atom in zip(atoms, count)]

	charge = '_'
	if system.charge > 0:
		charge += "+"
	elif system.charge < 0:
		charge += "-"
	charge += str(system.charge)
	name += charge

	spin = '_'
	if system.spin > 0: # positive spin
		spin += f"+{str(system.spin)}↑"
	elif system.spin < 0: # negative spin
		spin += f"+{str(system.spin)}↓"
	else:
		spin += "0"
	name += spin

	return ''.join(name)

if __name__ == '__main__':
	  
	print(element_name(_example_system))

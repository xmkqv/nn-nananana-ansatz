import numpy as np
from pydantic import BaseModel, validator, Extra, fields
from typing import Any

class ndarray(np.ndarray):
	# https://docs.pydantic.dev/usage/schema/#modifying-schema-in-custom-fields

	@classmethod
	def __modify_schema__(cls, field_schema: dict):
		field_schema.update(
			type= 'string',
	)

def str_to_numpy(s: str, dtype: type):
	s = s.replace(' ', '').replace('\n', '').replace('\t', '').replace('][', '],[')
	if s.startswith('['): # Check if s contains a nested array
		s = s[1:-1] # Remove the outer brackets
		nested_arrays = s.split('],[') # Split s into nested arrays
		if len(nested_arrays) == 1: # Check if s is a 1D array
			return str_to_numpy(nested_arrays[0], dtype= dtype) # Convert the 1D array
		nested_arrays[0] = nested_arrays[0][1:] # Remove the opening bracket from the first nested array
		nested_arrays[-1] = nested_arrays[-1][:-1] # Remove the closing bracket from the last nested array
		values = [str_to_numpy(nested_array, dtype= dtype) for nested_array in nested_arrays] # Recursively parse each nested array
		return np.array(values, dtype= dtype) # Combine the values into a single numpy array
	else:
		values = [float(v) for v in s.split(',')]
		return np.asarray(values, dtype= dtype)

class ModdedModel(BaseModel):
	
	class Config:
		# keep_untouched = () # left alone by pydantic
		# schema_extra: dict = dict()
		arbitrary_types_allowed = True
		validate_assignment = True
		extra = Extra.ignore
		underscore_attrs_are_private = True
		json_encoders = { # int32/float64 version doesn't work... yet #> pyfig:typfig
			ndarray: lambda v: str(v.tolist()).replace(' ', '').replace('\n', ''),
			# 'ndarray': lambda v: str(v.tolist()).replace(' ', ''),
			# ndarray[float]: lambda v: str(v.tolist()).replace(' ', ''),
			ndarray[int]: lambda v: str(v.tolist()).replace(' ', '').replace('\n', ''),
		}
		json_decoders = {
			ndarray: lambda v: str_to_numpy(v, dtype= float),
			# ndarray[float]: lambda v: str_to_numpy(v, dtype= float),
			ndarray[int]: lambda v: str_to_numpy(v, dtype= int),
		}

	@validator('*', pre= True, always= True)
	def validate_arrays(cls, v, field: fields.ModelField):
		# print(field.type_, str(field.outer_type_))
		if isinstance(field.type_, object) and issubclass(field.type_, ndarray):
			if '[' in str(field.outer_type_):
				if 'float' in str(field.outer_type_):
					dtype = 'float'
				elif 'int' in str(field.outer_type_):
					dtype = 'int'
				else:
					dtype = 'float'
					print('dtype not found', field.outer_type_, field.type_)
					raise NotImplementedError

			else:
				print('dtype not found', field.outer_type_, field.type_)
				dtype = 'float64'

			dtype = dict(
				float64= np.float64, 
				float32= np.float32, 
				int64= np.int64, 
				int32= np.int32, 
				float= float,
				int= int
			).get(dtype, np.float64)

			if isinstance(v, str):
				v = str_to_numpy(v, dtype= dtype)
			if isinstance(v, list):
				v = np.asarray(v, dtype= dtype)
			v = v.astype(dtype) # must be before as tobytes won't work (memory stuff), keeeeeeep
			v = field.type_(shape= v.shape, dtype= dtype, buffer= v.tobytes(), order='F')
		return v
	
	@classmethod
	def schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
		""" alter schema to allow custom types and json export of ndarray (etc) 
		default -> str for json export
		self is the class instance
		"""
		
		for k0, v0 in cls.__fields__.items(): # self() is the instance of the class
			if issubclass(v0.type_, ndarray):
				if not isinstance(v0.default, str):
					cls.__fields__[k0].default = str(v0.default)

			elif issubclass(v0.type_, ModdedModel):
				for k1, v1 in v0.default.__fields__.items():
					if issubclass(v1.type_, ndarray) and not isinstance(v1.default, str):
						cls.__fields__[k0].default.__fields__[k1].default = str(v1.default)

		return super().schema(*args, **kwargs)

def explore_obj(obj):
	for attr in dir(obj):
		if not attr.startswith('_'):
			attr_v = getattr(obj, attr)
			print(attr, attr_v, sep='\n', end='\n\n')


""" code for int32 / float64 version, which is tricky af
# ndarray[np.float64]: lambda v: str(v.tolist()).replace(' ', ''),
# ndarray[np.float64]: lambda v: str_to_numpy(v, dtype= np.float64),
# ndarray[np.int32]: lambda v: str_to_numpy(v, dtype= np.int32),
dtype = re.sub(r'.*\.([a-zA-Z0-9]*)\]', r'\g<1>', str(field.outer_type_))
dtypes = dict(float64= np.float64, float32= np.float32, int64= np.int64, int32= np.int32
@classmethod # this only gets called when the schema_parse is called
def __modify_schema__(cls, field_schema: dict):
	print(field_schema)
	exit()
	schema = field_schema.copy()
	new_schema = {
		'properties': {
			'a': dict(type= 'string', default= ''),
			'mo_coef': dict(type= 'string', default= ''),
			'a_z': dict(type= 'string', default= ''),
	}}
	field_schema.update(schema | new_schema)
- does not work without default_factory (???) - this is because the default_factory is used to create the default value for the field
- but if the field is not required, then the default_factory is not used
- without field, the numpy array initialised normally it doesn't work

# dtype = re.sub(r'.*\.([a-zA-Z0-9]*)\]', r'\g<1>', str(field.outer_type_)) # for extracting the dtype from the outer_type_

- alter __modify_schema__ to add a custom type to the schema
- must be a classmethod 
- single argument (field_schema)
- field_schema is a dict, type must be accepted by jsonschema
- extract the type (ndarray) of the value from the type_ attribute of the field
- extract the array type from the outer

# from pydantic import BaseModel, Field, validator, Extra, fields
"""
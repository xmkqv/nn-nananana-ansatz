from pydantic import BaseModel
from rich import print

import inspect
from loguru import logger

from pydantic import BaseModel, Field
from typing import Generator
from .pretty import Colors, style

from .typfig import ModdedModel

class Pydanmod(ModdedModel):
	""" pydantic but better """

	dir_base = Field(dir(BaseModel), const= True, exclude= True)

	def docs(self) -> str:
		""" get the docs for the model """
		stylestring = style('# docs #', color= 'bright_magenta', bold= True, blink= True)
		print(stylestring)
		return get_pydanmod_docs(self)
		
	def items(self, 
		base_model  = False, 
		methods     = False,
		hidden      = False,
		attributes  = True,
		properties  = True,
	) -> Generator[str, None, None]: # dict[str, Any] (??)
		for name in dir(self):
			if name in self.dir_base and not base_model:
				continue
			if name == 'dir_base':
				continue
			if name.startswith('_') and not hidden:
				continue
			v = getattr(self, name)
			cls_d = self.__class__.__dict__
			logger.trace(
				[
					name,
					callable(v) and not methods,
					f'in cls_d: {name in cls_d}',
					f'properties:{str(properties)}',
					f'attributes:{str(attributes)}',
				]
			)
			if callable(v) and not methods:
				continue
			if name in cls_d and not properties or name not in cls_d and not attributes:
				continue
			yield (name, v)

	class Config:
		arbitrary_types_allowed = True

def get_pydanmod_docs(model: Pydanmod) -> str:
	""" get the docs for the model (instance not class)
	note: inspect.cleandoc() removes leading whitespace from the docstring (ie the tab problem)
	"""
	docs = {}
	for k, v in model.items(methods= True, attributes= False, properties= True):
		thing = getattr(model.__class__, k, None)
		if thing is None:
			continue
		stylestring = style(f'## {k} ##', color= Colors.bright_cyan, bold= True, blink= True)
		print(stylestring)
		doc = inspect.getdoc(thing)
		if doc is None:
			doc = 'no docstring yet'
		stylestring = style(doc, color= Colors.bright_green, bold= False, blink= False)
		print(stylestring)
		docs[k] = stylestring
	return docs

def get_method_docs(model: Pydanmod, method_name: str) -> str:
	method = getattr(model, method_name, None)
	if method is None:
		raise ValueError(f"Method '{method_name}' not found in the model '{model.__class__.__name__}'.")
	return inspect.getdoc(method)




if __name__ == '__main__':
	class Model(Pydanmod):

		x = 'test'

		@property
		def y(self) -> str:
			return f'{self.x}blah'
		
		class Config:
			arbitrary_types_allowed = True

	from rich.pretty import pprint
	model = Model(x= '1')
	print(model.items())


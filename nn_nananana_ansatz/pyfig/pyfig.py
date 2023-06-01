from pathlib import Path
from typing import Callable, Any
import numpy as np

from enum import Enum
from pydantic import BaseModel, Field

from .typfig import ModdedModel
from .pydanmod import Pydanmod

class DTYPES(str, Enum):
    float64: str = 'float64'
    float32: str = 'float32'

class RUNMODE(str, Enum):
    pre: str = 'pre'
    train: str = 'train'
    eval: str = 'eval'
    test: str = 'test'

class DEVICES(str, Enum):
    cpu: str = 'cpu'
    gpu: str = 'cuda'

class Pyfig(Pydanmod):

	
    
    user: 				str 	= ''
    env: 				str     = Field('.', description= 'path to root with pyproject.toml, poetry.lock, and .venv')

    group_exp: 			bool	= False

    multimode: 			str		= 'train:eval' # opt_hypam:pre:train:eval
    @property
    def mode(self) -> RUNMODE:
        return self._mode

    _mode: 			RUNMODE 	= Field(None, description= 'pre:train:eval:test', exclude= True)
    
    seed:           	int   	= 808017424 # grr
    
    n_step: 	   		int   	= 1000
    
    dtype: 				DTYPES 	= DTYPES.float32

    cudnn_benchmark: 	bool 	= True

    n_log_metric:		int  	= 50
    n_log_state:		int  	= 1

    opt_obj_key:		str		= 'e'
    opt_obj_op_key:		str 	= 'std'

    @property
    def opt_obj_op(self) -> Callable:
        objectives = dict(
            min= min, 
            max= max,
            std= lambda x: x.std(),
        )
        return objectives[self.opt_obj_op_key]

    dashboard: 			bool 	= False
    
class TyperSource(BaseModel):
    """
    - must import everything for typer into the run file 
        - eg all submodels, all enums
    # Complete set of typer types for cli arguments
    # https://typer.tiangolo.com/tutorial/options/types/
    """

    module: str = 'run' # the file name
    main: str = 'run' # the function name
    model: str = 'pyfig'
    class_sep: str = '__'
    type_sep: str = ':'
    equals: str = '='
    indent: str = '\t'
    path: Path = Path('./tmp/cli.py')
    env: str = 'env'

    cli_source: str = None

    def __init__(self, pydanmod: BaseModel, **kwargs):
        super().__init__(pydanmod= pydanmod, **kwargs)

        args_source_list = self.pydanmod_fields_to_source(pydanmod)
        class_init_source = self.class_init_source_snippet(args_source_list.copy())
        args_source = ',\n'.join(['\t' + arg for arg in args_source_list])

        self.cli_source = f"""\
import typer
from pathlib import Path
import {self.module}

app = typer.Typer()

@app.command()
def main(
{args_source}
):

    {class_init_source}
    
\t{self.module}.{self.main}({self.model})
"""

    def __str__(self):
        return self.cli_source
    
    def write(self, path: Path= None):
        path = path if path is not None else self.path
        path.parent.mkdir(parents= True, exist_ok= True)
        path.write_text(self.cli_source)
    
    def annotation_to_source(self, annotation: Any):
        """ None is the exception """
        if annotation is None:
            return 'None'
        name = annotation.__name__
        if issubclass(annotation, Enum):
            name = f'{self.module}.{name}'
        return name

    @staticmethod
    def order_kwarg(args_source_list: list[str]):
        kw = [arg for arg in args_source_list if '=' in arg]
        no_kw = [arg for arg in args_source_list if '=' not in arg]
        return no_kw + kw

    def pydanmod_fields_to_source(
            self,
            pydanmod: BaseModel, 
            _name: str = '', # do not use this argument
            show: bool = True
        ) -> list[str]:
        """ generate a list of allowed args for typer """

        args = []
        for k, field in pydanmod.__fields__.items():
                

            name = _name + field.name
            default = field.default

            if name.lower() == self.env:
                continue

            if hasattr(field, '__annotations__'):
                annotations = field.__annotations__
            else:
                annotations = [field.annotation]

            annotations = [
                a.__args__ if hasattr(a, '__args__') else [a] for a in annotations
            ]
            annotations = [_a for a in annotations for _a in a] # mro = class parent structure and flatten list pattern
            mro = [a.mro() for a in annotations]
            mro = [a for _mro in mro for a in _mro] # flatten list pattern

                    ### special cases
                    
            if BaseModel in mro:
                basemodel = [a for a in annotations if issubclass(a, BaseModel)]
                args += self.pydanmod_fields_to_source(
                    basemodel[0], _name=f'{name.capitalize()}__'
                )
                continue

            if np.ndarray in mro and isinstance(default, np.ndarray):
                default = default.tolist()
            annotations = [a if np.ndarray not in a.mro() else list for a in annotations]

            if enums := [a for a in annotations if issubclass(a, Enum)]:
                if default is not None:
                    class_ = enums[0]
                    default = f'{self.module}.{str(class_(default))}'

            elif isinstance(default, str | Path):
                default = f"'{default}'" # must be before Enum

            ### end special cases
            annotation_sources = [
                self.annotation_to_source(a) 
                    for a in annotations
            ]

            types = "|".join(annotation_sources)

            default = '' if default is None else f' = {default}'

            arg_str = f'{name}: {types}{default}'

            args += [ arg_str ]

        args = self.order_kwarg(args)
        return args


    def class_init_source_snippet(self, args_source_list: list[str]):
        
        # get the cmdline arg name 
        args = [arg.split(self.type_sep)[0].strip() for arg in args_source_list]

        # get the class and var name
        args_split = [arg.split(self.class_sep) for arg in args]

        # arrange a tuple of (class, var) for each arg
        class_and_var = [arg if len(arg) > 1 else (self.model, arg[0]) for arg in args_split]

        inputs = {}
        for (cls, var) in class_and_var:

            if cls not in inputs:
                inputs[cls] = []

            inputs[cls] += [var]

        snippet = ''
        for cls_, v in inputs.items():
            v = self.order_kwarg(v)
            if cls_ != self.model:
                _pre = cls_.capitalize() + self.class_sep

            else:
                _pre = ''
                v += [c.lower() for c in inputs if c != self.model]

            variables = ",\n".join([f'\t\t{i} = {_pre + i}' for i in v])
            cls_: str
            snippet += f"""\n\n\
\t{cls_.lower()} = {self.module}.{cls_.capitalize()}(
{variables}
)
"""

        return snippet



# from decimal import Decimal

# from pydantic import (
#     BaseModel,
#     NegativeFloat,
#     NegativeInt,
#     PositiveFloat,
#     PositiveInt,
#     NonNegativeFloat,
#     NonNegativeInt,
#     NonPositiveFloat,
#     NonPositiveInt,
#     conbytes,
#     condecimal,
#     confloat,
#     conint,
#     conlist,
#     conset,
#     constr,
#     Field,
# )

# class Model(BaseModel):
#     upper_bytes: conbytes(to_upper=True)
#     lower_bytes: conbytes(to_lower=True)
#     short_bytes: conbytes(min_length=2, max_length=10)
#     strip_bytes: conbytes(strip_whitespace=True)

#     upper_str: constr(to_upper=True)
#     lower_str: constr(to_lower=True)
#     short_str: constr(min_length=2, max_length=10)
#     regex_str: constr(regex=r'^apple (pie|tart|sandwich)$')
#     strip_str: constr(strip_whitespace=True)

#     big_int: conint(gt=1000, lt=1024)
#     mod_int: conint(multiple_of=5)
#     pos_int: PositiveInt
#     neg_int: NegativeInt
#     non_neg_int: NonNegativeInt
#     non_pos_int: NonPositiveInt

#     big_float: confloat(gt=1000, lt=1024)
#     unit_interval: confloat(ge=0, le=1)
#     mod_float: confloat(multiple_of=0.5)
#     pos_float: PositiveFloat
#     neg_float: NegativeFloat
#     non_neg_float: NonNegativeFloat
#     non_pos_float: NonPositiveFloat

#     short_list: conlist(int, min_items=1, max_items=4)
#     short_set: conset(int, min_items=1, max_items=4)

#     decimal_positive: condecimal(gt=0)
#     decimal_negative: condecimal(lt=0)
#     decimal_max_digits_and_places: condecimal(max_digits=2, decimal_places=2)
#     mod_decimal: condecimal(multiple_of=Decimal('0.25'))

#     bigger_int: int = Field(..., gt=10000)
    

"""
nb update_forward_refs
“update_forward_refs” is a method in Pydantic that resolves forward references. 
Forward references are used when a variable is referenced before it is defined. 
This method updates ForwardRefs on fields based on this Model, globalns and localns 1.

In some cases, a ForwardRef wont be able to be resolved during model creation. 
For example, this happens whenever a model references itself as a field type. When this happens, 
youll need to call update_forward_refs after the model has been created before it can be used 1.

nb BaseModel Internals
__fields_set__
Set of names of fields which were set when the model instance was initialised
__fields__
a dictionary of the model's fields
__config__
the configuration class for the model, cf. model config
"""


# class Config:
# 	""" whether to ignore, allow, or forbid extra attributes during model initialization. 
# 	Accepts the string values of 'ignore', 'allow', or 'forbid'"""
# 	extra = "forbid"

# 	"""arbitrary_types_allowed
# 	whether to allow arbitrary user types for fields 
# 	(they are validated simply by checking if the value is an instance of the type). 
# 	If False, RuntimeError will be raised on model declaration (default: False). See an example in Field Types."""
# 	# arbitrary_types_allowed = True 

# 	# tuple of types that will be returned as is when passed to the model
# 	# (i.e. not converted to a dict or list)
# 	# not included in model schema
# 	# keep_untouched: tuple = (tuple, dict, list, set, frozenset) 

# 	# underscore_attrs_are_private: bool = True # whether to treat attributes with a leading underscore as private, default True
# 	""" An Object-Relational Mapping tool, ORM, 
# 	is a framework that can help and simplify the translation between the two paradigms: 
# 	objects and relational database tables"""
# 	# orm_mode: bool = False # whether to allow ORM mode, default False
# 	# allow_mutation: bool = True # whether to allow model attribute mutation, default True
# 	# validate_assignment: bool = True # whether to validate assignment to model attributes, default False

# # dtypes = dict(float64= torch.float64, float32= torch.float32, cpu= 'cpu')[self._dtype_str]

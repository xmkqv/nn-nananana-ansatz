from pydantic import create_model

def create_pydantic_model(schema: dict) -> object:
    """Create a Pydantic model from a JSON schema."""
    return create_model(
        schema["title"],
        **{
            k: (v["type"], ...)
            for k, v in schema["properties"].items()
        }
    )

def pydantic_model_to_code():
    """Convert a Pydantic model to Python code."""
    pass


# class NDArrayAnyDtype(np.ndarray):
#     """A custom NumPy ndarray class with support for any dtype."""

#     @classmethod
#     def __get_validators__(cls):
#         yield cls.validate

#     # @classmethod
#     # def validate(cls, value: list | tuple | np.ndarray):
#     #     if isinstance(value, (list, tuple)):
#     #         value = np.array(value)
#     #     if not isinstance(value, np.ndarray):
#     #         raise ValueError("Value must be a list, tuple, or NumPy array.")
#     #     return value.view(cls)
    
#     @classmethod
#     def validate(cls, value):
#         if isinstance(value, np.ndarray):
#             value = value.tolist()
#         return value

from pydantic import BaseModel

class JSProp(BaseModel):
    title: str
    type: str  
    description: str = None # description
    default: str = None # default value 
    enum: list = None # list of possible values

class JSON(BaseModel):
    pass

# class NDArrayAnyDtype(np.ndarray):
# 	"""A custom NumPy ndarray class with support for any dtype."""

# 	@classmethod
# 	def __get_validators__(cls):
# 		yield cls.validate

# 	@classmethod
# 	def validate(cls, value: Union[list, tuple, np.ndarray]):
# 		if isinstance(value, (list, tuple)):
# 			value = np.array(value)
# 		if not isinstance(value, np.ndarray):
# 			raise ValueError("Value must be a list, tuple, or NumPy array.")
# 		return value.view(cls)

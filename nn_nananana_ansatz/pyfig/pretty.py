from pathlib import Path

from pydantic import BaseModel
from typing import Optional

log = Path('.log')

from rich import table, color
from rich import print
from rich.style import Style
from rich.text import Text

from enum import Enum

'''RESOURCES
[rich-colors](https://rich.readthedocs.io/en/stable/appendix/colors.html)
'''

class Format(BaseModel):
    bold: Optional[bool] = False
    italic: Optional[bool] = False
    underline: Optional[bool] = False
    blink: Optional[bool] = False
    reverse: Optional[bool] = False
    
class Colors(str, Enum):
    """Rich colors."""
    black = "black"
    red = "red"
    green = "green"
    yellow = "yellow"
    blue = "blue"
    magenta = "magenta"
    cyan = "cyan"
    white = "white"
    bright_black = "bright_black"
    bright_red = "bright_red"
    bright_green = "bright_green"
    bright_yellow = "bright_yellow"
    bright_blue = "bright_blue"
    bright_magenta = "bright_magenta"
    bright_cyan = "bright_cyan"
    bright_white = "bright_white"


def style(
	string: str,
	color: str = "blue",
	bold: bool = False,
	blink: bool = False,
):
	return Text(string, # redundant, still educational
	    style=Style(
			color=color, 
			bold=bold, 
			blink=blink
	))


def get_light_colors(mode='rgb'):
	""" Get light colors from rich color palette """
	
	def isLight(rgb: list[int | float]):
		""" hsp (Highly Sensitive Poo) equation from http://alienryderflex.com/hsp.html """
		import math
		r, g, b = rgb
		hsp = math.sqrt(0.299 * (r * r) + 0.587 * (g * g) + 0.114 * (b * b))
		return hsp>127.5 # True is light, False is dark
	
	rgb = [color.Color.from_rgb(*c) for c in color.STANDARD_PALETTE._colors if isLight(c)]
	return rgb if mode=='rgb' else [c.name for c in rgb]

def print_table(lines: list[list[str]]) -> table.Table:
    """ Generate a rich table """
    lines = iter(lines)
    colors = list(get_light_colors(mode= 'name'))
    info = table.Table(*[
    	table.Column(header= col, justify= "left", style= colors[i]) 
    	for i, col in enumerate(next(lines))
    ])
    [info.add_row(*l) for l in lines]
    return info

from rich.filesize import decimal
from rich.markup import escape
from rich.text import Text
from rich.tree import Tree

def walk_directory(dir_path: Path, tree: Tree, ignore= []) -> None:
	""" build tree (map of the directory) """

	dir_path = Path(dir_path)
	dir_paths = sorted( # alphabetical order
		dir_path.iterdir(),
		key=lambda dir_path: (dir_path.is_file(), dir_path.name.lower()),
	)
	
	for p in dir_paths:
		if (
			p.name.startswith(".") 
			or p.name in ignore
			or p.name.endswith(".pyc")
			or p.name.startswith("__")
		):  
			continue # do not print hidden files, etc
		
		if p.is_dir():
			style = "dim" if p.name.startswith("__") else ""
			branch = tree.add(
				f"[bold magenta]:open_file_folder: [link file://{p}]{escape(p.name)}",
				style= style,
				guide_style= style,
			)
			walk_directory(p, branch, ignore=ignore)
		else:
			name = Text(p.name, "green") # COLOR
			name.highlight_regex(r"\..*$", "magenta1") # COLOR
			name.stylize(f"link file://{p}")
			file_size = p.stat().st_size
			# name.append(f" ({decimal(file_size)})", "light blue") # COLOR
			icon = "🐍 " if p.suffix == ".py" else "📄 "
			tree.add(Text(icon) + name)

from rich.panel import Panel
from rich.console import Console

def print_tree(directory: Path, ignore: list[str]=[]) -> None:
	print('Ignoring: ', ignore)
	tree = Tree(
		f":open_file_folder: [link file://{directory}]{directory}",
		guide_style= "bright_cyan",
		style= Style(
			color="bright_cyan", 
			bold=True, 
			dim=False,
			frame=True,
		)
	)
	walk_directory(Path(directory), tree, ignore= ignore)

	panel = Panel.fit(
		tree, 
		title="Directory Tree",
		style= Style(bgcolor='black', color='white'),
	)
	console = Console()
	print(panel)

### needs testing 
from rich.theme import Theme
from rich.style import Style

THEMES = dict(
dark = Theme(
        styles={
            "info": "dim cyan",
            "warning": "bold magenta",
            "danger": "bold red",
        },
    ),
light = Theme(
        styles={
            "info": Style(color="bright_green", bold=True),
            "warning": Style(color="bright_magenta", bold=True),
            "danger": Style(color="light_slate_blue", bold=True, underline=True, italic=True),
        },
    ),
mono = Theme(
        styles={
            "info": "italic",
            "warning": "bold",
            "danger": "reverse bold",
        },
    ),
)
""" command line interface for 
- environment setup
    - pyenv
    - poetry
    - docker
    - env vars
- application setup via pyfig and typer

notes
- typer does not have run_cli command
- typer does not support fstrings as inputs
"""

from pathlib import Path
import subprocess 
import typer
from rich import print

app = typer.Typer(rich_markup_mode="rich")

def run_cli(cmd: str | list[str], cwd: str = '.'):
    if isinstance(cmd, str):
        cmd = cmd.split(' ')
    print('Running: ', cmd, sep='\n')
    out = subprocess.run(cmd, cwd= cwd, text= True, capture_output= True)
    print(out.stdout)
    return out

@app.command()
def setup_python_env(python_version: str= '3.10.10', quiet: bool= False):
    """ setup python environment with pyenv """
    out = run_cli(f'pyenv install --skip-existing{quiet * " --verbose"} {python_version}')
    out = run_cli(f'pyenv global {python_version}')
    return None

module_stack: list=['pydantic', 'typer', 'rich', 'ruff']
project_stack: list=['pytorch-lightning', 'wandb']

@app.command()
def setup_poetry_env(env_reset: bool= False, quiet: bool= False):
    print('Making project.toml')
    if env_reset or not Path('pyproject.toml').exists():
        run_cli(f'poetry init --no-interaction {not quiet * "--quiet"}')
        for s in module_stack.split('\n'):
            run_cli(f'poetry add {s}')
    return None

@app.command()
def export_env_vars():
    paths = Path().cwd().rglob('.user.sh')
    if len(list(paths)):
        for path in paths:
            run_cli(f'bash {path}')
            run_cli('".user.sh" >> .gitignore')
    else:
        print('.user.sh does not exist, you might be missing env vars') 
    return None

@app.command()
def setup_docker_image(image_name: str= 'docker_env', file: str= 'dockerfile') -> None:
    run_cli(f'docker build --tag {image_name} --file {file} .')
    run_cli(f'docker run -it --rm -p 5000:5000 {image_name}')
    return None

rc_files = ['~/.bashrc', '~/.bash_profile', '~/.zshrc', '~/.zprofile', '~/.profile']

@app.command()
def setup_poetry() -> None:
    """ setup poetry
    nb: $HOME/.local/bin on Unix.
    """
    run_cli('curl -sSL https://install.python-poetry.org | python3 -')
    for rc_f in rc_files:
        p = Path(rc_f)
        if p.exists():
            run_cli(f'echo "export PATH=$HOME/.local/bin:$PATH" >> {rc_f}')
    return None



@app.command()
def setup(
    python_version: str= '3.10.10', 
    env_reset: bool= False, 
    quiet: bool= False, 
    install_poetry: bool= False
) -> None:
    """ setup poetry, pyenv, 
    ```bash
    python app.py setup --python_version 3.10.10
    python app.py setup --install_poetry 
    python app.py setup_poetry
    ```
    """
    python_path = run_cli('which python')
    print('cwd:', Path.cwd(), 'python: ', python_path, python_version)
    
    export_env_vars()
    if install_poetry:
        setup_poetry()
    setup_python_env(python_version, quiet)
    setup_poetry_env(env_reset, quiet)
    setup_docker_image()
    return None


@app.command()
def setup_repo(name: str):
    # run_cli('gh repo create [<name>] [flags]')
    """ setup git repo! this only works for max rn """
    run_cli('git init')
    run_cli('git add --all')
    # run_cli('git add README.md')
    run_cli('git commit -m "first commit"')
    run_cli('git branch -M main')
    github_user = run_cli('git config --global user.name')
    run_cli(f'git remote add origin git@github.com:{github_user}/{name}.git')
    run_cli('git push -u origin main')

from typer import Option

def write_file(path: str, content: str):
    with open(path, 'w') as f:
        f.write(content)

poetry_config = """
[virtualenvs]
create = true
in-project = true
"""

@app.command()
def project(
    quiet: bool     = Option(False, help= 'verbose', rich_help_panel="Secondary Arguments"), 
    name: str       = Option('', help= 'if provided, create new project, else init existing project'),
    pyfig_path: str = Option('../../modules/pyfig', help= 'path to pyfig module'),
):
    """ 
    [red]init a new project [/red] :fire:
    - write poetry.toml with create=true, in-project=true
    - create or init poetry project
    - activate virtual env
    - add project stack and module stack and pyfig
    - install
    - if want to setup github repo: launch setup_repo --name <name>
    """
    write_file('poetry.toml', poetry_config)
    if name:
        run_cli(f'poetry new {name} {quiet * "--quiet"}')
    else:
        run_cli(f'poetry init {not quiet * "--no-interaction"}')
    run_cli('poetry env info')
    run_cli('poetry env use 3.10.10')
    run_cli('poetry env info')
    run_cli('poetry shell')
    run_cli('poetry env info')
    for s in module_stack + project_stack:
        run_cli(f'poetry add {s}')
    run_cli(f'poetry add --editable {pyfig_path}')
    run_cli('poetry install')
    return None

"""
# bash
python cli.py --help

# pyproject.toml
[tool.poetry.scripts]
cli = "<project>.cli:app"

# bash
poetry install
ts-diff-qc --install-completion <shell e.g. bash>


"""
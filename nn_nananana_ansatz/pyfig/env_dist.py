import random
import string
from pathlib import Path

import yaml
from pydantic import BaseModel

from .pretty import print_table

class Docker(BaseModel):

	project_dir: Path

	dockerfile: Path|str = 'dockerfile'
	image_name: str = 'demo'

	_flags: dict[str, str] = dict(
		tag= '-t',
		file= '-f',
	)

	@property
	def build_cmd(self):
		""" build docker image """
		return f'docker build -t {self.image_name} -f {self.dockerfile} {self.project_dir}'


import random
import string
import time
from pathlib import Path
from subprocess import run as sub_run

import yaml
from rich import print

log = Path('.log')

from uuid import uuid1

from pydantic import BaseModel, Field


class ResourceLimits(BaseModel):
	memory: str  = '10Gi' # Maximum amount of memory the container can use
	cpu: str  = 1 # Maximum number of CPU cores the container can use
	gpu: str | None = None # ""
	nvidia_gpu: str | None = Field(None, alias="nvidia.com/gpu") # ""

class ResourceRequests(BaseModel):
	memory: str = '10Gi'  # Amount of memory the container requests
	cpu: str = 1   # Number of CPU cores the container requests
	gpu: str | None = None # ""
	nvidia_gpu: str | None = Field(None, alias="nvidia.com/gpu") # ""

class Volume(BaseModel):
	name: str = Field(default_factory= lambda: str(uuid1())) # Name of the volume
	persistentVolumeClaim: dict[str, str]  # Name of the PersistentVolumeClaim

class volumeMount(BaseModel):
	name: str = Field(default_factory= lambda: str(uuid1())) # Name of the volume
	mountPath: str = "/scb-usra" # Path to mount the volume inside the container

class Container(BaseModel):
	name: str = Field(default_factory= lambda: str(uuid1())) # Name of the container
	image: str = "gitlab-registry.nrp-nautilus.io/usra/scb-env-one:latest" # base docker
	command: list[str] = ["/bin/bash", "-c"] # run bash at start
	args: list[str] = [". activate base; sleep infinity"] # Arguments to the command
	volumeMounts: list[volumeMount]  # Mount points for volumes
	resources: dict[str, ResourceLimits | ResourceRequests]  # Resource limits for the container

class Metadata(BaseModel): 
	""" info about the job """
	name: str       = "ata-pod"
	namespace: str  = "scb-usra" 

class Spec(BaseModel):
	containers: list[Container]
	volumes: list[Volume]
	restartPolicy: str = "Never"

class Pod(BaseModel):
	apiVersion: str = "v1"  # Kubernetes API version
	kind: str       = "Pod" # Type of Kubernetes resource
	metadata: Metadata = Metadata() # Metadata for the resource
	spec: Spec # Specification for the resource

vol1 = Volume(
	name= "scb-usra",  # name in the pod
	persistentVolumeClaim= dict(
		claimName= "scb-usra",  # ???
))

con1 = Container(
	name=       "a container",  # Name of the container
	volumeMounts= [
		volumeMount(name= "scb-usra", mountPath=  "/scb-usra"),
	],
	resources= dict(
		request= ResourceRequests(memory="10Gi", cpu= "1"),
		limits= ResourceLimits(memory="10Gi", cpu= "1"),
	)
)

class Kubernetes(BaseModel):

	namespace: str 		= 'scb-usra'
	link_cmd: str 		= 'ln -s /scb-usra /scb-usra'
	repo: str 			= ''

	my_tag: str = '-mx-'
	pod_path: Path | None = None
	_tmp: Path = Path('/tmp/kub')

	@property
	def useful_cmds():
		get_schema = 'kubectl get --raw "/openapi/v2" | jq .'
		get_schema = 'kubectl api-resources --verbs=list --namespaced -o name | xargs -n 1 kubectl explain > k8schema.txt'
		
	@property
	def pod_name(self) -> str:
		return self.my_tag + ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
	
	def get_pods(self, show= True, periodic: tuple[int]|None = None) -> str:
		pods = self.run(f'kubectl get pods -n {self.namespace}', show= show)
		if periodic:
			t0 = time.time()
			while True:
				pods = self.run(f'kubectl get pods -n {self.namespace}', show= show)
				time.sleep(periodic[0])
				t1 = time.time()
				print(f'elapsed: {t1-t0:.2f} seconds')
				if (t1-t0) > periodic[1]:
					break
		return pods
	
	def get_config(self) -> str:
		return self.run('kubectl config view')
	
	def get_context(self) -> str:
		return self.run('kubectl config current-context')
	
	def get_namespace(self) -> str:
		return self.run('kubectl get namespace')
	
	def create_pod(self, path: Path|str|None = None, pod: Pod | None = None) -> str:
		
		path = path or self._tmp / pod.metadata.name
		path = path.with_suffix(suffix= '.yaml')
		path.parent.mkdir(exist_ok= True, parents= True)
		yaml.dump(pod.dict(exclude_none= True), open(path, 'w'))

		try:
			path = path or self.pod_path
			del self.pod_path
		except AttributeError as e:
			print('pod has already been created')

		return self.run(f'kubectl create -f {path}', show= True)
	
	@staticmethod
	def run(cmd: str, show: bool= True) -> str:
		out = sub_run(cmd, shell= True, capture_output= True, text= True)
		out = out.stdout if out.returncode == 0 else out.stderr
		fmt = [o.strip().split() for o in out.splitlines()]
		if show:
			print_table(fmt, show= show)
		return out
	
	def delete_all_my_pods(self, tag: str = None, show: bool= False):
		tag = tag or self.my_tag
		pods = self.get_pods(show= False) # ['pod_name status ... \n ... \n']
		pods = [l.strip().split() for l in pods.splitlines()] # [['pod_name', 'status', '...'], ...
		del_pods = [self.run(f'kubectl delete pod/{row[0]}', show= show) for row in pods if tag in row[0]]
		pods = self.get_pods(show= show) # ['pod_name status ... \n ... \n']

	# def __init__(self):
	# 	super().__init__()
	# 	self._tmp.mkdir(exist_ok=True)
	# 	self.pod_path = (self._tmp / self.pod_name).with_suffix('.yaml')
	# 	self.pod_path.write_text(yaml.dump(self.kub_c))


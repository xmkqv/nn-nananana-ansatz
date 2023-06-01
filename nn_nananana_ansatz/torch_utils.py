import torch
import optree

def npify_tree(v: dict|list, return_flat_with_spec=False):
	if not isinstance(v, dict|list):
		if isinstance(v, torch.Tensor):
			v: torch.Tensor
			v: torch.Tensor = v.detach().cpu().numpy()
		return v
	leaves, treespec = optree.tree_flatten(v)
	leaves = [v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for v in leaves]
	if return_flat_with_spec:
		return leaves, treespec
	return optree.tree_unflatten(treespec=treespec, leaves=leaves)
import torch
import random
import numpy as np

def set_seed(seed):
	'''
		Setting a seed to make our experiments reproducible
		seed: seed value
	'''
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	random.seed(seed)


def improved(metric, best_metric, criteria):
	if criteria == "perplexity":
		if metric < best_metric:
			return True
	else:
		if metric > best_metric:
			return True
	return False


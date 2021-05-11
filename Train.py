from arguments import get_args
import Trainer
import numpy as np
import random

if __name__ == "__main__":
	args = get_args()
	global step

elif args.pretrained_model == "t5":
	Trainer.main(args)
else:
	print("model does not exist!")

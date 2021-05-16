from constants import INPUT_KEYS
import torch
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict


def generate_input(instance, mode):
	'''
		Function description
	'''

	if mode == 1:
		str_source = "ANSWER_MULTI " + instance["context"] + " <QUESTION> " + \
				instance["question"] + " <OPT> " + " <OPT> ".join(instance["answers"])
	elif mode == 2:
		str_source = "ANSWER_MULTI " + instance["context"] + " <QUESTION> " + \
				instance["question"] + " <TYPE_QUESTION> " + instance["question_type"] + \
				" <OPT> " + " <OPT> ".join(instance["answers"]) 
	elif mode == 3:
		str_source = "ANSWER_MULTI " + instance["context"] + " <QUESTION> " + \
				instance["question"]
	elif mode == 4:
		str_source = "ANSWER_MULTI " + instance["context"] + " <QUESTION> " + \
				instance["question"] + " <TYPE_QUESTION> " + instance["question_type"]

	return str_source


def convert_to_tensor(dataset, tokenizer, mode=1, max_input=512, max_output=60):
	'''
		Function description
	'''
	#print(instance["question_type"])

	datasets = defaultdict(list)
	for index, instance in enumerate(dataset):
		str_source = generate_input(instance, mode)
		str_target = "<pad> " + instance["answers"][int(instance["correct_answer_id"])]

		source = tokenizer.batch_encode_plus([str_source], max_length= max_input, 
			padding='max_length', return_tensors='pt', truncation=True)
    
		target = tokenizer.batch_encode_plus([str_target], max_length= max_output, 
			padding='max_length', return_tensors='pt', truncation=True)
    
		for input_name, input_array in source.items():
			datasets["encoder_" + input_name].append(input_array.squeeze().tolist())

		for input_name, input_array in target.items():
			datasets["decoder_" + input_name].append(input_array.squeeze().tolist())
      
	tensor_datasets = []
	for input_name in INPUT_KEYS:
		tensor_datasets.append(torch.tensor(datasets[input_name]))

	return TensorDataset(*tensor_datasets)


def get_dataloader(dataset, tokenizer, batch_size, shuffle, mode=1, max_input=512, max_output=60):
	'''
		Function description
	'''
	tensor_dataset = convert_to_tensor(dataset, tokenizer, mode, max_input, max_output)

	params = {'batch_size': batch_size,
			'shuffle': shuffle,
			'num_workers': 0
			}
	return DataLoader(tensor_dataset, **params)


from utils import set_seed, improved
import torch
from transformers import AdamW
from transformers import T5Tokenizer, T5ForConditionalGeneration
from constants import SPECIAL_TOKENS
from datasets import load_dataset
from mcr_dataset import get_dataloader
import os
from operator import itemgetter
import sacrebleu
from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer, util


def predict(model, loader, tokenizer, max_output, beam_size, device):
	'''
		Function description
	'''

	model.eval()
	predictions = []
	with torch.no_grad():
		for _, data in enumerate(loader, 0):
			input_ids, attention_mask, _, _ = data
			ids = input_ids.to(device, dtype = torch.long)
			mask = attention_mask.to(device, dtype = torch.long)
      
			generated_ids = model.generate(
				input_ids = ids,
				attention_mask = mask,
				pad_token_id=tokenizer.pad_token_id,
				max_length=max_output, 
				num_beams=beam_size,
				no_repeat_ngram_size=2,
				num_return_sequences=1,
				repetition_penalty=2.5,
				length_penalty=1.0,
				early_stopping=True,
				eos_token_id=tokenizer.eos_token_id)
      
			preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in generated_ids]
			predictions.extend(preds)
	return predictions


def measure_similarity(options, hypothesis, scorer):
	'''
		Function description
	'''

	if type(scorer) == BERTScorer:
		_, _, similarities = scorer.score(options, \
				[hypothesis for _ in range(len(options))], \
				batch_size=8, verbose=False)

	elif type(scorer) == SentenceTransformer:
		similarities = [util.pytorch_cos_sim(scorer.encode(option,convert_to_tensor=True), \
											scorer.encode(hypothesis,convert_to_tensor=True)) \
											for option in options]
	return similarities


def evaluate_and_predict(model, loader, tokenizer, max_length, beam_size, device, reference, scorer=None):
	'''
		Function description
	'''

	options = [(instance["answers"], instance["correct_answer_id"]) for instance in reference]
	correct = 0
	total = 0
	answers = []
	hypotheses = predict(model, loader, tokenizer, max_length, beam_size, device)
	for idx, (candidates_ref, hypothesis) in enumerate(zip(options, hypotheses)):
		total += 1
		similarities = measure_similarity(candidates_ref[0], hypothesis, scorer)

		index, element = max(enumerate(similarities), key=itemgetter(1))
		if index == candidates_ref[1]:
			correct += 1
			answers.append((1, candidates_ref[0][index], hypothesis))
		else:
			answers.append((0, candidates_ref[0][index], hypothesis))

	return correct/total, answers


def evaluate_bleu(model, loader, tokenizer, max_length, beam_size, device, reference):
	'''
		Function description
	'''

	model.eval()
	references = [(instance["answers"][instance["correct_answer_id"]]) for instance in reference]
	hypotheses = predict(model, loader, tokenizer, max_length, beam_size, device)

	bleu = sacrebleu.corpus_bleu(hypothesis, [references])

	return bleu.score



def evaluate_bs_accuracy(model, loader, tokenizer, max_length, beam_size, device, reference, scorer):
	'''
		Function description
	'''

	options = [(instance["answers"], instance["correct_answer_id"]) for instance in reference]
	correct = 0
	total = 0
	hypotheses = predict(model, loader, tokenizer, max_length, beam_size, device)
	for candidates_ref, hypothesis in zip(options, hypotheses):
		total += 1
		_, _, f = scorer.score(candidates_ref[0], \
				[hypothesis for _ in range(len(candidates_ref[0]))], batch_size=8, verbose=False)

		index, element = max(enumerate(f), key=itemgetter(1))
		if index == candidates_ref[1]:
			correct += 1 
	return correct/total


def evaluate_tr_accuracy(model, loader, tokenizer, max_length, beam_size, device, reference, scorer):
	'''
		Function description
	'''

	options = [(instance["answers"], instance["correct_answer_id"]) for instance in reference]
	correct = 0
	total = 0
	hypotheses = predict(model, loader, tokenizer, max_length, beam_size, device)
	for candidates_ref, hypothesis in zip(options, hypotheses):
		similarities = [util.pytorch_cos_sim(scorer.encode(candidate_ref,convert_to_tensor=True), \
											scorer.encode(hypothesis,convert_to_tensor=True)) \
											for candidate_ref in candidates_ref[0]]
		total += 1

		index, element = max(enumerate(similarities), key=itemgetter(1))
		if index == candidates_ref[1]:
			correct += 1 
	return correct/total


def evaluate_loss(model, loader, tokenizer, device):
	'''
		Function description
	'''

	model.eval()
	total_loss = 0
	n = 0
	for index, data in enumerate(loader, 0):
		input_ids, attention_mask, decoder_input_ids, decoder_attention_mask = data
		#print(decoder_attention_mask)

		input_ids = input_ids.to(device, dtype = torch.long)
		attention_mask = attention_mask.to(device, dtype = torch.long)

		y = decoder_input_ids.to(device, dtype = torch.long)
		y_ids = y[:, :-1].contiguous()
		lm_labels = y[:, 1:].clone().detach()
		lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100

		outputs = model(input_ids = ids, attention_mask = mask, \
					decoder_input_ids=y_ids, labels=lm_labels)
		loss = outputs[0]
		total_loss += loss.item()
		n += 1

	return total_loss / n


def train_epoch(model, epoch, loader, tokenizer, optimizer, print_every, device):
	'''
		Function description
	'''

	total_loss = 0.0
	n = 0
	model.train()
	for index, data in enumerate(loader, 0):
		input_ids, attention_mask, decoder_input_ids, decoder_attention_mask = data
		#print(decoder_attention_mask)

		input_ids = input_ids.to(device, dtype = torch.long)
		attention_mask = attention_mask.to(device, dtype = torch.long)

		y = decoder_input_ids.to(device, dtype = torch.long)
		y_ids = y[:, :-1].contiguous()
		lm_labels = y[:, 1:].clone().detach()
		lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100

		outputs = model(input_ids = input_ids, attention_mask = attention_mask, \
                    decoder_input_ids=y_ids, labels=lm_labels) #use_cache=False for MT5

		loss = outputs[0]
		total_loss += loss.item()
		n += 1
        
		if (index+1) % print_every == 0:
			print(f'Epoch: {epoch} | Step: {index+1} | Loss: {total_loss/n}')
			n = 0
			total_loss = 0

		loss.backward(retain_graph=False)
		optimizer.step()
		optimizer.zero_grad()


def main(args):

	set_seed(args.seed)

	device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')

	tokenizer = T5Tokenizer.from_pretrained(args.model)
	tokenizer.add_tokens(SPECIAL_TOKENS)

	train_dataset = load_dataset('quail', split='train')
	dev_dataset = load_dataset('quail', split='validation')
	test_dataset = load_dataset('quail', split='challenge')

	train_loader = get_dataloader(train_dataset, tokenizer, batch_size=args.batch_size, shuffle=True, \
						max_input = args.max_input, max_output = args.max_output)
	dev_loader = get_dataloader(dev_dataset, tokenizer, batch_size=args.batch_size, shuffle=False, \
						max_input = args.max_input, max_output = args.max_output)
	test_loader = get_dataloader(test_dataset, tokenizer, batch_size=args.batch_size, shuffle=False, \
						max_input = args.max_input, max_output = args.max_output)


	model = T5ForConditionalGeneration.from_pretrained(args.model)
	if torch.cuda.device_count() > 1:
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		model = nn.DataParallel(model)

	model = model.to(device)
	model.resize_token_embeddings(len(tokenizer))

	if args.fixed_embeddings:
		fixed_name = "shared.weight"
		for name, param in model.named_parameters():
			if fixed_name == name:
				param.requires_grad = False
				print("Freezing ", fixed_name)


	#if args.optimizer == "adam":
	optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)

	if args.early_stopping_criteria == "perplexity":
		best_metric = float('inf')
	elif args.early_stopping_criteria == "bertscore":
		scorer = BERTScorer(lang="en", device=device, batch_size=8)
		best_metric = 0.0
	elif args.early_stopping_criteria == "transformer":
		scorer = SentenceTransformer('stsb-roberta-large')
		best_metric = 0.0

	patience = args.early_stopping_patience


	for epoch in range(args.epochs):

		if patience == 0:
			print("The training will stop because it reaches the limit of patience")
			break

		train_epoch(model, epoch, train_loader, tokenizer, optimizer, args.print_every, device)

		if args.early_stopping_criteria == "perplexity":
			loss = evaluate_loss(model, dev_loader, tokenizer, device)
			validation_metric = round(math.exp(loss), 3)
		elif args.early_stopping_criteria == "bertscore":
			validation_metric = evaluate_bs_accuracy(model, dev_loader, tokenizer, args.max_output, args.beam_size, device, dev_dataset, scorer):
		elif args.early_stopping_criteria == "transformer":
			validation_metric = evaluate_bleu(model, dev_loader, tokenizer, args.max_output, args.beam_size, device, dev_dataset, scorer)
		#else:
		#	validation_metric = evaluate_bleu(model, dev_loader, tokenizer, args.max_length, args.beam_size, device, dev_dataset)

		print(f'Validation {args.early_stopping_criteria}: {validation_metric:.3f}')
		if improved(validation_metric, best_metric, args.early_stopping_criteria):
			print(f'The {args.early_stopping_criteria} improved from {best_metric:.3f} to {validation_metric:.3f}')
			best_metric = validation_metric
			print("Saving checkpoint ...")
			model.save_pretrained(args.save_dir)
			if not os.path.exists(args.save_dir + "vocab"):
				os.mkdir(args.save_dir + "vocab")
			tokenizer.save_pretrained(args.save_dir + "vocab")
			patience = args.early_stopping_patience
			print("Model saved")
		else:
			patience -= 1
			print(f'Patience ({patience}/{args.early_stopping_patience})')

	del model
	print("Loading best checkpoint ...")
	model = T5ForConditionalGeneration.from_pretrained(args.save_dir)#
	model.to(device)
	print("Model was loaded sucessfully.")


	del scorer

	if args.similarity == "bertscore":
		similarity_method = BERTScorer(lang="en", device=device, batch_size=8)
	else:
		similarity_method = SentenceTransformer('stsb-roberta-large')


	acc, results = evaluate_and_predict(model, dev_loader, tokenizer, args.max_output, args.beam_size, device, \
							dev_dataset, scorer=similarity_method)
	with open(args.save_dir + "/dev.out", "w") as f:
		f.write(str(acc) + "\n")
		for result in results:
			f.write(str(result[0]) + "\t" + result[1] + "\t" + result[2] + "\n")


	acc, results = evaluate_and_predict(model, test_loader, tokenizer, args.max_output, args.beam_size, device, \
							test_dataset, scorer=similarity_method)
	with open(args.save_dir + "/test.out", "w") as f:
		f.write(str(acc) + "\n")
		for result in results:
			f.write(str(result[0]) + "\t" + result[1] + "\t" + result[2] + "\n")


	


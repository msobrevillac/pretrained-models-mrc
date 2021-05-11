""" File to hold arguments """
import argparse

# data arguments

parser = argparse.ArgumentParser(description="Main Arguments")

# training parameters
parser.add_argument(
  '-epochs', '--epochs', type=int, required=False, default=3, help='Number of training epochs')
parser.add_argument(
  '-print-every', '--print_every', type=int, default=500, required=False, help='Print the loss/ppl every training steps')

parser.add_argument(
  '-batch-size', '--batch_size', type=int, required=False, default=32, help='Batch size')
parser.add_argument(
  '-max-input', '--max_input', type=int, required=False, default=512, help='Max length in encoder')
parser.add_argument(
  '-max-output', '--max_output', type=int, required=False, default=60, help='Max length in decoder')


# hyper-parameters
parser.add_argument(
  '-optimizer','--optimizer', type=str, required=False, default="AdamW", help='Optimizer that will be used')
parser.add_argument(
  '-lr','--learning_rate', type=float, required=False, default=0.0001, help='Learning rate')
parser.add_argument(
  '-adam-epsilon','--adam_epsilon', type=float, default=1.0e-8, required=False, help='Adam epsilon')

parser.add_argument(
  '-beam-size','--beam_size', type=int, required=False, default=5, help='Beam search size ')
parser.add_argument(
  '-beam-alpha', '--beam_alpha', type=float, required=False, default=0.2, help='Alpha value for Beam search')

parser.add_argument(
  '-seed', '--seed', type=int, default=32, required=False, help='Seed')
parser.add_argument(
  '-gpu','--gpu', action='store_true', required=False, help='Use GPU or CPU')

parser.add_argument(
  '-fixed-embed','--fixed_embeddings', action='store_true', required=False, help='Use GPU or CPU')

parser.add_argument(
  '-save-dir','--save_dir', type=str, required=True, default="/content/", help='Output directory')

parser.add_argument(
  '-model','--model', type=str, required=False, default="t5-base", help='Path for a pre-trained model file')

parser.add_argument(
  '-pretrained-model', '--pretrained-model', default='t5', type=str, choices=['bart', 't5'], required=False, help='Pretrained model to be used')

parser.add_argument(
  '-early-stopping-patience','--early-stopping-patience', type=int, default=15, required=False, help='Early stopping patience')

parser.add_argument(
  '-early-stopping-criteria','--early-stopping-criteria', type=str, default="perplexity", choices=['perplexity', 'transformer', 'bertscore'], help='Criteria to stop training (perplexity|transformer|bertscore)')

parser.add_argument(
  '-similarity','--similarity', type=str, default="bertscore", choices=['transformer', 'bertscore'], help='Function to measure the similarity between answer and options (transformer|bertscore)')


def get_args():
  args = parser.parse_args()
  return args



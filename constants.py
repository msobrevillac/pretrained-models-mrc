SPECIAL_TOKENS_BERT = ["[CLS]", "[SEP]", "[PAD]", "[unused0]"]
MODEL_INPUTS = ["input_ids", "labels", "token_type_ids", "attention_mask"]
PADDED_INPUTS = ["input_ids", "labels", "token_type_ids", "attention_mask"]

INPUT_KEYS = ["encoder_input_ids", "encoder_attention_mask",
			"decoder_input_ids", "decoder_attention_mask"]
SPECIAL_TOKENS = ["<OPT>", "ANSWER_MULTI",
                  "<TYPE_QUESTION>", "<QUESTION>"]

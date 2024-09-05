


# def _build_tokenized_answer(
#     prompt: str,
#     answer: str,
#     tokenizer: Optional[PreTrainedTokenizerBase] = None,
# ) -> Dict[str, Any]:
#     """
#     Build tokenized response, handling vision models and different tokenizers.
#     """

#     def tokenize(text):
#         return tokenizer(text, add_special_tokens=False)

#     full_tokenized = tokenize(prompt + answer)
#     prompt_tokenized = tokenize(prompt)

#     prompt_input_ids = prompt_tokenized["input_ids"]
#     answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
#     answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]

#     if len(full_tokenized["input_ids"]) != len(prompt_input_ids + answer_input_ids):
#         raise ValueError("Prompt input ids and answer input ids should have the same length.")

#     # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
#     # can be merged together when tokenizing prompt+answer. This could result
#     # on the last token from the prompt being different when tokenized on its own
#     # vs when done as prompt+answer.
#     response_token_ids_start_idx = len(prompt_input_ids)

#     # If tokenized prompt is different than both prompt+answer, then it means the
#     # last token has changed due to merging.
#     if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
#         response_token_ids_start_idx -= 1

#     prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
#     prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

#     if len(prompt_input_ids) != len(prompt_attention_mask):
#         raise ValueError("Prompt input ids and attention mask should have the same length.")

#     return_dict = {
#         "prompt_input_ids": prompt_input_ids,
#         "prompt_attention_mask": prompt_attention_mask,
#         "input_ids": answer_input_ids,
#         "attention_mask": answer_attention_mask,
#     }
#     return return_dict


# def _process_prompt(
#     prompts: List[str], processor: Optional[Callable], tokenizer: PreTrainedTokenizerBase
# ) -> List[Dict[str, List[int]]]:
#     """
#     Processes a list of prompts by tokenizing them, optionally using a processor for additional processing.
#     """
#     prompt_tokens = [tokenizer(prompt, add_special_tokens=False) for prompt in prompts]
#     return [{f"prompt_{k}": v for k, v in tokens.items()} for tokens in prompt_tokens]


# def _process_answer(
#     prompts: List[str],
#     answers: List[str],
#     processor: Optional[Callable],
#     tokenizer: PreTrainedTokenizerBase,
# ) -> List[Dict[str, Any]]:
#     return [
#         _build_tokenized_answer(prompt, answer, processor=processor, tokenizer=tokenizer)
#         for prompt, answer, image in zip(prompts, answers)
#     ]




# def _adjust_prompt_length(
#     prompt_tokens: List[Dict[str, List[int]]],
#     chosen_tokens: List[Dict[str, List[int]]],
#     rejected_tokens: List[Dict[str, List[int]]],
# ) -> List[int]:
#     prompt_len_input_ids = []
#     for p_tokens, c_tokens, r_tokens in zip(prompt_tokens, chosen_tokens, rejected_tokens):
#         c_len = len(c_tokens["prompt_input_ids"])
#         r_len = len(r_tokens["prompt_input_ids"])
#         min_len = min(c_len, r_len)

#         for k, v in p_tokens.items():
#             p_tokens[k] = v[:min_len]

#         num_diff_tokens = sum([a != b for a, b in zip(c_tokens["prompt_input_ids"], r_tokens["prompt_input_ids"])])
#         num_diff_len = abs(c_len - r_len)
#         if num_diff_tokens > 1 or num_diff_len > 1:
#             raise ValueError(
#                 "Chosen and rejected prompt_input_ids might only differ on the last token due to tokenizer merge ops."
#             )
#         prompt_len_input_ids.append(min_len)
#     return prompt_len_input_ids

# def _add_special_tokens(
#     tokenizer: PreTrainedTokenizerBase,
#     prompt_len_input_ids: List[int],
#     prompt_tokens: List[Dict[str, List[int]]],
#     chosen_tokens: List[Dict[str, List[int]]],
#     rejected_tokens: List[Dict[str, List[int]]],
# ) -> Tuple[List[Dict[str, List[int]]], List[Dict[str, List[int]]], List[Dict[str, List[int]]]]:
#     for i in range(len(prompt_tokens)):
#         prompt_tokens[i], chosen_tokens[i], rejected_tokens[i] = add_bos_token_if_needed(
#             tokenizer.bos_token_id,
#             prompt_len_input_ids[i],
#             prompt_tokens[i],
#             len(chosen_tokens[i]["prompt_input_ids"]),
#             chosen_tokens[i],
#             len(rejected_tokens[i]["prompt_input_ids"]),
#             rejected_tokens[i],
#         )

#         chosen_tokens[i], rejected_tokens[i] = add_eos_token_if_needed(
#             tokenizer.eos_token_id, chosen_tokens[i], rejected_tokens[i]
#         )
#     return prompt_tokens, chosen_tokens, rejected_tokens


# def _truncate_tokens(
#     chosen_tokens: List[Dict[str, List[int]]],
#     rejected_tokens: List[Dict[str, List[int]]],
#     prompt_tokens: List[Dict[str, List[int]]],
#     args: DPOConfig,
# ) -> None:
#     """
#     Truncates the tokens in chosen, rejected, and prompt sequences to ensure they fit within the maximum length constraints.
#     """
#     if args.truncation_mode not in ["keep_start", "keep_end"]:
#         raise ValueError(f"Invalid truncation mode: {args.truncation_mode}")

#     for c_tokens, r_tokens, p_tokens in zip(chosen_tokens, rejected_tokens, prompt_tokens):
#         longer_response_length = max(len(c_tokens["input_ids"]), len(r_tokens["input_ids"]))

#         # if combined sequence is too long, truncate the prompt
#         for answer_tokens in [c_tokens, r_tokens, p_tokens]:
#             if len(answer_tokens["prompt_input_ids"]) + longer_response_length > args.max_length:
#                 if args.truncation_mode == "keep_start":
#                     for k in ["prompt_input_ids", "prompt_attention_mask"]:
#                         answer_tokens[k] = answer_tokens[k][: args.max_prompt_length]
#                 elif args.truncation_mode == "keep_end":
#                     for k in ["prompt_input_ids", "prompt_attention_mask"]:
#                         answer_tokens[k] = answer_tokens[k][-args.max_prompt_length :]

#         # if that's still too long, truncate the response from the end
#         for answer_tokens in [c_tokens, r_tokens]:
#             if len(answer_tokens["prompt_input_ids"]) + longer_response_length > args.max_length:
#                 for k in ["input_ids", "attention_mask"]:
#                     answer_tokens[k] = answer_tokens[k][: args.max_length - args.max_prompt_length]



# def _build_sequence_tokens(
#     batch: Dict[str, List[int]], tokens: List[Dict[str, List[int]]], args: DPOConfig, prefix: str
# ) -> None:
#     for token in tokens:
#         sequence_tokens = {f"{prefix}_{k}": token[f"prompt_{k}"] + token[k] for k in ["input_ids", "attention_mask"]}

#         # the labels are the same as the input_ids, but with the prompt tokens masked
#         sequence_tokens[f"{prefix}_labels"] = sequence_tokens[f"{prefix}_input_ids"][:]
#         sequence_tokens[f"{prefix}_labels"][: len(token["prompt_input_ids"])] = [args.label_pad_token_id] * len(
#             token["prompt_input_ids"]
#         )
#         for k, v in sequence_tokens.items():
#             batch[k].append(v)

# def _append_prompt_tokens_to_batch(batch: Dict[str, List[int]], prompt_tokens: List[Dict[str, List[int]]]) -> None:
#     for p_tokens in prompt_tokens:
#         for k, v in p_tokens.items():
#             batch[k].append(v)

# def _tokenize(
#     features: Dict[str, List],
#     tokenizer: PreTrainedTokenizerBase,
#     args: DPOConfig,
#     processor: Optional[Callable] = None,
#     model: Optional[PreTrainedModel] = None,
# ) -> Dict[str, List]:
#     """
#     Tokenizes and processes a batch of input features using the provided tokenizer and processor.
#     """
#     batch = defaultdict(list)

#     prompt = features["prompt"]

#     prompt_tokens = _process_prompt(prompt, processor, tokenizer, None)
#     chosen_tokens = _process_answer(prompt, features["chosen"], processor, tokenizer, None)
#     rejected_tokens = _process_answer(prompt, features["rejected"], processor, tokenizer, None)

#     prompt_len_input_ids = _adjust_prompt_length(prompt_tokens, chosen_tokens, rejected_tokens)

#     prompt_tokens, chosen_tokens, rejected_tokens = _add_special_tokens(
#         tokenizer, prompt_len_input_ids, prompt_tokens, chosen_tokens, rejected_tokens
#     )

#     _truncate_tokens(chosen_tokens, rejected_tokens, prompt_tokens, args)

#     _build_sequence_tokens(batch, chosen_tokens, args, "chosen")
#     _build_sequence_tokens(batch, rejected_tokens, args, "rejected")

#     _append_prompt_tokens_to_batch(batch, prompt_tokens)

#     return dict(batch)


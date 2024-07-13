# 2024-07-06 16:21:34



```sh
# run SFT
python -u train.py \
    model=blank_model model.name_or_path=/PATH/TO/LLAMA/WEIGHTS \
    model.block_name=LlamaDecoderLayer \
    datasets=[hh,shp] \
    loss=sft exp_name=anthropic_shp_sft_llama_7b \
    gradient_accumulation_steps=2 batch_size=64 \
    eval_batch_size=32 \
    trainer=FSDPTrainer \
    sample_during_eval=false

```



this fork adds drpo https://github.com/eric-mitchell/direct-preference-optimization/compare/main...haobozhang:direct-preference-optimization:main

ompute the DPO loss for Plackett-Luce model https://github.com/spacegoing/pldpo/commit/80d7f1c1e0042f0858461b338cc4f5de7040a635


LORA https://github.com/gzliyu/direct-preference-optimization/commit/43147a224387b4047f5921e95d77a751243e29b0

https://github.com/andersonbcdefg/dpo-lora
https://github.com/unslothai/unsloth
unsloth/llama-3-8b-bnb-4bit",  


Why does the trainer log every step?
self.training_bar.write(str(logs))
https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_callback.py#L626
https://github.com/huggingface/transformers/blob/1082361a1978d30db5c3932d1ee08914d74d9697/src/transformers/utils/notebook.py#L335

# 2024-07-07 13:10:56

It's kinda working! Now for a direct comparison
- without sft
- shall I use GENIE? or a tiny subet https://github.com/Joshuaclymer/GENIES/blob/22c8afb2551851fb3f2d1a2dcf70e7608908f6b1/src/api/compute_generalization_metrics.py#L11
  - Train the base model on each source distribution and then evaluate it on the target distribution.

# 2024-07-07 17:38:16

OK it seems to be running now
- try on base model
- for longer


# 2024-07-08 08:15:56

Weird errors with some adapters not changing

    What are the normal training details?
    - https://github.com/eric-mitchell/direct-preference-optimization/blob/main/config/config.yaml
    - batch_size: 4
    -  batch_size / (grad_accumulation_steps * num_gpus)
    -  lr: 5e-7
    -  gradient_accumulation_steps: 1
    -  max_grad_norm: 10.0
    -  
    # the maximum allowed length for an input (prompt + response)
    max_length: 512

    # the maximum allowed length for a prompt
    max_prompt_length: 256
    n_eval_examples: 256
    hh has how many train: 169,352

# 2024-07-08 18:51:28

Ok yeah only the default adapter is changing?

Hmm this seems to be working? this was with toxic dpo though

- repro default    0.555029
- None             0.554399
- drpo default     0.531139

# 2024-07-08 20:58:59

Ah found the problem1!! I was passing peft_config to the trainer, which unloaded merged ! and then made it's own adapter, fn hell man

Data plan:
- train on my prefered mix of https://huggingface.co/datasets/nvidia/HelpSteer
  - https://huggingface.co/datasets/sablo/HelpSteer_binarized  best and worst scoring (average of helpfulness and correctness) 
- then test on TruthfulQA, and anthropic or eluether Sycophancy :)

note 1e-6 does work for reprPO, I think, no wait it was a random walk, too low

## HelpSteer

of https://huggingface.co/datasets/nvidia/HelpSteer
  - https://huggingface.co/datasets/sablo/HelpSteer_binarized  best and worst scoring (average of helpfulness and correctness) 



https://github.com/jondurbin/bagel has great dpo
https://github.com/jondurbin/bagel/blob/3c7d2410a5a5ad2fd31b63529ef541135feefce4/bagel/data_sources/helpsteer.py#L4

https://huggingface.co/datasets/Columbia-NLP/DPO-HelpSteer
  We reformat the nvidia/HelpSteer dataset into a common format used across all DPO datasets in this collection. Specifically, we:

      convert all scores to a [1, 10] scale by np.mean([helpfulness+1, correctness+1, coherence+1, complexity+1, 4-verbosity])*2.0
      the original dset considers 4 responses per prompt. We construct preference pairs by 1) take the best scoring response as chosen, and 2) randomly sample responses with score lower than best response as rejected. We skip prompts/data rows where all responses have the same score.

# 2024-07-11 14:00:21

Circuit breaker was released?!

what can I learn?

## norm
they used norm instead of mse in a few places
  - the adapter_good or retain control
  - `retain_loss = torch.norm(lora_retain_hidden - orig_retain_hidden, dim=-1, p=2, dtype=torch.float).nanmean()`
  - mean(or p2 norm over the last dim). not the same as mse it seems
  - the norm is over the h dim of each layer

The circuit breaker loss is diff, as it's just eradicating instead of shifting bad behaviour, but they also do a layer norm... which gives me ideas


maybe I should retain magnitude and direction, but change only direction?
That kind of makes sense, focus on that!

## mask

- [x] they use a hidden layer mask


```
retain_attention_mask # this is the normal attention mask for the good inputs
# times by layers
layers_retain_attention_mask = retain_attention_mask.repeat(len(orig_retain_outputs), 1, 1).unsqueeze(-1)
```

## log

they log the cosine similarity, good idea
they do evals.. good idea

## detach

yes they detach the orig hs


## hparams


150 steps only!  but I might need more
lora dropout 0.05
all modules
num_train_epochs=3.0,
model_max_length=8192,
per_device_train_batch_size=16,
learning_rate = 1e-4
wd 0
bg16 true
tf32 true
gradient_checkpointing True !!


# 2024-07-11 15:14:34

- [x] Masking
- [x] mean sum error over hs, not mse over all
- [x] hparams
- [ ] they didn't use 4bit! 
  - [ ] 8bit
  - [x] grad checkpoint
- [ ] try only changing directions? hmm but then it will try to make it -1e-12 to satify both?


# ideas
https://claude.ai/chat/6683ae44-b3bb-4ef7-b29d-292aaf752501
- 8bit
- from claude
  - cosine
  - norm mse
  - gradient penalty like gans use to stabalise


# chat with claude 2024-07-12 12:06:05

Certainly. After our discussion and refinements, here are the remaining promising ideas, rated on a scale of 1-10 based on their potential effectiveness, stability, and ease of implementation for your specific use case:

1. Normalized MSE (8/10):
```python
def normalized_mse(x, y):
    x_norm = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
    y_norm = y / (torch.norm(y, dim=-1, keepdim=True) + 1e-8)
    return torch.mean((x_norm - y_norm)**2)
```
Pros: Focuses on direction while maintaining MSE's stable gradients.
Cons: May still be sensitive to outliers.

2. Combined Magnitude and Direction Loss (9/10):
```python
def combined_loss(x, y, alpha=0.5):
    cos_sim = torch.nn.functional.cosine_similarity(x, y, dim=-1)
    direction_loss = 1 - cos_sim
    magnitude_loss = torch.norm(x, dim=-1) - torch.norm(y, dim=-1)
    return alpha * direction_loss + (1-alpha) * magnitude_loss
```
Pros: Balances both direction and magnitude, adaptable via alpha parameter.
Cons: May require tuning to find the optimal alpha.

3. Adaptive Loss (8/10):
```python
def adaptive_loss(x, y, threshold=0.9):
    cos_sim = torch.nn.functional.cosine_similarity(x, y, dim=-1)
    mse = torch.mean((x - y)**2, dim=-1)
    return torch.where(cos_sim > threshold, mse, 1 - cos_sim)
```
Pros: Adapts based on current similarity, potentially combining benefits of different approaches.
Cons: Threshold selection may require tuning.

4. Top-k MSE (7/10):
```python
def top_k_mse(x, y, k):
    diff = (x - y)**2
    top_k_diff = torch.topk(diff, k, dim=-1).values
    return torch.mean(top_k_diff)
```
Pros: Focuses on most significant differences, can reduce noise.
Cons: Choosing k may require tuning, may lose some information.

5. Weight Decay (as a form of regularization) (7/10):
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
```
Pros: Simple to implement, helps prevent overfitting.
Cons: Affects all weights uniformly, which may not always be desirable.

6. Gradient Clipping (6/10):
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
Pros: Can help stabilize training, especially useful if you encounter exploding gradients.
Cons: Doesn't address vanishing gradients, may slow convergence if threshold is too low.

These methods are ordered roughly by their potential effectiveness for your task. The Combined Magnitude and Direction Loss seems particularly promising as it allows you to balance the importance of direction and magnitude explicitly.

Would you like to discuss how to implement or combine any of these approaches in your specific use case?

# 2024-07-14 07:29:15 experments


None      0.521159
DPO       0.522979
TQA scores (mean prob on correct ans)
- original mse ReprPO    0.555155
- Using only RR loss and weight decay: ReprPO: 0.536886 (this seems lik it could beenfit from longer training) as logits/rejected and so on started improving, meaning increased coherency
- topk 0.5%    0.319544 (try more here...)
- direction ReprPO    0.554511
- CKA ReprPO    0.175671 (this one never even shows a hint of success)

| Method                                         | TQA Score |
|------------------------------------------------|-----------|
| CKA ReprPO                                     | 0.175671  |
| Topk 0.5%                                      | 0.319544  |
| None                                           | 0.521159  |
| DPO                                            | 0.522979  |
| Using only RR loss and weight decay: ReprPO    | 0.536886  |
| Direction ReprPO                               | 0.554511  |
| Original mse ReprPO                            | 0.555155  |

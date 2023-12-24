"""A set of sampling helpers, each of which can be called from sample.py.

These include:
1. top_p
2. beam_search
3. speculative_sampling

You can combine speculative_sampling or beam_search with top_p and top_k.
See sample.py for usage examples.
"""
import torch
from torch.nn import functional as F
"""
Mask logits according to the top_p algorithm (https://arxiv.org/abs/1904.09751).

The algorithm adds indices in descending order of probability until the total 
probability >= p. It then masks the logits of indices that aren't included in
this summation. We mask by setting the logits to -inf.

Usage:
    logits, _ = model(idx)
    logits = top_p(logits, p=0.95)

Args:
- logits: Unnormalized float logits of size [batch_size, ..., d]. It's assumed 
  that we are computing over the last index.
- p: A float probability mass to reach.

Returns:
- logits: The masked logits.
"""


def top_p(logits, p=0.95):
    # Get probs / indices in descending order, along with cumulative sum.
    probs = torch.softmax(logits, dim=-1)
    top_values, top_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative_sum = torch.cumsum(top_values, -1)

    # Mask out all the values after we pass the top_p by first getting all
    # the values < top_p and then including the next index.
    mask = cumulative_sum < p
    prepend = torch.ones(mask.shape[:-1] + (1,)).to(logits.device)
    mask = torch.cat([prepend, mask[..., :-1]], dim=-1).to(torch.bool)

    # Apply the mask to the logits according to the top_indices.
    mask_scatter = torch.ones_like(logits).to(torch.bool).scatter_(
        index=top_indices, src=mask, dim=-1)
    return torch.masked_fill(logits, ~mask_scatter, -torch.inf)


"""
Run beam_search using the given model on the idx input.

Usage:
    results = beam_search(model, idx, max_new_tokens, num_beams=5, 
                          temperature=1., top_k=None, top_p=0.95, 
                          return_beams=False, return_log_probs=False)
    max_beam = results['max_beam']

Args:
- model: A GPT model from model.py.
- idx: A torch tensor of size [batch_size, seq_length] containing the input.
- max_new_tokens: An int specifying the maximum number of tokens to generate.
- num_beams: An int specifying the number of beams to use.
- temperature: A float specifying the temperature to use.
- top_k: An int specifying the number of top k to use. Don't run top_k if None 
  or <= 0.
- top_p: A float specifying the top p to use. Don't run if None, not between 0 
  and 1, or if top_k is set.
- return_beams: A boolean specifying whether to return the beams.
- return_log_probs: A boolean specifying whether to return the log probs.

Returns a dict that has the following keys:
- max_beam: A torch tensor of size [batch_size, max_new_tokens + 1] containing
  the most likely beam.
- beams: None if not return_beams, else the final beams of size
  [batch_size, num_beams, max_new_tokens + 1].
- log_probs: None if not return_log_probs, else the log probs of the final 
  beams, has size [batch_size, num_beams].
"""


def beam_search(model,
                idx,
                max_new_tokens,
                num_beams=5,
                temperature=1.,
                top_k=None,
                top_p=None,
                return_beams=False,
                return_log_probs=False):
    block_size = model.config.block_size
    batch_size, _ = idx.shape
    log_beam_probs = torch.zeros(batch_size, num_beams).to(idx.device)
    idx_beams = idx[:, None, :].repeat((1, num_beams, 1))

    for num_token in range(max_new_tokens):
        if num_token == 0:
            # Do a single beam for the first token. We do this to get around
            # thinking about the variety issue.
            idx_beam_cond = idx_beams[:, 0, :]
        elif idx_beams.size(2) <= block_size:
            # Use the entire sequence if it's less than the block size, but
            # reshape to put the beams into the batch.
            idx_beam_cond = idx_beams.view(batch_size * num_beams, -1)
        else:
            # Use the last block_size tokens. Also reshape to put the beams into
            # the batch.
            idx_beam_cond = idx_beams[:, :, -block_size:].view(
                batch_size * num_beams, -1)

        # Get the logits and log probs for the next token.
        logits, _ = model(idx_beam_cond)
        logits = logits[:, -1, :] / temperature
        logits = model.get_top_k_top_p(logits, top_k, top_p)
        vocab_size = logits.shape[-1]
        log_probs = torch.log_softmax(logits, dim=-1)

        if num_token == 0:
            # For the first token, we can abridge the process and just take the
            # top num_beams tokens as the next beams. The associated log_probs
            # are accounted for in the log_beam_probs.
            top_values, top_indices = torch.topk(log_probs, num_beams)
            log_beam_probs += top_values
            idx_beams = torch.cat([idx_beams, top_indices[:, :, None]], dim=-1)
            continue

        # For num_token > 0, we need to get the top num_beams * num_beams, then
        # prune to the top num_beams. Start by adding the running log_beam_probs
        # to the current log_probs.
        log_probs = log_probs.view((batch_size, num_beams, -1))
        log_probs += log_beam_probs[:, :, None]
        # Now get the top scores out of all of the beams.
        log_probs = log_probs.view(batch_size, -1)
        top_values, top_indices = torch.topk(log_probs, num_beams)
        # Index into the top choices to get the new beams and log_beam_probs.
        word_choice = torch.remainder(top_indices, vocab_size)
        beam_index = (top_indices / vocab_size).long()
        log_beam_probs = log_probs[:, top_indices][:, 0, :]
        select_beams = idx_beams[:, beam_index][:, 0, :, :]
        idx_beams = torch.cat([select_beams, word_choice[:, :, None]], dim=-1)

    # We always indexed into the log_probs according to the sorted top_indices.
    # Consequently, the first idx_beams is the beam with the maximal log_prob.
    max_beam = idx_beams[:, 0]
    ret = {'max_beam': max_beam, 'beams': None, 'log_probs': None}
    if return_beams:
        ret['beams'] = idx_beams
    if return_log_probs:
        ret['log_probs'] = log_beam_probs
    return ret


"""
Run Speculative Sampling (https://arxiv.org/abs/2302.01318)

This requires batch size = 1. Doing it with batch size > 1 is an optimization
involving careful tracking of the lengths. We can do this later.

Usage:
    result = speculative_sampling(target_model, draft_model, draft_length, idx, 
                                  max_new_tokens, temperature=1., top_k=None, 
                                  top_p=None)

Args:
- target_model: A GPT model from model.py. This is a bigger model and is the
  arbiter for whether to accept or reject a token.
- draft_model: A GPT model from model.py. This is typically a much smaller model 
  than the target model and is used to generate a draft sequence from which the
  target model will accept or reject tokens.
- draft_length: An int specifying the length of the draft sequence that the
  draft model will generate before presenting to the target model.
- idx: A torch tensor of size [batch_size, seq_length] containing the input.
- max_new_tokens: An int specifying the maximum number of tokens to generate.
- temperature: A float specifying the temperature to use.
- top_k: An int specifying the number of top k to use. Don't run top_k if None
  or <= 0.
- top_p: A float specifying the top p to use. Don't run if None, not between 0
  and 1, or if top_k is set.
"""


def speculative_sampling(target_model,
                         draft_model,
                         draft_length,
                         idx,
                         max_new_tokens,
                         temperature=1.,
                         top_k=None,
                         top_p=None):
    batch_size, starting_seq_len = idx.shape
    assert (batch_size == 1)

    while idx.size(1) - starting_seq_len < max_new_tokens:
        num_generated_tokens = idx.size(1) - starting_seq_len
        draft_length = min(draft_length, max_new_tokens - num_generated_tokens)

        # Run the draft model to get the probabilities and draft indices.
        draft_probs_all = []
        draft_probs_chosen = []
        for _ in range(draft_length):
            idx_cond = draft_model.get_idx_cond(idx)
            logits = draft_model.get_last_logits(idx_cond, temperature, top_k,
                                                 top_p)
            probs = F.softmax(logits, dim=-1)
            draft_probs_all.append(probs[:, None])
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            draft_probs_chosen.append(probs.gather(dim=-1, index=idx_next))

        # Now run the target model on the draft indices. Because the model is
        # causal, this will get all of the requisite logits at once.
        # NOTE: We can optimize get_all_logits to only run the lm_head on the
        # last draft_length + 1 logits.
        idx_cond = target_model.get_idx_cond(idx)
        logits = target_model.get_all_logits(idx_cond, temperature, top_k,
                                             top_p)
        logits = logits[:, -draft_length - 1:]
        target_probs = torch.softmax(logits, dim=-1)
        target_probs_dl = target_probs[:, :-1].gather(
            dim=-1, index=idx_cond[:, -draft_length:, None])[:, :, 0]
        target_probs_end = target_probs[:, -1]
        draft_probs_chosen = torch.cat(draft_probs_chosen, dim=1)

        # Now get the accept probabilities. If all accept, then we can sample an
        # additional token. If not, then we need to first only keep up to what
        # was accepted and then sample a tokenfrom the (q - p)_+ distribution.
        target_div_draft = target_probs_dl / (draft_probs_chosen + 1e-8)
        target_div_draft = torch.min(target_div_draft,
                                     torch.ones_like(target_div_draft))
        random_uniform = torch.FloatTensor(batch_size, draft_length) \
            .uniform_(0, 1).to(idx.device)
        accept = random_uniform < target_div_draft
        if accept.all():
            # Accepting everything.
            sample = torch.multinomial(target_probs_end, num_samples=1)
            idx = torch.cat((idx, sample), dim=1)
        else:
            # The first rejection is when accept = False, i.e the first argmin.
            first_rejection = torch.argmin(accept, dim=1)
            # Cull from idx everything that wasn't before the first rejection.
            idx = idx[:, :first_rejection[0] - draft_length]
            # Get the (target - draft)_+ distribution to sample from.
            draft_probs_all = torch.cat(draft_probs_all, dim=1)
            target_minus_draft = target_probs[:, first_rejection[
                0]] - draft_probs_all[:, first_rejection[0]]
            target_minus_draft = F.relu(target_minus_draft)
            target_minus_draft_sum = target_minus_draft.sum(1, keepdims=True)
            target_minus_draft = target_minus_draft / target_minus_draft_sum
            # And now sample from (target - draft)_+.
            sample = torch.multinomial(target_minus_draft, num_samples=1)
            idx = torch.cat((idx, sample), dim=1)
    return idx


def test_top_p_indices():
    batch_size = 2
    seq_length = 3
    vocab_size = 5
    # TODO:
    # 1. Test with only one choice.
    # 2. Set a seed with even probs. Ensure that it comes back fixed.
    # 3. Test with two choices at 0.6 and 0.4 and no seed. Run 100 times and make sure it
    #     comes back with both choices possible when p = 0.8.
    # 4. Do the same and make sure it comes back with only one choice when p = 0.6.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def test_beam_search():
    pass


def test_speculative_sampling():
    pass

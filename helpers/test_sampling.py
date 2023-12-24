"""Testing for the functions in sampling.py"""
import numpy as np
import torch

import sampling

def test_top_p():
    batch_size = 2
    seq_len = 3
    vocab_size = 4
    logits = torch.rand(batch_size, seq_len, vocab_size)

    # Test that p=1 returns the identiy.
    logits_top_p = sampling.top_p(logits, p=1)
    assert torch.all(logits_top_p == logits)

    # Test that p=0 throws an error.
    try:
        fails = sampling.top_p(logits, p=0)
        assert False
    except ValueError as e:
        pass

    # Test that p>1 throws an error.
    try:
        fails = sampling.top_p(logits, p=5)
        assert False
    except ValueError as e:
        pass

    # Test that the right logits are returned masked when p = 0.9.
    np_logits = np.array([[[0.95, 0.05, 0., 0.], [0.1, 0.2, 0.3, 0.4], [0.2, 0.4, 0.05, 0.35]], 
                          [[0.0, 0.05, 0.91, 0.04], [0.4, 0.3, 0.22, 0.08], [0.2, 0.4, 0.05, 0.35]]])
    torch_logits = torch.log(torch.from_numpy(np_logits))
    logits_top_p = sampling.top_p(torch_logits, p=0.9)
    expected = torch.from_numpy(np.array([
        [[0.95, -np.inf, -np.inf, -np.inf], [.1, .2, .3, .4], [0.2, 0.4, -torch.inf, 0.35]],
        [[-torch.inf, -torch.inf, 0.91, -torch.inf], [.4, .3, .22, -torch.inf], [0.2, 0.4, -torch.inf, 0.35]]
    ]))
    expected = torch.where(expected == -torch.inf, expected, torch.log(expected))
    assert torch.all(logits_top_p == expected)


def test_beam_search():
    pass


def test_speculative_sampling():    
    # Test that the accepts are created correctly.
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)    
    target_probs = np.array([
        [0.1, 0.2, 0.3, 0.5, 0.8],
        [0.2, 0.4, 0.05, 0.95, 0.]
    ])
    draft_probs = np.array([
        [0.09, 0.25, 0.5, 0.9, 0.7],
        [0.1, 0.4, 0.01, 0.99, 0.5]
    ])
    # At this seed, random float is:
    random_uniform = np.array(
        [[0.4963, 0.7682, 0.0885, 0.1320, 0.3074],
       [0.6341, 0.4901, 0.8964, 0.4556, 0.6323]]
    )
    accepts = [
        [target_probs[i, j] / draft_probs[i, j] > random_uniform[i, j] for j in range(5)] for i in range(2)
    ]
    accepts = np.array(accepts)

    draft_probs = torch.from_numpy(draft_probs)
    target_probs = torch.from_numpy(target_probs)
    expected = torch.from_numpy(accepts)

    accepts = sampling._get_speculative_sampling_accepts(target_probs, draft_probs)
    assert torch.all(accepts == expected)

    # Test that the (target - draft)_+ distro is created correctly.
    target_probs = np.array([[[0.95, 0.05, 0., 0.], [0.1, 0.2, 0.3, 0.4], [0.2, 0.4, 0.05, 0.35]], 
                        [[0.0, 0.05, 0.91, 0.04], [0.4, 0.3, 0.22, 0.08], [0.2, 0.4, 0.05, 0.35]]])
    draft_probs = np.array([[[0.95, 0.05, 0., 0.], [0.2, 0.25, 0.2, 0.35], [0.2, 0.4, 0.05, 0.35]], 
                        [[0.0, 0.05, 0.91, 0.04], [0.4, 0.3, 0.22, 0.08], [0.05, 0.4, 0.15, 0.40]]])
    first_rejection = np.array([1, 2])
    target_at_rejection = np.array([[0.1, 0.2, 0.3, 0.4], [0.2, 0.4, 0.05, 0.35]])
    draft_at_rejection = np.array([[0.2, 0.25, 0.2, 0.35], [0.05, 0.4, 0.15, 0.40]])
    minus = np.maximum(target_at_rejection - draft_at_rejection, 0)
    minus = minus / minus.sum(axis=-1, keepdims=True)
    expected = torch.from_numpy(minus)

    target_probs = torch.from_numpy(target_probs)
    draft_probs = torch.from_numpy(draft_probs)
    first_rejection = torch.from_numpy(first_rejection)
    distro = sampling._get_speculative_sampling_target_minus_draft_distribution(draft_probs, target_probs, first_rejection)

    assert torch.all(distro == expected)


if __name__ == "__main__":
    print("Running tests for sampling.py")
    test_top_p()
    test_beam_search()
    test_speculative_sampling()
    print("All tests passed!")    
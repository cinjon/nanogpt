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
    pass


if __name__ == "__main__":
    print("Running tests for sampling.py")
    test_top_p()
    test_beam_search()
    test_speculative_sampling()
    print("All tests passed!")    
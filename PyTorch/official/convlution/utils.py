import torch

def create_pad_mask(t, pad):
    """
    Creates a padding mask.

    Parameters:
    - t (Tensor): The input tensor, typically the input sequence or target sequence.
    - pad (int): The index of the padding token.

    Returns:
    - mask (Tensor): A padding mask tensor with shape (batch_size, 1, seq_len),
                     where each position is a boolean indicating if the position is a padding token.
    """
    mask = (t == pad).unsqueeze(-2)
    return mask


def create_target_self_mask(target_len, device=None):
    """
    Creates a self-attention mask for the target sequence.

    Parameters:
    - target_len (int): The length of the target sequence.
    - device (torch.device, optional): The device on which the tensor is located (e.g., 'cpu' or 'cuda').

    Returns:
    - target_self_mask (Tensor): A self-attention mask tensor with shape (1, target_len, target_len),
                                 where the upper triangular part (excluding the diagonal) is True,
                                 indicating that these positions are masked.
    """
    # ones = torch.ones(target_len, target_len, dtype=torch.uint8, device=device)
    ones = torch.ones(target_len, target_len, dtype=torch.bool, device=device)
    # torch.triu返回输入张量的上三角部分, diagonal表示向右上偏移量
    target_self_mask = torch.triu(ones, diagonal=1).unsqueeze(0)
    return target_self_mask
"""
Loss functions
Adapted from https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def dpo_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    beta: float,
    reference_free: bool = False,
    normalize: bool = True,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.
        normalize: If True, normalize the loss by the number of examples in the batch.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios

    if normalize:
        losses = -F.logsigmoid(beta * logits)
    else:
        losses = -beta * logits
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = (
        beta * (policy_rejected_logps - reference_rejected_logps).detach()
    )

    return losses, chosen_rewards, rejected_rewards


def ipo_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    eta: float,
) -> torch.FloatTensor:
    """Compute the IPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        eta: Coefficient of the KL divergence.

    Returns:
        The losses tensor contains the IPO loss for each example in the batch.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    logits = pi_logratios - ref_logratios
    losses = torch.square(logits - (1 / 2 / eta))

    return losses


def nash_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    last_chosen_logps: torch.FloatTensor,
    last_rejected_logps: torch.FloatTensor,
    eta: float,
    tau_eta_ratio: float,
) -> torch.FloatTensor:
    """Compute the nash loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        last_chosen_logps: Log probabilities of the policy model for the chosen responses in the last iteration. Shape: (batch_size,)
        last_rejected_logps: Log probabilities of the policy model for the rejected responses in the last iteration. Shape: (batch_size,)
        eta: Coefficient of the KL divergence.
        tau: Coefficient of the KL divergence for the last iteration.

    Returns:
        The losses tensor contains the IPO loss for each example in the batch.
    """
    chosen_logratios = policy_chosen_logps - tau_eta_ratio * reference_chosen_logps - (1 - tau_eta_ratio) * last_chosen_logps
    rejected_logratios = policy_rejected_logps - tau_eta_ratio * reference_rejected_logps - (1 - tau_eta_ratio) * last_rejected_logps
    logits = chosen_logratios - rejected_logratios
    losses = torch.square(logits - (1 / 2 / eta))

    return losses
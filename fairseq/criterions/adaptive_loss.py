# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import torch.nn.functional as F
from fairseq import utils
from fairseq.logging import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.constants import DDP_BACKEND_CHOICES
from omegaconf import II
from typing import List

@dataclass
class AdaptiveLossConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")
    ddp_backend: DDP_BACKEND_CHOICES = II("distributed_training.ddp_backend")
    use_mean_reduction: bool = II("optimization.use_mean_reduction")
    update_freq: List[int] = II("optimization.update_freq")

@register_criterion("adaptive_loss", dataclass=AdaptiveLossConfig)
class AdaptiveLoss(FairseqCriterion):
    """This is an implementation of the loss function accompanying the adaptive softmax approximation for
    graphical processing units (GPU), described in the paper "Efficient softmax approximation for GPUs"
    (http://arxiv.org/abs/1609.04309)."""

    def __init__(self, task, sentence_avg, use_mean_reduction):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.use_mean_reduction = use_mean_reduction
        
        # TODO: Investigate a better way to toggle logging_outputs_can_be_summed()
        global adaptive_loss_use_mean_reduction
        adaptive_loss_use_mean_reduction = self.use_mean_reduction

    @classmethod
    def build_criterion(cls, cfg: AdaptiveLossConfig, task):
        if cfg.ddp_backend in {"c10d", "pytorch_ddp"}:
            raise Exception(
                "AdaptiveLoss is not compatible with the PyTorch "
                "version of DistributedDataParallel. Please use "
                "`--ddp-backend=legacy_ddp` instead."
            )
        
        # Using a mean reduction is incompatible with sentence-level averaging
        if cfg.use_mean_reduction and cfg.sentence_avg:
            raise Exception(
                "AdaptiveLoss cannot be used with both a mean reduction and "
                "sentence-level averaging. Please remove one of these options."
            )
        
        # Using a mean reduction is incompatible with gradient accumulation
        assert len(cfg.update_freq) == 1
        if cfg.update_freq[0] > 1:
            raise Exception(
                "AdaptiveLoss cannot be used with gradient accumulation. "
                "Please set `--update-freq=1` instead."
            )

        return cls(task, cfg.sentence_avg, cfg.use_mean_reduction)
    
    def forward(self, model, sample, reduce=True):
        """
        Compute the loss for the given sample. If reduce=True, a mean or sum reduction will be
        applied depending on the value of self.use_mean_reduction. 

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert (
            hasattr(model.decoder, "adaptive_softmax")
            and model.decoder.adaptive_softmax is not None
        )
        adaptive_softmax = model.decoder.adaptive_softmax

        net_output = model(**sample["net_input"])
        orig_target = model.get_targets(sample, net_output)

        nsentences = orig_target.size(0)
        orig_target = orig_target.view(-1)

        bsz = orig_target.size(0)

        logits, target = adaptive_softmax(net_output[0], orig_target)
        assert len(target) == len(logits)

        loss = net_output[0].new(1 if reduce else bsz).zero_()

        if self.use_mean_reduction:
            # target[0].size() --> batch_size, but target[1] and target[2] are smaller.
            # thus, we want to use a mean reduction for target[0], and divide a sum reduction
            # for target[1] and target[2] by the batch_size. 
            for i in range(len(target)):
                if target[i] is not None:
                    assert target[i].min() >= 0 and target[i].max() <= logits[i].size(1)

                    reduction = "none"
                    divisor = 1 if i == 0 else bsz
                    if reduce:
                        reduction = "mean" if i == 0 else "sum"

                    loss_i = F.cross_entropy(
                        logits[i],
                        target[i],
                        ignore_index=self.padding_idx,
                        reduction=reduction
                    )
                    loss += (loss_i / divisor)

            orig = utils.strip_pad(orig_target, self.padding_idx)

            ntokens = orig.numel()
            sample_size = 1  # gradient is already scaled by batch_size via reduction
            logging_output = {
                "loss": loss.data,
                "ntokens": ntokens,
                "nsentences": nsentences,
                "sample_size": sample_size
            }
            return loss, sample_size, logging_output
        else:
            for i in range(len(target)):
                if target[i] is not None:
                    assert target[i].min() >= 0 and target[i].max() <= logits[i].size(1)
                    loss += F.cross_entropy(
                        logits[i],
                        target[i],
                        ignore_index=self.padding_idx,
                        reduction="sum" if reduce else "none"
                    )
            
            orig = utils.strip_pad(orig_target, self.padding_idx)
            ntokens = orig.numel()
            sample_size = sample["target"].size(0) if self.sentence_avg else ntokens
            logging_output = {
                "loss": loss.data,
                "ntokens": ntokens,
                "nsentences": nsentences,
                "sample_size": sample_size,
            }
            return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        if adaptive_loss_use_mean_reduction:
            # ntokens tells how many tokens were in each batch, and we need to carefully combine
            # these into scaling factors that can combine our losses into something that can 
            # be stored in a fp16
            total_tokens = utils.item(
                sum(log.get("ntokens", 0) for log in logging_outputs)
            )
            loss_sum = utils.item(
                sum(log.get("loss", 0) * (log.get("ntokens", 0)/total_tokens) for log in logging_outputs)
            )
            
            metrics.log_scalar(
                "loss", loss_sum / math.log(2), 1, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )
        else:
            loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
            ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
            sample_size = utils.item(
                sum(log.get("sample_size", 0) for log in logging_outputs)
            )
            
            metrics.log_scalar(
                "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
            )

            if sample_size != ntokens:
                metrics.log_scalar(
                    "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
                )
                metrics.log_derived(
                    "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
                )
            else:
                metrics.log_derived(
                    "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improve distributed training speed.
        """
        return not adaptive_loss_use_mean_reduction

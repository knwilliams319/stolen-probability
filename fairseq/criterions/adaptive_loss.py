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

    @classmethod
    def build_criterion(cls, cfg: AdaptiveLossConfig, task):
        if cfg.ddp_backend in {"c10d", "pytorch_ddp"}:
            raise Exception(
                "AdaptiveLoss is not compatible with the PyTorch "
                "version of DistributedDataParallel. Please use "
                "`--ddp-backend=legacy_ddp` instead."
            )
    
        # Using a mean reduction with AdaptiveLoss may improve precision with fp16,
        # but doesn't give correct outputs when accumulating loss/gradients over more than
        # a single batch.
        # The mean reduction will be applied automatically if gradient accumulation is
        # disabled (update_freq == 1).
        assert len(cfg.update_freq) == 1
        return cls(task, cfg.sentence_avg, cfg.update_freq[0] == 1)
    
    def forward(self, model, sample, train_sample, reduce=True):
        """
        Compute the loss for the given sample. If the sample comes from the training set and
        self.use_mean_reduction is True, then the loss will be computed using a mean reduction.
        Otherwise, the loss will be computed using a sum reduction. 

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if train_sample and self.use_mean_reduction:
            return self._forward_with_mean(model, sample, reduce)
        else:
            return self._forward_with_sum(model, sample, reduce)
        
    def _forward_with_sum(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

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

    def _forward_with_mean(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

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

        # sum reduction is bad for fp16 if loss is huge. fairseq tries to use sum reduction
        # and scale gradients by batch_size later, which is correct when using options like
        # gradient accumulation.

        # instead, if we assume gradient accumulation will never be used, we can use a mean
        # reduction to fit the gradient inside a fp16 and remove the re-scaling that occurs
        # later.
        for i in range(len(target)):
            # target[0].size() --> batch_size, but target[1] and target[2] are smaller.
            # thus, we want to use a mean reduction for target[0], and divide a sum reduction
            # for target[1] and target[2] by the batch_size. 
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

        # set sample_size = 1 since we're using a mean reduction. 
        # set ntokens = 1 since we're using a mean reduction; we don't want to output nll_loss
        assert self.sentence_avg == False
        ntokens = 1
        sample_size = 1
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
        to True will improves distributed training speed.
        """
        return True
    
    def different_forward_for_train_test(self) -> bool:
        """
        Whether the `forward` method should be invoked differently for
        training vs. validation data. If True, a criterion's forward()
        method should accept a third positional warg train_sample which indicates
        if the passed sample came from the training set. This is False
        for all criterion except AdaptiveLoss. 

        For AdaptiveLoss, we will use a different forward() method if
        we're using a mean reduction over training passes. Validation
        passes should still use the sum reduction since the loss needs
        to be accumulated over multiple batches. 
        """
        return True

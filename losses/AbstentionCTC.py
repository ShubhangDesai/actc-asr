import torch
import torch.nn as nn
import numpy as np
from torch.multiprocessing import Pool

class AbstentionCTC(nn.Module):
    def __init__(self, blank=0, abstention=1):
        self.blank = blank
        self.abstention = abstention
        self.pool = Pool(8)

    def create_target_p(self, target):
        target_p = [self.blank]
        for c in target:
            target_p.append(c)
            target_p.append(self.blank)

        return target_p

    def empty_dp(self, T, target_p):
        return [[torch.tensor(-1.0).float() for _ in range(len(target_p))] for _ in range(T)]

    def compute_probs_dp(self, log_prob, dp, target_p):
        ninf = torch.log(torch.from_numpy(np.array(0.0)))
        for t in range(len(dp)):
            for s in range(len(dp[0])):
                if t == 0:
                    if s == 0:
                        a = log_prob[0, self.blank]
                        b = log_prob[0, self.abstention]
                        res = max(a, b) + torch.log(1 + torch.exp(-torch.abs(a-b)))
                    elif s == 1:
                        res = log_prob[0, target_p[1]]
                    else:
                        res = ninf

                    dp[t][s] = res
                    continue


                prob1, prob2 = log_prob[t, target_p[s]], log_prob[t, self.abstention]
                if prob1 == -float("Inf") and prob2 == -float("Inf"): prob = ninf
                else: prob = max(prob1, prob2) + torch.log(1 + torch.exp(-torch.abs(prob1-prob2)))

                a = (prob + dp[t-1][s]).float()
                b = ((dp[t-1][s-1] if s-1 >= 0 else ninf) + log_prob[t, target_p[s]]).float()

            	# in case of a blank or a repeated label, we only consider s and s-1 at t-1, so we're done
                if target_p[s] == self.blank or (s >= 2 and target_p[s-2] == target_p[s]):
                    if a == -float("Inf") and b == -float("Inf"): res = ninf
                    else: res = max(a, b) + torch.log(1 + torch.exp(-torch.abs(a-b)))

                    dp[t][s] = res
                    continue

            	# otherwise, in case of a non-blank and non-repeated label, we additionally add s-2 at t-1
                c = ((dp[t-1][s-2] if s-2 >= 0 else ninf) + log_prob[t, target_p[s]]).float()

                m = max([a, b, c])
                if a == -float("Inf") and b == -float("Inf") and c == -float("Inf"): res = ninf
                else: res = m + torch.log(torch.exp(-torch.abs(a-m)) + torch.exp(-torch.abs(b-m)) + torch.exp(-torch.abs(c-m)))

                dp[t][s] = res

        return dp

    def compute_instance_loss(self, inps):
        log_prob, target = inps

        T = log_prob.shape[0]
        target_p = self.create_target_p(target)
        dp = self.empty_dp(T, target_p)

        log_prob.requires_grad = True
        print('here')
        dp = self.compute_probs_dp(log_prob, dp, target_p)
        print('here2')
        a, b = dp[T-1][len(target_p)-1], dp[T-1][len(target_p)-2]

        loss = max(a, b) + torch.log(1 + torch.exp(-torch.abs(a-b)))

        loss.backward()
        print('here3')
        return log_prob.grad

    def compute_loss(self, log_probs, targets, input_lengths, target_lengths):
        grads = self.pool.map(self.compute_instance_loss, [(log_probs[:input_lengths[i], i, :].detach(), targets[i][:target_lengths[i]]) for i in range(log_probs.shape[1])])
        print('here4')
        dlog_probs = torch.stack(grads).permute((1, 0, 2))
        loss = (dlog_probs * log_probs).sum() / len(grads) / -4

        return loss

    def __call__(self, log_probs, targets, input_lengths, target_lengths, w=None):
        return self.compute_loss(log_probs, targets, input_lengths, target_lengths)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

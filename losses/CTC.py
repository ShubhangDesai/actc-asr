import torch
import torch.nn as nn
import numpy as np
from torch.multiprocessing import Pool

class CTC(nn.Module):
    def __init__(self, blank=0):
        self.blank = blank
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
                        res = log_prob[0, self.blank]
                    elif s == 1:
                        res = log_prob[0, target_p[1]]
                    else:
                        res = ninf

                    dp[t][s] = res
                    continue


                a = dp[t-1][s].float()
                b = dp[t-1][s-1].float() if s-1 >= 0 else ninf

            	# in case of a blank or a repeated label, we only consider s and s-1 at t-1, so we're done
                if target_p[s] == self.blank or (s >= 2 and target_p[s-2] == target_p[s]):
                    if a == ninf and b == ninf: res = ninf
                    else: res = max(a, b) + torch.log(1 + torch.exp(-torch.abs(a-b))) + log_prob[t, target_p[s]]

                    dp[t][s] = res
                    continue

            	# otherwise, in case of a non-blank and non-repeated label, we additionally add s-2 at t-1
                c = dp[t-1][s-2] if s-2 >= 0 else ninf

                m = max([a, b, c])
                if a == ninf and b == ninf and c == ninf: res = ninf
                else: res = m + torch.log(torch.exp(-torch.abs(a-m)) + torch.exp(-torch.abs(b-m)) + torch.exp(-torch.abs(c-m))) + log_prob[t, target_p[s]]

                dp[t][s] = res

        return dp

    def compute_instance_loss(self, inps):
        log_prob, target = inps

        T = log_prob.shape[0]
        target_p = self.create_target_p(target)
        dp = self.empty_dp(T, target_p)

        log_prob.requires_grad = True
        dp = self.compute_probs_dp(log_prob, dp, target_p)
        a, b = dp[T-1][len(target_p)-1], dp[T-1][len(target_p)-2]

        loss = max(a, b) + torch.log(1 + torch.exp(-torch.abs(a-b)))

        loss.backward()
        return log_prob.grad

    def compute_loss(self, log_probs, targets, input_lengths, target_lengths):
        grads = self.pool.map(self.compute_instance_loss, [(log_probs[:input_lengths[i], i, :].detach(), targets[i][:target_lengths[i]]) for i in range(log_probs.shape[1])])
        dlog_probs = torch.stack(grads).permute((1, 0, 2))
        loss = (dlog_probs * log_probs).sum() / len(grads) / -4

        return loss

    def __call__(self, log_probs, targets, input_lengths, target_lengths, forget_rate, w=None):
        return self.compute_loss(log_probs, targets, input_lengths, target_lengths)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

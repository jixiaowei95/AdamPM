import torch
from torch.optim import Optimizer



class AdamPM(Optimizer):
    def __init__(self, params, lr=0.0003, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, group_size=4):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, group_size=group_size)
        super().__init__(params, defaults)
        print(f"\n AdamPM  : lr = {lr} , betas = {betas} , weight_decay = {weight_decay} , group_size = {group_size}\n")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, (beta1, beta2), eps = group['lr'], group['betas'], group['eps']
            group_size = group['group_size']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None: continue

                grad = p.grad
                state = self.state[p]

                grad_2d = grad.reshape(-1, grad.size(-1))
                cols = p.shape[-1]
                rows = p.numel() // cols
                num_groups = (cols + group_size - 1) // group_size

                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p)

                    state['v'] = torch.zeros(rows, num_groups, device=p.device, dtype=p.dtype)

                m, v = state['m'], state['v']
                t = state['step'] + 1

                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                grad_sq = grad_2d ** 2
                full_groups = cols // group_size

                if full_groups == num_groups:
                    grad_group_means = grad_sq.reshape(rows, num_groups, group_size).mean(dim=2)
                else:
                    full_means = grad_sq[:, :full_groups * group_size].reshape(rows, full_groups, group_size).mean(dim=2)
                    last_mean = grad_sq[:, full_groups * group_size:].mean(dim=1, keepdim=True)
                    grad_group_means = torch.cat([full_means, last_mean], dim=1)

                v.mul_(beta2).add_(grad_group_means, alpha=1 - beta2)

                del grad_sq, grad_group_means
                if full_groups != num_groups:
                    del full_means, last_mean

                v_hat = v / (1 - beta2 ** t)

                v_expanded = v_hat.repeat_interleave(group_size, dim=1)
                v_expanded = v_expanded[:, :cols]

                v_expanded = v_expanded.reshape_as(p)

                v_expanded.sqrt_()
                v_expanded.add_(eps)

                step_size = lr / (1 - beta1 ** t)

                if weight_decay != 0:
                    print(f"weight_decay = {weight_decay}")
                    p.mul_(1 - lr * weight_decay)

                p.addcdiv_(m, v_expanded, value=-step_size)

                state['step'] = t

        return loss




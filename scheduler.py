from math import cos, pi, tanh


def anneal_linear(start, end, proportion):
    return start + proportion * (end - start)


def anneal_cos(start, end, proportion):
    cos_val = cos(pi * proportion) + 1

    return end + (start - end) / 2 * cos_val


def anneal_cospow(start, end, proportion):
    power = 5

    cos_val = 0.5 * (cos(pi * proportion) + 1) + 1
    cos_val = power ** cos_val - power
    cos_val = cos_val / (power ** 2 - power)

    return end + (start - end) * cos_val


def anneal_poly(start, end, proportion, power=0.9):
    return (start - end) * (1 - proportion) ** power + end


def anneal_tanh(start, end, proportion, lower=-6, upper=3):
    return end + (start - end) / 2 * (1 - tanh(lower + (upper - lower) * proportion))


class Phase:
    def __init__(self, start, end, n_iter, anneal_fn):
        self.start, self.end = start, end
        self.n_iter = n_iter
        self.anneal_fn = anneal_fn
        self.n = 0

    def step(self):
        self.n += 1

        return self.anneal_fn(self.start, self.end, self.n / self.n_iter)

    def reset(self):
        self.n = 0

    @property
    def is_done(self):
        return self.n >= self.n_iter


class CycleScheduler:
    def __init__(
        self,
        optimizer,
        lr_max,
        n_iter,
        divider=25,
        warmup_proportion=0.3,
        phase=('linear', 'cos'),
    ):
        self.optimizer = optimizer

        phase1 = int(n_iter * warmup_proportion)
        phase2 = n_iter - phase1
        lr_min = lr_max / divider

        phase_map = {
            'linear': anneal_linear,
            'cos': anneal_cos,
            'cospow': anneal_cospow,
            'poly': anneal_poly,
            'tanh': anneal_tanh,
        }

        self.lr_phase = [
            Phase(lr_min, lr_max, phase1, phase_map[phase[0]]),
            Phase(lr_max, lr_min / 1e4, phase2, phase_map[phase[1]]),
        ]

        self.phase = 0

    def step(self):
        lr = self.lr_phase[self.phase].step()

        for group in self.optimizer.param_groups:
            group['lr'] = lr

        if self.lr_phase[self.phase].is_done:
            self.phase += 1

        if self.phase >= len(self.lr_phase):
            for phase in self.lr_phase:
                phase.reset()

            self.phase = 0

        return lr

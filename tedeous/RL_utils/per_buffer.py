import torch
from collections import deque, namedtuple

Transition = namedtuple('Transition',
                        ('state', 'next_state', 'action', 'reward', 'done', 'model_reward', 'opt_model_i'))
    

# ---------- Prioritized Experience Replay (proportional) ----------
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, eps=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.memory, self.prior, self.pos = [], [], 0

    def __len__(self):
        return len(self.memory)

    def push(self, *args, priority=None):
        # args: (state, next_state, action, reward, done, model_reward, opt_model_i)
        tr = Transition(*args)
        if priority is None:
            p = max(self.prior) if self.prior else 1.0
        else:
            p = float(priority)

        if len(self.memory) < self.capacity:
            self.memory.append(tr); self.prior.append(p)
        else:
            self.memory[self.pos] = tr; self.prior[self.pos] = p
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4, device='cpu'):
        pr = torch.tensor(self.prior, dtype=torch.float, device=device)
        probs = (pr + self.eps) ** self.alpha
        probs = probs / probs.sum()

        idxs = torch.multinomial(
            probs, batch_size,
            replacement=len(self.memory) < batch_size
        )

        # w_j = (N * P(j))^{-β} / max_i w_i
        N = len(self.memory)
        weights = (N * probs[idxs]).pow(-beta)
        weights = (weights / weights.max()).float()

        batch = [self.memory[int(i)] for i in idxs]
        return batch, idxs, weights

    def sample_uniform(self, batch_size, device='cpu'):
        """
        Равномерная выборка без смещения (для warmup). Веса = 1.
        """
        N = len(self.memory)
        if N == 0:
            raise RuntimeError("Buffer is empty")
        idxs = torch.randint(0, N, (batch_size,), device=device)
        batch = [self.memory[int(i)] for i in idxs]
        weights = torch.ones(batch_size, dtype=torch.float, device=device)
        return batch, idxs, weights

    def update_priorities(self, idxs, new_p):
        for i, p in zip(idxs.tolist(), new_p.tolist()):
            self.prior[int(i)] = float(max(p, self.eps))
    
    # per_buffer.py  --- ДОБАВИТЬ внутрь класса PrioritizedReplayBuffer
    def _build_sequence_from_start(self, start_idx: int, L: int):
        """
        Собираем переходы [start_idx .. start_idx+L-1], обрываем на done
        и не даём вылезти за конец буфера. Без циклического wrap-around.
        """
        seq = []
        i = start_idx
        N = len(self.memory)
        steps = 0
        while i < N and steps < L:
            tr = self.memory[i]
            seq.append(tr)
            steps += 1
            # если эпизод закончился — выходим (не включаем следующий)
            if getattr(tr, "done", 0) != 0:
                break
            i += 1
        return seq

    def sample_sequences(self, batch_size: int, L: int, beta=None, uniform=False, device='cpu'):
        """
        Возвращает:
        - seqs: list[list[Transition]] длиной B, каждая — последовательность длиной ≤L,
        - idxs: Tensor[B] стартовых индексов (их и обновляем в update_priorities),
        - is_w: Tensor[B] importance-sampling веса.
        """
        N = len(self.memory)
        if N == 0:
            raise RuntimeError("Buffer is empty")

        # --- выбор стартовых индексов ---
        if uniform:
            # равномерно
            idxs = torch.randint(0, N, (batch_size,), device=device)
            is_w = torch.ones(batch_size, dtype=torch.float, device=device)
        else:
            # PER по стартовым элементам
            pr = torch.tensor(self.prior, dtype=torch.float, device=device)
            probs = (pr + self.eps) ** self.alpha
            probs = probs / probs.sum()

            replacement = N < batch_size
            idxs = torch.multinomial(probs, batch_size, replacement=replacement)

            # IS-веса по стартовой точке (классика PER)
            assert beta is not None, "beta must be provided for PER sampling"
            weights = (N * probs[idxs]).pow(-beta)
            is_w = (weights / weights.max()).float()

        # --- сбор последовательностей ---
        seqs = [ self._build_sequence_from_start(int(i), L) for i in idxs.tolist() ]

        return seqs, idxs, is_w


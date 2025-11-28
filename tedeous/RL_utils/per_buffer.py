import torch
import random
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'next_state', 'action', 'reward', 'done', 'model_reward', 'opt_model_i'))
    

# ---------- Prioritized Experience Replay (proportional) ----------
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, eps=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.memory, self.prior, self.pos = [], [], 0

        # --- success replay ---
        # Порог по model_reward, выше которого терминальный переход считаем "успехом"
        self.success_threshold = 0.0
        # Набор индексов переходов, которые являются успешными терминалами (done == 1)
        self.success_indexes = set()

    def __len__(self):
        return len(self.memory)

    def push(self, *args, priority=None):
        """
        args: (state, next_state, action, reward, done, model_reward, opt_model_i)
        """
        tr = Transition(*args)

        # --- базовый приоритет ---
        if priority is None:
            p = max(self.prior) if self.prior else 1.0
        else:
            p = float(priority)

        # --- индекс, в который пишем ---
        if len(self.memory) < self.capacity:
            idx = len(self.memory)
            self.memory.append(tr)
            self.prior.append(p)
        else:
            idx = self.pos
            self.memory[idx] = tr
            self.prior[idx] = p
            self.pos = (self.pos + 1) % self.capacity

        # --- обновляем success_indexes для этого индекса ---
        # считаем успешным терминалом: done == 1 и model_reward > success_threshold
        done = getattr(tr, "done", 0)
        model_reward = float(getattr(tr, "model_reward", 0.0))

        is_success = (done == 1) and (model_reward > self.success_threshold)

        if is_success:
            self.success_indexes.add(idx)
        else:
            # если в этом слоте раньше был успех, а теперь нет — убираем
            self.success_indexes.discard(idx)


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
    
    def _build_sequence_ending_at(self, end_idx: int, L: int):
        """
        Строим последовательность длиной ≤ L, которая заканчивается переходом end_idx.
        Идём НАЗАД по буферу, пока не встретим начало эпизода (предыдущий done!=0)
        или не наберём L шагов. Без циклического wrap-around.
        """
        seq_rev = []
        i = end_idx
        steps = 0
        while i >= 0 and steps < L:
            tr = self.memory[i]
            seq_rev.append(tr)
            steps += 1
            # если это НЕ последний элемент (т.е. не end_idx) и done!=0,
            # значит, начался предыдущий эпизод — останавливаемся
            if steps > 1 and getattr(tr, "done", 0) != 0:
                break
            i -= 1

        seq_rev.reverse()
        return seq_rev


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
            # равномерный сэмплинг по НЕтерминальным start_idx
            valid_idxs = []
            tries = 0
            max_tries = batch_size * 10  # чтобы не зациклиться, если терминалов много

            while len(valid_idxs) < batch_size and tries < max_tries:
                cand = torch.randint(0, N, (1,), device=device).item()
                tr = self.memory[cand]
                # Берём в качестве старта только НЕтерминальный переход
                if getattr(tr, "done", 0) == 0:
                    valid_idxs.append(cand)
                tries += 1

            # если по какой-то причине не набрали полный батч — добьём чем есть (включая терминалы)
            if len(valid_idxs) < batch_size:
                extra = torch.randint(0, N, (batch_size - len(valid_idxs),), device=device).tolist()
                valid_idxs.extend(extra)

            idxs = torch.tensor(valid_idxs, dtype=torch.long, device=device)
            is_w = torch.ones(batch_size, dtype=torch.float, device=device)

        else:
            # PER по стартовым элементам, но тоже избегаем start_idx с done=1
            pr = torch.tensor(self.prior, dtype=torch.float, device=device)
            probs = (pr + self.eps) ** self.alpha
            probs = probs / probs.sum()

            valid_idxs = []
            tries = 0
            max_tries = batch_size * 20

            while len(valid_idxs) < batch_size and tries < max_tries:
                # сэмплим единичный индекс по распределению probs
                cand = torch.multinomial(probs, 1, replacement=True).item()
                tr = self.memory[cand]
                if getattr(tr, "done", 0) == 0:
                    valid_idxs.append(cand)
                tries += 1

            # если не набрали полный батч — добираем обычным PER с возможными терминалами
            if len(valid_idxs) < batch_size:
                remaining = batch_size - len(valid_idxs)
                replacement = N < remaining
                extra = torch.multinomial(probs, remaining, replacement=replacement).tolist()
                valid_idxs.extend(extra)

            idxs = torch.tensor(valid_idxs, dtype=torch.long, device=device)

            # IS-веса по стартовой точке (классика PER)
            assert beta is not None, "beta must be provided for PER sampling"
            weights = (N * probs[idxs]).pow(-beta)
            is_w = (weights / weights.max()).float()

        # --- сбор последовательностей ---
        # здесь уже без изменений: для каждого стартового индекса строим последовательность вперёд
        seqs = [self._build_sequence_from_start(int(i), L) for i in idxs.tolist()]

        return seqs, idxs, is_w

    
    def sample_success_sequences(self, batch_size: int, L: int, device='cpu'):
        """
        Сэмплирует batch_size последовательностей длиной ≤L,
        КАЖДАЯ из которых заканчивается успешным терминальным переходом
        (индекс в self.success_indexes).

        Возвращает:
          - seqs: list[list[Transition]]
          - idxs: Tensor[batch_size] стартовых индексов последовательностей
          - is_w: Tensor[batch_size] весов (здесь просто 1.0)
        """
        if not self.success_indexes:
            # если пока нет ни одного успеха — просто fallback на uniform sequences
            seqs, idxs, is_w = self.sample_sequences(batch_size, L, beta=None, uniform=True, device=device)
            return seqs, idxs, is_w

        success_list = list(self.success_indexes)
        N_succ = len(success_list)

        seqs = []
        idxs = []

        for _ in range(batch_size):
            end_idx = success_list[random.randint(0, N_succ - 1)]
            seq = self._build_sequence_ending_at(end_idx, L)

            # на всякий случай: если получилось пусто — fallback к стартовому варианту
            if not seq:
                start_idx = max(end_idx - L + 1, 0)
                seq = self._build_sequence_from_start(start_idx, L)

            seqs.append(seq)

            # стартовый индекс последовательности — индекс первого элемента
            # (его и будем использовать для обновления приоритетов)
            start_idx = end_idx - (len(seq) - 1)
            start_idx = max(start_idx, 0)
            idxs.append(start_idx)

        idxs = torch.tensor(idxs, dtype=torch.long, device=device)
        is_w = torch.ones(batch_size, dtype=torch.float, device=device)

        return seqs, idxs, is_w



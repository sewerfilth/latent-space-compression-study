import os
import struct
import zlib
import time
import json
from datetime import datetime
import sys
try:
    import psutil
except ImportError:
    psutil = None
import torch
import torch.nn as nn
import torch.nn.functional as F
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QLabel, QFileDialog, QProgressBar, QHBoxLayout,
    QGroupBox, QFormLayout, QComboBox, QTextEdit,
    QTabWidget, QMessageBox, QScrollArea, QLineEdit,
    QSlider, QInputDialog
)

PRESET_FILE = "pico_presets.json"

# ==========================================
# 0. GLOBALS / LSC-1 PARAMETERS
# ==========================================
LATENT_DIM = 12
CODEBOOK_SIZE = 8192

DEFAULT_SYNC_WINDOW_CHUNKS = 64
DEFAULT_RESIDUAL_Q = 127.0
DEFAULT_ROT_STRENGTH = 1.0

NORM_OFFSET = 128.0
NORM_SCALE = 50.0

SYNC_WINDOW_CHUNKS = DEFAULT_SYNC_WINDOW_CHUNKS
RESIDUAL_Q = DEFAULT_RESIDUAL_Q
ROT_STRENGTH = DEFAULT_ROT_STRENGTH

AR_LAMBDA = 0.5
STAB_LAMBDA = 0.25
SEQ_LEN = 8

LSC1_MAGIC = 0x4C534331  # "LSC1"


# ==========================================
# 1. CORE ARCHITECTURE (LSC-1 + GENETIC LATENTS)
# ==========================================
class SphericalConvCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.push = nn.Sequential(
            nn.Conv1d(1, 16, 4, 4), nn.ReLU(),
            nn.Conv1d(16, 32, 4, 4), nn.ReLU(),
            nn.Conv1d(32, LATENT_DIM, 4, 4)
        )

        self.male_head = nn.Linear(LATENT_DIM, LATENT_DIM)
        self.female_head = nn.Linear(LATENT_DIM, LATENT_DIM)

        torch.manual_seed(42)
        initial_cb = F.normalize(
            torch.randn(CODEBOOK_SIZE, LATENT_DIM),
            p=2, dim=1
        )
        self.codebook = nn.Parameter(initial_cb)

        # Deconstruct pop for functional mutation
        self.dec1 = nn.ConvTranspose1d(LATENT_DIM, 32, 4, 4)
        self.dec2 = nn.ConvTranspose1d(32, 16, 4, 4)
        self.dec3 = nn.ConvTranspose1d(16, 1, 4, 4)

    def _apply_kernel_mutation(self, weight, prime, device):
        R = rotation_from_prime(prime, LATENT_DIM, device, ROT_STRENGTH)
        # weight shape: [in_channels (12), out_channels (32), kernel_size (4)]
        # Rotate the basis vectors of the input projection
        return torch.einsum('ij,jkl->ikl', R, weight)

    def _nearest_codebook_indices(self, flat_latent, chunk_size=4096):
        device = flat_latent.device
        codebook = self.codebook.to(device)
        all_indices = []
        N = flat_latent.size(0)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            sub = flat_latent[start:end]
            dist = torch.cdist(sub, codebook)
            idx = torch.argmin(dist, dim=1)
            all_indices.append(idx)

        return torch.cat(all_indices, dim=0)

    def _encode_continuous(self, x):
        cont = self.push(x)
        cont = F.normalize(cont, p=2, dim=1)
        return cont  # [B, 12, 16]

    def _split_male_female(self, cont):
        seq = cont.transpose(1, 2)  # [B, 16, 12]
        B, T, D = seq.shape
        flat = seq.reshape(-1, D)

        m_flat = self.male_head(flat)
        f_flat = self.female_head(flat)

        m_flat = F.normalize(m_flat, p=2, dim=1)
        f_flat = F.normalize(f_flat, p=2, dim=1)

        m = m_flat.view(B, T, D)
        f = f_flat.view(B, T, D)
        return m, f  # [B, 16, 12]

    def forward_train(self, x, primes=None):
        cont = self._encode_continuous(x)          # [B, 12, 16]
        m, f = self._split_male_female(cont)       # [B, 16, 12]

        flat_m = m.reshape(-1, LATENT_DIM)
        idx = self._nearest_codebook_indices(flat_m)
        quant_m = self.codebook[idx].reshape(-1, 16, LATENT_DIM)

        quant_m_ste = m + (quant_m - m).detach()
        quant_m_ste_ch = quant_m_ste.transpose(1, 2)  # [B, 12, 16]
        
        # Reconstruct with mutating kernels if primes provided
        recons = []
        for i in range(quant_m_ste_ch.size(0)):
            chunk_latent = quant_m_ste_ch[i:i+1]
            p = primes[i] if primes is not None else None
            recons.append(self._decode_core(chunk_latent, p))
        
        recon = torch.cat(recons, dim=0)

        return recon, cont, quant_m_ste, m, f

    @torch.no_grad()
    def encode_to_latents(self, x):
        cont = self._encode_continuous(x)
        m, f = self._split_male_female(cont)
        return m.reshape(-1, LATENT_DIM), f.reshape(-1, LATENT_DIM)

    @torch.no_grad()
    def encode_to_indices(self, x):
        cont = self._encode_continuous(x)
        m, _ = self._split_male_female(cont)
        flat_m = m.reshape(-1, LATENT_DIM)
        return self._nearest_codebook_indices(flat_m)

    def _decode_core(self, quant_ch, prime=None):
        w1 = self.dec1.weight
        if prime is not None:
            w1 = self._apply_kernel_mutation(w1, prime, quant_ch.device)
        
        x = F.conv_transpose1d(quant_ch, w1, self.dec1.bias, stride=4, padding=0)
        x = F.relu(x)
        x = self.dec2(x)
        x = F.relu(x)
        x = self.dec3(x)
        return x

    @torch.no_grad()
    def decode_from_indices(self, indices, prime=None):
        quant = self.codebook[indices].reshape(-1, 16, LATENT_DIM).transpose(1, 2)
        x = self._decode_core(quant, prime)
        return (
            (x * NORM_SCALE + NORM_OFFSET)
            .clamp(0, 255)
            .to(torch.uint8)
            .squeeze()
            .cpu()
            .numpy()
            .tobytes()
        )

    @torch.no_grad()
    def decode_from_latent(self, latent_m, prime=None):
        # Ensure we have [B, LATENT_DIM, 16] for the decoder
        if latent_m.dim() == 2:
            quant = latent_m.unsqueeze(-1).repeat(1, 1, 16)
        else:
            quant = latent_m.reshape(-1, 16, LATENT_DIM).transpose(1, 2)
            
        x = self._decode_core(quant, prime)
        return (
            (x * NORM_SCALE + NORM_OFFSET)
            .clamp(0, 255)
            .to(torch.uint8)
            .squeeze()
            .cpu()
            .numpy()
            .tobytes()
        )


# ==========================================
# 1.5 PRIMES + ROTATIONS + GENETIC AR
# ==========================================
def generate_primes(n, existing_primes=None):
    primes = existing_primes if existing_primes else []
    candidate = primes[-1] + 1 if primes else 2
    while len(primes) < n:
        is_prime = True
        for p in primes:
            if p * p > candidate:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return primes
class PrimeManager:
    def __init__(self, initial=20000, growth_factor=2.0):
        self.primes = generate_primes(initial)
        self.growth_factor = growth_factor

    def get(self, index):
        if index >= len(self.primes):
            new_target = int(len(self.primes) * self.growth_factor)
            if new_target <= len(self.primes):
                new_target = len(self.primes) + 1000
            self.primes = generate_primes(new_target, self.primes)
        return self.primes[index]


PRIME_MANAGER = PrimeManager()


def givens_rotation_matrix(dim, i, j, theta, device):
    R = torch.eye(dim, device=device)
    c = torch.cos(theta)
    s = torch.sin(theta)
    R[i, i] = c
    R[j, j] = c
    R[i, j] = -s
    R[j, i] = s
    return R


def rotation_from_prime(prime, dim, device, rot_strength=1.0):
    cpu = torch.device("cpu")
    torch.manual_seed(prime)
    R = torch.eye(dim, device=cpu)
    for _ in range(6):
        i = torch.randint(0, dim, (1,), device=cpu).item()
        j = torch.randint(0, dim, (1,), device=cpu).item()
        while j == i:
            j = torch.randint(0, dim, (1,), device=cpu).item()
        theta = torch.rand(1, device=cpu) * 2 * torch.pi * rot_strength
        G = givens_rotation_matrix(dim, i, j, theta, cpu)
        R = G @ R
    return R.to(device)


def apply_rotation_to_latent(latent, prime, device, rot_strength=1.0):
    R = rotation_from_prime(prime, LATENT_DIM, device, rot_strength)
    return latent @ R.T


def genetic_combine(m_latent, f_latent, alpha=0.7):
    return alpha * m_latent + (1.0 - alpha) * f_latent


def predict_next_latent(v_prev, prime, device, rot_strength=1.0):
    """
    Implements the core LSC-1 transition: V_{t+1} = R(p_t, pi) * V_t
    This collapses the latent space into a predictable topological trajectory.
    """
    if v_prev is None:
        return torch.zeros((1, LATENT_DIM), device=device)
    
    # Apply the deterministic orthogonal rotation
    v_next_pred = apply_rotation_to_latent(v_prev, prime, device, rot_strength)
    
    # Ensure the prediction stays on the S^11 hypersphere
    return F.normalize(v_next_pred, p=2, dim=-1)


# ==========================================
# 1.6 SIMPLE rANS WRAPPER (zlib backend)
# ==========================================
def rans_encode(data: bytes) -> bytes:
    return zlib.compress(data, level=9)


def rans_decode(data: bytes) -> bytes:
    return zlib.decompress(data)
# ==========================================
# 1.8 UI COMPONENTS
# ==========================================
class DriftGraph(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(150)
        self.data = []
        self.max_points = 200

    def add_value(self, val):
        self.data.append(val)
        if len(self.data) > self.max_points:
            self.data.pop(0)
        self.update()

    def clear(self):
        self.data = []
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#010409"))

        if not self.data:
            return

        w, h = self.width(), self.height()
        painter.setPen(QPen(QColor("#444"), 1, Qt.DashLine))
        painter.drawLine(0, int(h * 0.6), w, int(h * 0.6))  # 20.0 Drift Threshold ref

        painter.setPen(QPen(QColor("#58a6ff"), 2))
        step = w / (self.max_points - 1)
        for i in range(len(self.data) - 1):
            x1, y1 = i * step, h - (self.data[i] * (h / 50.0))
            x2, y2 = (i + 1) * step, h - (self.data[i+1] * (h / 50.0))
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))


# ==========================================
# 2. TRAINING WORKER (AR + GENETIC LOSS)
# ==========================================
class TrainingWorker(QThread):
    progress = pyqtSignal(int)
    log_stamp = pyqtSignal(str)
    finished = pyqtSignal()
    stopped = pyqtSignal()

    def __init__(self, codec, filepath, config):
        super().__init__()
        self.codec = codec
        self.filepath = filepath
        self.cfg = config
        self._pause = False
        self._stop = False

    def pause(self):
        self._pause = True

    def resume(self):
        self._pause = False

    def stop(self):
        self._stop = True

    def _wait_if_paused(self):
        while self._pause and not self._stop:
            self.msleep(100)

    def _read_sequence(self, f, device, seq_len=SEQ_LEN, chunk_size=1024):
        seq = []
        for _ in range(seq_len):
            chunk = f.read(chunk_size)
            if not chunk:
                break
            padded = chunk + b'\x00' * (chunk_size - len(chunk))
            t = (torch.tensor(list(padded), dtype=torch.float32, device=device) - NORM_OFFSET) / NORM_SCALE
            seq.append(t.view(1, 1, chunk_size))
        if not seq:
            return None
        return torch.cat(seq, dim=0)  # [L, 1, 1024]

    def run(self):
        torch.set_num_threads(self.cfg['threads'])
        device = self.cfg['device']
        self.codec.to(device)
        self.codec.train()
        process = psutil.Process(os.getpid()) if psutil else None

        opt = torch.optim.Adam(self.codec.parameters(), lr=self.cfg['lr'])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=3
        )

        passes_per_epoch = max(1, self.cfg.get('passes', 2))
        self.log_stamp.emit(f"--- ROUND START: {datetime.now().strftime('%H:%M:%S')} ---")

        for epoch in range(self.cfg['epochs']):
            if self._stop:
                self.stopped.emit()
                return

            total_loss = 0.0
            count = 0

            for _ in range(passes_per_epoch):
                if self._stop:
                    self.stopped.emit()
                    return

                with open(self.filepath, 'rb') as f:
                    while True:
                        self._wait_if_paused()
                        if self._stop:
                            self.stopped.emit()
                            return

                        seq = self._read_sequence(f, device, seq_len=SEQ_LEN)
                        if seq is None:
                            break

                        if process:
                            mem_gb = process.memory_info().rss / (1024 ** 3)
                            if mem_gb > self.cfg.get('ram_limit_gb', 999):
                                self.log_stamp.emit(f"RAM Limit Exceeded: {mem_gb:.2f}GB. Stopping.")
                                self.stopped.emit()
                                return

                        batch = seq  # [L, 1, 1024]
                        L = batch.size(0)
                        opt.zero_grad()

                        # Get primes for kernel mutation
                        primes = [PRIME_MANAGER.get(count * SEQ_LEN + i) for i in range(L)]

                        recon, cont, quant_m, m, f_lat = self.codec.forward_train(batch, primes)

                        recon_loss = F.mse_loss(recon, batch)
                        commit_loss = self.cfg['commit'] * F.mse_loss(
                            cont, quant_m.transpose(1, 2).detach()
                        )

                        m_flat = m.reshape(L, -1, LATENT_DIM).mean(dim=1)      # [L, 12]
                        f_flat = f_lat.reshape(L, -1, LATENT_DIM).mean(dim=1)  # [L, 12]

                        ar_loss = torch.tensor(0.0, device=device)
                        stab_loss = torch.tensor(0.0, device=device)

                        prev_m = prev_f = prev2_m = prev2_f = None

                        for t in range(L):
                            m_t = m_flat[t:t+1]
                            f_t = f_flat[t:t+1]

                            if prev_m is not None:
                                # Force the model to learn the prime-based rotation path
                                pred_m = predict_next_latent(
                                    prev_m, PRIME_MANAGER.get(t), device, ROT_STRENGTH
                                )
                                ar_loss = ar_loss + F.mse_loss(m_t, pred_m)
                                # Stab loss ensures the "Female" features evolve slowly
                                stab_loss = stab_loss + F.mse_loss(f_t, prev_f)

                            prev_m = m_t.detach()
                            prev_f = f_t.detach()

                        if L > 1:
                            ar_loss = ar_loss / (L - 1)
                            stab_loss = stab_loss / (L - 1)

                        loss = recon_loss + commit_loss + AR_LAMBDA * ar_loss + STAB_LAMBDA * stab_loss
                        loss.backward()
                        opt.step()

                        with torch.no_grad():
                            self.codec.codebook.div_(
                                torch.norm(self.codec.codebook, dim=1, keepdim=True)
                            )

                        total_loss += loss.item()
                        count += 1

            avg_loss = total_loss / max(1, count)
            sched.step(avg_loss)

            est_ratio = 32.0 / (1.0 + (avg_loss * 5.0))
            self.log_stamp.emit(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"Ep {epoch+1} | Loss: {avg_loss:.5f} | Est Ratio: {est_ratio:.2f}x"
            )
            self.progress.emit(int(((epoch + 1) / self.cfg['epochs']) * 100))

        self.finished.emit()


# ==========================================
# 2.5 VALIDATION WORKER
# ==========================================
class ValidationWorker(QThread):
    log_stamp = pyqtSignal(str)
    drift_update = pyqtSignal(float)
    finished = pyqtSignal(float)

    def __init__(self, codec, filepath, device):
        super().__init__()
        self.codec = codec
        self.filepath = filepath
        self.device = device

    def run(self):
        global SYNC_WINDOW_CHUNKS, RESIDUAL_Q, ROT_STRENGTH

        self.codec.to(self.device)
        self.codec.eval()

        orig_size = os.path.getsize(self.filepath)
        compressed_size = 4 + 8 + 8 + 4 + 4 + 4
        self.log_stamp.emit("--- VALIDATION START (Physical Dry Run) ---")

        with open(self.filepath, 'rb') as f:
            prev_m = prev_f = prev2_m = prev2_f = None
            chunk_index = 0
            current_sync = SYNC_WINDOW_CHUNKS
            since_anchor = 0

            while True:
                chunk = f.read(1024)
                if not chunk:
                    break

                padded = chunk + b'\x00' * (1024 - len(chunk))
                t = (
                    torch.tensor(list(padded), dtype=torch.float32, device=self.device)
                    - NORM_OFFSET
                ) / NORM_SCALE
                t = t.view(1, 1, 1024)

                m_lat, f_lat = self.codec.encode_to_latents(t)

                if since_anchor >= current_sync or prev_m is None:
                    idx = self.codec.encode_to_indices(t)
                    # Mutation applies to Anchor chunks too for consistency
                    approx = self.codec.decode_from_indices(idx, PRIME_MANAGER.get(chunk_index))
                    residual = bytes(a ^ b for a, b in zip(padded, approx))

                    res_tensor = torch.tensor(
                        list(residual), dtype=torch.float32, device=self.device
                    ) - NORM_OFFSET
                    res_q = (res_tensor / 4.0).round().clamp(-32, 31).to(torch.int8)
                    comp_res = rans_encode(res_q.cpu().numpy().tobytes())

                    drift = res_tensor.abs().mean().item()
                    self.drift_update.emit(drift)

                    compressed_size += 1 + 32 + 4 + len(comp_res)
                    prev2_m = prev_m
                    prev2_f = prev_f
                    prev_m = m_lat.detach()
                    prev_f = f_lat.detach()
                    since_anchor = 0
                else:
                    pred_m = predict_next_latent(
                        prev_m, PRIME_MANAGER.get(chunk_index), self.device, ROT_STRENGTH
                    )

                    eps = (m_lat - pred_m).clamp(-1.0, 1.0)
                    eps_q = (eps * RESIDUAL_Q).round().to(torch.int8).cpu().numpy().tobytes()
                    comp_eps = rans_encode(eps_q)

                    # Decoder mutates using the current prime
                    approx = self.codec.decode_from_latent(m_lat, PRIME_MANAGER.get(chunk_index))
                    residual = bytes(a ^ b for a, b in zip(padded, approx))

                    res_tensor = torch.tensor(
                        list(residual), dtype=torch.float32, device=self.device
                    ) - NORM_OFFSET
                    res_q = (res_tensor / 4.0).round().clamp(-32, 31).to(torch.int8)
                    comp_res = rans_encode(res_q.cpu().numpy().tobytes())

                    compressed_size += 1 + 4 + 4 + len(comp_eps) + 4 + len(comp_res)
                    prev2_m = prev_m
                    prev2_f = prev_f
                    prev_m = m_lat.detach()
                    prev_f = f_lat.detach()
                    since_anchor += 1

                    drift = res_tensor.abs().mean().item()
                    self.drift_update.emit(drift)
                    if drift > 20.0:
                        current_sync = max(4, current_sync // 2)
                    elif drift < 5.0 and current_sync < SYNC_WINDOW_CHUNKS * 8:
                        current_sync = current_sync + 1

                chunk_index += 1

        ratio = orig_size / max(1, compressed_size)
        self.finished.emit(ratio)


# ==========================================
# 3. CODEC WORKER (EXPORT / IMPORT)
# ==========================================
class CodecWorker(QThread):
    progress = pyqtSignal(int)
    log_stamp = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, codec, in_path, out_path, device, mode="export"):
        super().__init__()
        self.codec = codec
        self.in_path = in_path
        self.out_path = out_path
        self.device = device
        self.mode = mode

    def run(self):
        self.codec.to(self.device)
        self.codec.eval()
        if self.mode == "export":
            self.export_pico()
        else:
            self.import_pico()
        self.finished.emit()

    def export_pico(self):
        global SYNC_WINDOW_CHUNKS, RESIDUAL_Q, ROT_STRENGTH

        orig_size = os.path.getsize(self.in_path)
        total_chunks = (orig_size + 1023) // 1024
        self.log_stamp.emit(f"Exporting: {os.path.basename(self.out_path)}")

        with open(self.in_path, 'rb') as f_in, open(self.out_path, 'wb') as f_out:
            first_chunk = f_in.read(1024)
            if not first_chunk:
                return
            padded_first = first_chunk + b'\x00' * (1024 - len(first_chunk))
            t_first = (
                torch.tensor(list(padded_first), dtype=torch.float32, device=self.device)
                - NORM_OFFSET
            ) / NORM_SCALE
            t_first = t_first.view(1, 1, 1024)
            m_first, f_first = self.codec.encode_to_latents(t_first)
            # Quantize Anchor Latent to Int8[12] for Wire Format
            anchor_latent_vec = m_first[0].detach().cpu().numpy()
            anchor_latent_q = (anchor_latent_vec.clip(-1, 1) * 127).astype('int8').tobytes()

            f_out.write(struct.pack('<I', LSC1_MAGIC))
            anchor_ts = time.time_ns()
            f_out.write(struct.pack('<Q', anchor_ts))
            f_out.write(anchor_latent_q) # 0x0C - 0x17
            f_out.write(struct.pack('<I', 0)) # 0x18 - 0x1B: rANS Entropy Seed
            f_out.write(struct.pack('<Q', orig_size))
            f_out.write(struct.pack('<I', SYNC_WINDOW_CHUNKS))
            f_out.write(struct.pack('<f', float(RESIDUAL_Q)))
            f_out.write(struct.pack('<f', float(ROT_STRENGTH)))

            f_in.seek(0)

            prev_m = prev_f = prev2_m = prev2_f = None
            current_sync = SYNC_WINDOW_CHUNKS
            since_anchor = 0

            for i in range(total_chunks):
                chunk = f_in.read(1024)
                if not chunk:
                    break

                padded = chunk + b'\x00' * (1024 - len(chunk))
                t = (
                    torch.tensor(list(padded), dtype=torch.float32, device=self.device)
                    - NORM_OFFSET
                ) / NORM_SCALE
                t = t.view(1, 1, 1024)

                m_lat, f_lat = self.codec.encode_to_latents(t)

                if since_anchor >= current_sync or prev_m is None:
                    f_out.write(b'\x01')
                    prime = PRIME_MANAGER.get(i)
                    idx = self.codec.encode_to_indices(t)
                    approx = self.codec.decode_from_indices(idx, prime)

                    residual = bytes(a ^ b for a, b in zip(padded, approx))
                    res_tensor = torch.tensor(
                        list(residual), dtype=torch.float32, device=self.device
                    ) - NORM_OFFSET
                    res_q = (res_tensor / 4.0).round().clamp(-32, 31).to(torch.int8)
                    comp_res = rans_encode(res_q.cpu().numpy().tobytes())

                    f_out.write(struct.pack('<16H', *idx.cpu().tolist()))
                    f_out.write(struct.pack('<I', len(comp_res)))
                    f_out.write(comp_res)

                    prev2_m = prev_m
                    prev2_f = prev_f
                    prev_m = m_lat.detach()
                    prev_f = f_lat.detach()
                    since_anchor = 0
                else:
                    f_out.write(b'\x02')
                    prime = PRIME_MANAGER.get(i)
                    f_out.write(struct.pack('<I', prime))

                    pred_m = predict_next_latent(
                        prev_m, prime, self.device, ROT_STRENGTH
                    )

                    eps = (m_lat - pred_m).clamp(-1.0, 1.0)
                    eps_q = (eps * RESIDUAL_Q).round().to(torch.int8).cpu().numpy().tobytes()
                    comp_eps = rans_encode(eps_q)

                    approx = self.codec.decode_from_latent(m_lat, prime)
                    residual = bytes(a ^ b for a, b in zip(padded, approx))
                    res_tensor = torch.tensor(
                        list(residual), dtype=torch.float32, device=self.device
                    ) - NORM_OFFSET
                    res_q = (res_tensor / 4.0).round().clamp(-32, 31).to(torch.int8)
                    comp_res = rans_encode(res_q.cpu().numpy().tobytes())

                    f_out.write(struct.pack('<I', len(comp_eps)))
                    f_out.write(comp_eps)
                    f_out.write(struct.pack('<I', len(comp_res)))
                    f_out.write(comp_res)

                    prev2_m = prev_m
                    prev2_f = prev_f
                    prev_m = m_lat.detach()
                    prev_f = f_lat.detach()
                    since_anchor += 1

                    drift = res_tensor.abs().mean().item()
                    if drift > 20.0:
                        current_sync = max(4, current_sync // 2)
                    elif drift < 5.0 and current_sync < SYNC_WINDOW_CHUNKS * 8:
                        current_sync = current_sync + 1

                self.progress.emit(int(((i + 1) / total_chunks) * 100))

    def import_pico(self):
        global SYNC_WINDOW_CHUNKS, RESIDUAL_Q, ROT_STRENGTH

        self.log_stamp.emit(f"Importing/Restoring to: {os.path.basename(self.out_path)}")

        with open(self.in_path, 'rb') as f_in, open(self.out_path, 'wb') as f_out:
            magic_data = f_in.read(4)
            if not magic_data: return
            magic = struct.unpack('<I', magic_data)[0]
            if magic != LSC1_MAGIC:
                self.log_stamp.emit("Invalid LSC-1 magic header.")
                return

            anchor_ts = struct.unpack('<Q', f_in.read(8))[0]
            anchor_latent_bytes = f_in.read(12)
            rans_seed = struct.unpack('<I', f_in.read(4))[0]
            orig_size = struct.unpack('<Q', f_in.read(8))[0]
            SYNC_WINDOW_CHUNKS = struct.unpack('<I', f_in.read(4))[0]
            RESIDUAL_Q = struct.unpack('<f', f_in.read(4))[0]
            ROT_STRENGTH = struct.unpack('<f', f_in.read(4))[0]

            prev_m = prev_f = prev2_m = prev2_f = None
            chunk_index = 0

            while True:
                flag = f_in.read(1)
                if not flag:
                    break

                if flag == b'\x01':
                    idx_data = f_in.read(32)
                    if not idx_data:
                        break

                    idx = torch.tensor(
                        struct.unpack('<16H', idx_data),
                        dtype=torch.long,
                        device=self.device
                    )
                    # Note: We need the prime for flag 0x01 as well
                    approx = self.codec.decode_from_indices(idx, PRIME_MANAGER.get(chunk_index))

                    res_len_data = f_in.read(4)
                    if not res_len_data:
                        break
                    res_len = struct.unpack('<I', res_len_data)[0]
                    res_q = rans_decode(f_in.read(res_len))

                    res_tensor = torch.frombuffer(
                        res_q, dtype=torch.int8
                    ).to(self.device).float()
                    residual = (res_tensor * 4.0 + NORM_OFFSET).clamp(0, 255).to(torch.uint8)
                    residual_bytes = bytes(residual.cpu().numpy().tolist())

                    restored = bytes(a ^ b for a, b in zip(approx, residual_bytes))
                    f_out.write(restored)

                    with torch.no_grad():
                        m_lat = self.codec.codebook[idx].reshape(-1, LATENT_DIM)
                        f_lat = self.codec.female_head(m_lat)
                        f_lat = F.normalize(f_lat, p=2, dim=1)

                        prev2_m = prev_m
                        prev2_f = prev_f
                        prev_m = m_lat.detach()
                        prev_f = f_lat.detach()

                elif flag == b'\x02':
                    prime_data = f_in.read(4)
                    if not prime_data:
                        break
                    prime = struct.unpack('<I', prime_data)[0]

                    eps_len_data = f_in.read(4)
                    if not eps_len_data:
                        break
                    eps_len = struct.unpack('<I', eps_len_data)[0]
                    eps_q = rans_decode(f_in.read(eps_len))

                    res_len_data = f_in.read(4)
                    if not res_len_data:
                        break
                    res_len = struct.unpack('<I', res_len_data)[0]
                    res_q = rans_decode(f_in.read(res_len))

                    with torch.no_grad():
                        pred_m = predict_next_latent(
                            prev_m, prime, self.device, ROT_STRENGTH
                        )
                        eps_tensor = torch.frombuffer(
                            eps_q, dtype=torch.int8
                        ).to(self.device).view_as(pred_m)
                        eps = (eps_tensor.float() / RESIDUAL_Q).clamp(-1.0, 1.0)
                        m_lat = (pred_m + eps).clamp(-1.0, 1.0)
                        approx = self.codec.decode_from_latent(m_lat, prime)

                    res_tensor = torch.frombuffer(
                        res_q, dtype=torch.int8
                    ).to(self.device).float()
                    residual = (res_tensor * 4.0 + NORM_OFFSET).clamp(0, 255).to(torch.uint8)
                    residual_bytes = bytes(residual.cpu().numpy().tolist())

                    restored = bytes(a ^ b for a, b in zip(approx, residual_bytes))
                    f_out.write(restored)

                    with torch.no_grad():
                        f_lat = self.codec.female_head(m_lat)
                        f_lat = F.normalize(f_lat, p=2, dim=1)
                        prev2_m = prev_m
                        prev2_f = prev_f
                        prev_m = m_lat.detach()
                        prev_f = f_lat.detach()
                else:
                    self.log_stamp.emit("Unknown record flag, aborting.")
                    break

                chunk_index += 1

            f_out.truncate(orig_size)


# ==========================================
# 3.5 GENETIC UTILITIES
# ==========================================
def multi_stage_genetic_prediction(prev_m, prev_f, prev2_m, prev2_f,
                                   prime1, prime2, device, rot_strength=1.0):
    if prev_m is None or prev_f is None:
        return torch.zeros((1, LATENT_DIM), device=device)

    rot_m1 = apply_rotation_to_latent(prev_m, prime1, device, rot_strength)
    rot_f1 = apply_rotation_to_latent(prev_f, prime1, device, rot_strength * 0.5)
    pred1 = genetic_combine(rot_m1, rot_f1, alpha=0.7)

    if prev2_m is None or prev2_f is None:
        return pred1

    rot_m2 = apply_rotation_to_latent(prev2_m, prime2, device, rot_strength * 0.5)
    rot_f2 = apply_rotation_to_latent(prev2_f, prime2, device, rot_strength * 0.25)
    pred2 = genetic_combine(rot_m2, rot_f2, alpha=0.6)

    return 0.7 * pred1 + 0.3 * pred2



# ==========================================
# 4. MAIN GUI (PRESETS + MULTIPLIER + PAUSE/STOP)
# ==========================================
class PicoCompressor(QWidget):
    def __init__(self):
        super().__init__()
        self.shared_codec = SphericalConvCodec()
        self.selected_file_for_validation = None

        self.base_presets = self._default_presets()
        self.custom_presets = self._load_custom_presets()
        self.current_base_preset = None

        self.init_ui()
        self._set_default_preset()

        self.mem_timer = QTimer(self)
        self.mem_timer.timeout.connect(self._update_mem_status)
        self.mem_timer.start(2000)

    def _update_mem_status(self):
        if psutil:
            try:
                mem_gb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
                limit = self.edit_ram.text() or "0"
                status = f"RAM Usage: {mem_gb:.2f} GB / {limit} GB Limit"
                self.lbl_state.setText(status)
            except Exception:
                pass
        else:
            self.lbl_state.setText("RAM Usage: psutil not installed")

    def _default_presets(self):
        return {
            "Slow": {
                "epochs": 20,
                "lr": 0.001,
                "commit": 0.75,
                "sync": 128,
                "res_q": 200.0,
                "rot": 0.75,
                "prime_growth": 1.5,
                "passes": 2
            },
            "Balanced": {
                "epochs": 40,
                "lr": 0.005,
                "commit": 0.25,
                "sync": 64,
                "res_q": 127.0,
                "rot": 1.0,
                "prime_growth": 2.0,
                "passes": 2
            },
            "Aggressive": {
                "epochs": 80,
                "lr": 0.02,
                "commit": 0.10,
                "sync": 32,
                "res_q": 80.0,
                "rot": 1.5,
                "prime_growth": 3.0,
                "passes": 3
            }
        }

    def _load_custom_presets(self):
        if not os.path.exists(PRESET_FILE):
            return {}
        try:
            with open(PRESET_FILE, "r") as f:
                data = json.load(f)
            return data.get("custom", {})
        except Exception:
            return {}

    def _save_custom_presets(self):
        data = {"custom": self.custom_presets}
        try:
            with open(PRESET_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def init_ui(self):
        self.setWindowTitle("PicoCompressor (LSC‑1 Genetic‑AR)")
        self.resize(900, 1100)
        self.setMinimumSize(600, 700)
        self.setStyleSheet(
            "QWidget { background-color: #0d1117; color: #c9d1d9; "
            "font-family: sans-serif; }"
        )

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet("border: none;")

        container = QWidget()
        layout = QVBoxLayout(container)

        # Model continuity
        state_group = QGroupBox("Model Continuity")
        slayout = QVBoxLayout()
        self.lbl_state = QLabel("RAM State: Fresh")
        self.lbl_state.setStyleSheet("color: #58a6ff; font-weight: bold;")
        slayout.addWidget(self.lbl_state)

        brow = QHBoxLayout()
        self.btn_reset = QPushButton("New Brain")
        self.btn_reset.clicked.connect(self.reset_model)
        self.btn_load = QPushButton("Load Weights")
        self.btn_load.clicked.connect(self.load_weights)
        self.btn_save = QPushButton("Save Weights")
        self.btn_save.clicked.connect(self.save_weights)
        brow.addWidget(self.btn_reset)
        brow.addWidget(self.btn_load)
        brow.addWidget(self.btn_save)
        slayout.addLayout(brow)
        state_group.setLayout(slayout)
        layout.addWidget(state_group)

        # Console
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet(
            "background-color: #010409; color: #7ee787; "
            "font-family: 'Consolas'; font-size: 11px;"
        )
        layout.addWidget(self.console)

        # Preset controls
        preset_group = QGroupBox("Profiles")
        playout = QVBoxLayout()

        prow1 = QHBoxLayout()
        self.combo_preset = QComboBox()
        self._refresh_preset_combo()
        self.combo_preset.currentIndexChanged.connect(self._on_preset_changed)

        self.slider_multiplier = QSlider(Qt.Horizontal)
        self.slider_multiplier.setMinimum(1)
        self.slider_multiplier.setMaximum(100)
        self.slider_multiplier.setValue(1)
        self.slider_multiplier.valueChanged.connect(self._on_multiplier_changed)

        self.lbl_multiplier = QLabel("1x")
        self.lbl_multiplier.setFixedWidth(40)

        prow1.addWidget(QLabel("Preset:"))
        prow1.addWidget(self.combo_preset)
        playout.addLayout(prow1)

        prow2 = QHBoxLayout()
        prow2.addWidget(QLabel("Intensity:"))
        prow2.addWidget(self.slider_multiplier)
        prow2.addWidget(self.lbl_multiplier)
        playout.addLayout(prow2)

        prow3 = QHBoxLayout()
        btn_save_preset = QPushButton("Save Preset")
        btn_save_preset.clicked.connect(self._save_preset_from_current)
        btn_default = QPushButton("Default")
        btn_default.clicked.connect(self._set_default_preset)
        prow3.addWidget(btn_save_preset)
        prow3.addWidget(btn_default)
        playout.addLayout(prow3)

        preset_group.setLayout(playout)
        layout.addWidget(preset_group)

        # Tabs
        tabs = QTabWidget()

        # System tab
        res_tab = QWidget()
        rform = QFormLayout()

        self.edit_ram = QLineEdit("4")
        self.edit_threads = QLineEdit("8")

        self.combo_device = QComboBox()
        self.combo_device.addItem("CPU", "cpu")
        if torch.cuda.is_available():
            self.combo_device.addItem("NVIDIA CUDA", "cuda")
        if torch.backends.mps.is_available():
            self.combo_device.addItem("Apple Neural (MPS)", "mps")

        rform.addRow("RAM (GB):", self.edit_ram)
        rform.addRow("Threads:", self.edit_threads)
        rform.addRow("Device:", self.combo_device)

        res_tab.setLayout(rform)
        tabs.addTab(res_tab, "System")

        # Training tab
        train_tab = QWidget()
        tform = QFormLayout()

        self.btn_select_train = QPushButton("Select Training File")
        self.btn_select_train.clicked.connect(self.select_training_file)
        tform.addRow(self.btn_select_train)

        self.btn_start_train = QPushButton("Start Training")
        self.btn_start_train.clicked.connect(self.start_training)
        self.btn_pause_train = QPushButton("Pause")
        self.btn_pause_train.clicked.connect(self.pause_training)
        self.btn_resume_train = QPushButton("Resume")
        self.btn_resume_train.clicked.connect(self.resume_training)
        self.btn_stop_train = QPushButton("Stop")
        self.btn_stop_train.clicked.connect(self.stop_training)

        self.btn_collapse = QPushButton("Collapse Manifold")
        self.btn_collapse.clicked.connect(self.collapse_manifold)

        tform.addRow(self.btn_start_train)
        tform.addRow(self.btn_collapse)
        tform.addRow(self.btn_pause_train)
        tform.addRow(self.btn_resume_train)
        tform.addRow(self.btn_stop_train)

        train_tab.setLayout(tform)
        tabs.addTab(train_tab, "Training")

        # Validation tab
        val_tab = QWidget()
        vform = QFormLayout()

        self.btn_select_val = QPushButton("Select File for Validation")
        self.btn_select_val.clicked.connect(self.select_validation_file)
        vform.addRow(self.btn_select_val)

        self.btn_run_val = QPushButton("Run Validation")
        self.btn_run_val.clicked.connect(self.run_validation)
        vform.addRow(self.btn_run_val)
        
        self.drift_graph = DriftGraph()
        vform.addRow(QLabel("Real-time Epsilon Drift (MAE):"))
        vform.addRow(self.drift_graph)

        val_tab.setLayout(vform)
        tabs.addTab(val_tab, "Validation")

        # Export tab
        exp_tab = QWidget()
        eform = QFormLayout()

        self.btn_select_export = QPushButton("Select File to Export")
        self.btn_select_export.clicked.connect(self.select_export_file)
        eform.addRow(self.btn_select_export)

        self.btn_export = QPushButton("Export to .pico")
        self.btn_export.clicked.connect(self.export_pico)
        eform.addRow(self.btn_export)

        exp_tab.setLayout(eform)
        tabs.addTab(exp_tab, "Export")

        # Import tab
        imp_tab = QWidget()
        iform = QFormLayout()

        self.btn_select_import = QPushButton("Select .pico File")
        self.btn_select_import.clicked.connect(self.select_import_file)
        iform.addRow(self.btn_select_import)

        self.btn_import = QPushButton("Import / Restore")
        self.btn_import.clicked.connect(self.import_pico)
        iform.addRow(self.btn_import)

        layout.addWidget(tabs)
        scroll.setWidget(container)
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll)

    # ============================
    # GUI CALLBACKS
    # ============================
    def toggle_ui(self, enabled=True):
        self.btn_start_train.setEnabled(enabled)
        self.btn_collapse.setEnabled(enabled)
        self.btn_run_val.setEnabled(enabled)
        self.btn_export.setEnabled(enabled)
        self.btn_import.setEnabled(enabled)
        self.btn_load.setEnabled(enabled)
        self.btn_reset.setEnabled(enabled)
        self.btn_save.setEnabled(enabled)

    def collapse_manifold(self):
        global SYNC_WINDOW_CHUNKS
        # Reconciles the sync window length to training execution points (SEQ_LEN)
        # This ensures mutation anchors align with batch boundaries for stable rebound.
        SYNC_WINDOW_CHUNKS = max(SEQ_LEN, (SYNC_WINDOW_CHUNKS // SEQ_LEN) * SEQ_LEN)
        
        # Perform a topological collapse (normalization) on existing weights 
        # instead of re-initializing (resetting) the brain.
        with torch.no_grad():
            self.shared_codec.codebook.data = F.normalize(self.shared_codec.codebook.data, p=2, dim=1)
            # Ensure linear heads maintain orthonormal projection integrity
            self.shared_codec.male_head.weight.data = F.normalize(self.shared_codec.male_head.weight.data, p=2, dim=0)
            self.shared_codec.female_head.weight.data = F.normalize(self.shared_codec.female_head.weight.data, p=2, dim=0)

        self.lbl_state.setText("RAM State: Collapsed (Preserved)")
        self.console.append(f"Manifold Collapsed: Sync window at {SYNC_WINDOW_CHUNKS}. Brain orientation preserved and projected to S^11.")

    def reset_model(self):
        if hasattr(self, "train_worker") and self.train_worker.isRunning(): return
        self.shared_codec = SphericalConvCodec()
        self.lbl_state.setText("RAM State: Fresh")
        self.console.append("Model reset.")

    def load_weights(self):
        if hasattr(self, "train_worker") and self.train_worker.isRunning(): return
        path, _ = QFileDialog.getOpenFileName(self, "Load Weights", "", "Weights (*.pt)")
        if not path:
            return
        try:
            self.shared_codec.load_state_dict(torch.load(path, map_location="cpu"))
            self.lbl_state.setText("RAM State: Loaded")
            self.console.append(f"Weights loaded: {path}")
        except Exception as e:
            self.console.append(f"Error loading weights: {e}")

    def save_weights(self):
        if hasattr(self, "train_worker") and self.train_worker.isRunning(): return
        path, _ = QFileDialog.getSaveFileName(self, "Save Weights", "", "Weights (*.pt)")
        if not path:
            return
        try:
            torch.save(self.shared_codec.state_dict(), path)
            self.console.append(f"Weights saved: {path}")
        except Exception as e:
            self.console.append(f"Error saving weights: {e}")

    def _refresh_preset_combo(self):
        self.combo_preset.clear()
        for k in self.base_presets:
            self.combo_preset.addItem(k)
        for k in self.custom_presets:
            self.combo_preset.addItem(k + " (Custom)")

    def _on_preset_changed(self):
        name = self.combo_preset.currentText()
        if name.endswith("(Custom)"):
            name = name.replace(" (Custom)", "")
            preset = self.custom_presets.get(name, {})
        else:
            preset = self.base_presets.get(name, {})
        self.current_base_preset = preset
        self._apply_preset(preset)

    def _apply_preset(self, preset):
        global SYNC_WINDOW_CHUNKS, RESIDUAL_Q, ROT_STRENGTH
        if not preset:
            return
        SYNC_WINDOW_CHUNKS = preset["sync"]
        RESIDUAL_Q = preset["res_q"]
        ROT_STRENGTH = preset["rot"]
        self.console.append(f"Preset applied: {preset}")

    def _on_multiplier_changed(self):
        val = self.slider_multiplier.value()
        self.lbl_multiplier.setText(f"{val}x")

    def _save_preset_from_current(self):
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if not ok or not name:
            return
        preset = dict(self.current_base_preset)
        preset["sync"] = SYNC_WINDOW_CHUNKS
        preset["res_q"] = RESIDUAL_Q
        preset["rot"] = ROT_STRENGTH
        self.custom_presets[name] = preset
        self._save_custom_presets()
        self._refresh_preset_combo()
        self.console.append(f"Custom preset saved: {name}")

    def _set_default_preset(self):
        self.combo_preset.setCurrentIndex(1)
        self.console.append("Default preset loaded.")

    # ============================
    # TRAINING / VALIDATION / EXPORT / IMPORT
    # ============================

    def select_training_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Training File", "", "All Files (*)")
        if path:
            self.training_file = path
            self.console.append(f"Training file selected: {path}")

    def start_training(self):
        if not hasattr(self, "training_file"):
            self.console.append("No training file selected.")
            return

        try:
            ram_limit = float(self.edit_ram.text() or 4.0)
        except ValueError:
            ram_limit = 4.0

        cfg = {
            "epochs": self.current_base_preset["epochs"],
            "lr": self.current_base_preset["lr"],
            "commit": self.current_base_preset["commit"],
            "passes": self.current_base_preset["passes"],
            "ram_limit_gb": ram_limit,
            "threads": int(self.edit_threads.text()),
            "device": self.combo_device.currentData(),
        }

        self.toggle_ui(False)
        self.train_worker = TrainingWorker(self.shared_codec, self.training_file, cfg)
        self.train_worker.log_stamp.connect(self.console.append)
        self.train_worker.progress.connect(lambda v: self.console.append(f"Progress: {v}%"))
        self.train_worker.finished.connect(lambda: [self.console.append("Training finished."), self.toggle_ui(True)])
        self.train_worker.stopped.connect(lambda: [self.console.append("Training stopped."), self.toggle_ui(True)])
        self.train_worker.start()

    def pause_training(self):
        if hasattr(self, "train_worker"):
            self.train_worker.pause()
            self.console.append("Training paused.")

    def resume_training(self):
        if hasattr(self, "train_worker"):
            self.train_worker.resume()
            self.console.append("Training resumed.")

    def stop_training(self):
        if hasattr(self, "train_worker"):
            self.train_worker.stop()
            self.console.append("Training stop requested.")

    def select_validation_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select File for Validation", "", "All Files (*)")
        if path:
            self.selected_file_for_validation = path
            self.console.append(f"Validation file selected: {path}")

    def run_validation(self):
        if not self.selected_file_for_validation:
            self.console.append("No validation file selected.")
            return

        self.toggle_ui(False)
        self.drift_graph.clear()
        self.val_worker = ValidationWorker(self.shared_codec, self.selected_file_for_validation, self.combo_device.currentData())
        self.val_worker.log_stamp.connect(self.console.append)
        self.val_worker.drift_update.connect(self.drift_graph.add_value)
        self.val_worker.finished.connect(lambda r: [self.console.append(f"Validation ratio: {r:.2f}x"), self.toggle_ui(True)])
        self.val_worker.start()

    def select_export_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select File to Export", "", "All Files (*)")
        if path:
            self.export_file = path
            self.console.append(f"Export file selected: {path}")

    def export_pico(self):
        if not hasattr(self, "export_file"):
            self.console.append("No export file selected.")
            return

        out_path, _ = QFileDialog.getSaveFileName(self, "Save .pico", "", "Pico Files (*.pico)")
        if not out_path:
            return

        self.toggle_ui(False)
        self.codec_worker = CodecWorker(self.shared_codec, self.export_file, out_path, self.combo_device.currentData(), mode="export")
        self.codec_worker.log_stamp.connect(self.console.append)
        self.codec_worker.progress.connect(lambda v: self.console.append(f"Export: {v}%"))
        self.codec_worker.finished.connect(lambda: [self.console.append("Export complete."), self.toggle_ui(True)])
        self.codec_worker.start()

    def select_import_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select .pico File", "", "Pico Files (*.pico)")
        if path:
            self.import_file = path
            self.console.append(f"Import file selected: {path}")

    def import_pico(self):
        if not hasattr(self, "import_file"):
            self.console.append("No .pico file selected.")
            return

        out_path, _ = QFileDialog.getSaveFileName(self, "Restore Output", "", "All Files (*)")
        if not out_path:
            return

        self.toggle_ui(False)
        self.codec_worker = CodecWorker(self.shared_codec, self.import_file, out_path, self.combo_device.currentData(), mode="import")
        self.codec_worker.log_stamp.connect(self.console.append)
        self.codec_worker.progress.connect(lambda v: self.console.append(f"Import: {v}%"))
        self.codec_worker.finished.connect(lambda: [self.console.append("Import complete."), self.toggle_ui(True)])
        self.codec_worker.start()


# ==========================================
# 5. APPLICATION ENTRY POINT
# ==========================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = PicoCompressor()
    win.show()
    sys.exit(app.exec_())

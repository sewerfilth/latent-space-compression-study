import sys
import os
import struct
import zlib
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QLabel, QFileDialog, QProgressBar, QHBoxLayout,
    QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QComboBox, QTextEdit, QTabWidget, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# ==========================================
# 0. GLOBALS / LSC-1 PARAMETERS
# ==========================================
LATENT_DIM = 12
CODEBOOK_SIZE = 8192
SYNC_WINDOW_CHUNKS = 64  # "Hard Sync" window in chunks

# ==========================================
# 1. CORE ARCHITECTURE (LSC-1-LIKE)
# ==========================================
class SphericalConvCodec(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: 1024 bytes -> 16 latent vectors
        self.push = nn.Sequential(
            nn.Conv1d(1, 16, 4, 4), nn.ReLU(),
            nn.Conv1d(16, 32, 4, 4), nn.ReLU(),
            nn.Conv1d(32, LATENT_DIM, 4, 4)
        )

        torch.manual_seed(42)
        initial_cb = F.normalize(
            torch.randn(CODEBOOK_SIZE, LATENT_DIM),
            p=2, dim=1
        )
        self.codebook = nn.Parameter(initial_cb)

        # Decoder: 16 latent vectors -> 1024 approximated bytes
        self.pop = nn.Sequential(
            nn.ConvTranspose1d(LATENT_DIM, 32, 4, 4), nn.ReLU(),
            nn.ConvTranspose1d(32, 16, 4, 4), nn.ReLU(),
            nn.ConvTranspose1d(16, 1, 4, 4)
        )

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

    def forward_train(self, x):
        continuous = self.push(x)
        continuous = F.normalize(continuous, p=2, dim=1)

        flat_latent = continuous.transpose(1, 2).reshape(-1, LATENT_DIM)
        indices = self._nearest_codebook_indices(flat_latent)

        quantized = self.codebook[indices].reshape(-1, 16, LATENT_DIM).transpose(1, 2)

        quantized_ste = continuous + (quantized - continuous).detach()

        recon = self.pop(quantized_ste)
        return recon, continuous, quantized

    @torch.no_grad()
    def encode_to_indices(self, x):
        continuous = F.normalize(self.push(x), p=2, dim=1)
        flat_latent = continuous.transpose(1, 2).reshape(-1, LATENT_DIM)
        indices = self._nearest_codebook_indices(flat_latent)
        return indices

    @torch.no_grad()
    def encode_to_latent(self, x):
        # [1,1,1024] -> [16, LATENT_DIM] flattened
        continuous = F.normalize(self.push(x), p=2, dim=1)
        flat_latent = continuous.transpose(1, 2).reshape(-1, LATENT_DIM)
        return flat_latent  # [16, 12] flattened as 16*12 if needed

    @torch.no_grad()
    def decode_from_indices(self, indices):
        quantized = self.codebook[indices].reshape(-1, 16, LATENT_DIM).transpose(1, 2)
        x = self.pop(quantized)
        return (
            (x * 50.0 + 128.0)
            .clamp(0, 255)
            .to(torch.uint8)
            .squeeze()
            .cpu()
            .numpy()
            .tobytes()
        )

    @torch.no_grad()
    def decode_from_latent(self, latent):
        # latent: [16, LATENT_DIM]
        quantized = latent.reshape(-1, 16, LATENT_DIM).transpose(1, 2)
        x = self.pop(quantized)
        return (
            (x * 50.0 + 128.0)
            .clamp(0, 255)
            .to(torch.uint8)
            .squeeze()
            .cpu()
            .numpy()
            .tobytes()
        )

# ==========================================
# 1.5 PRIME-INDEXED ROTATIONS (LSC-1 STYLE)
# ==========================================
def generate_primes(n):
    primes = []
    candidate = 2
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

PRIME_CACHE = generate_primes(100000)  # plenty for long streams

def givens_rotation_matrix(dim, i, j, theta, device):
    R = torch.eye(dim, device=device)
    c = torch.cos(theta)
    s = torch.sin(theta)
    R[i, i] = c
    R[j, j] = c
    R[i, j] = -s
    R[j, i] = s
    return R

def rotation_from_prime(prime, dim, device):
    # Deterministic, prime-seeded orthonormal rotation via chained Givens
    torch.manual_seed(prime)
    R = torch.eye(dim, device=device)
    for _ in range(6):  # 6 pairs as in the whitepaper
        i = torch.randint(0, dim, (1,)).item()
        j = torch.randint(0, dim, (1,)).item()
        while j == i:
            j = torch.randint(0, dim, (1,)).item()
        theta = torch.rand(1, device=device) * 2 * torch.pi
        G = givens_rotation_matrix(dim, i, j, theta, device)
        R = G @ R
    # Optional re-orthonormalization
    Q, _ = torch.linalg.qr(R)
    return Q

def apply_rotation_to_latent(latent_flat, prime, device):
    # latent_flat: [16, LATENT_DIM]
    R = rotation_from_prime(prime, LATENT_DIM, device)
    return (latent_flat @ R.T)  # [16, 12]

# ==========================================
# 2. TRAINING + VALIDATION WORKERS
# ==========================================
class TrainingWorker(QThread):
    progress = pyqtSignal(int)
    log_stamp = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, codec, filepath, config):
        super().__init__()
        self.codec = codec
        self.filepath = filepath
        self.cfg = config

    def run(self):
        torch.set_num_threads(self.cfg['threads'])
        device = self.cfg['device']
        self.codec.to(device)
        self.codec.train()

        optimizer = torch.optim.Adam(self.codec.parameters(), lr=self.cfg['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )

        buffer_size = 256

        self.log_stamp.emit(f"--- ROUND START: {datetime.now().strftime('%H:%M:%S')} ---")

        for epoch in range(self.cfg['epochs']):
            total_loss, count = 0.0, 0
            with open(self.filepath, 'rb') as f:
                while True:
                    batch_list = []
                    for _ in range(buffer_size):
                        chunk = f.read(1024)
                        if not chunk:
                            break
                        padded = chunk + b'\x00' * (1024 - len(chunk))
                        t = (
                            torch.tensor(list(padded), dtype=torch.float32, device=device)
                            - 128.0
                        ) / 50.0
                        batch_list.append(t.view(1, 1, 1024))
                    if not batch_list:
                        break

                    batch = torch.cat(batch_list, dim=0)
                    optimizer.zero_grad()
                    recon, cont, quant = self.codec.forward_train(batch)
                    loss = F.mse_loss(recon, batch) + (
                        self.cfg['commit'] * F.mse_loss(cont, quant.detach())
                    )
                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        self.codec.codebook.div_(
                            torch.norm(self.codec.codebook, dim=1, keepdim=True)
                        )

                    total_loss += loss.item()
                    count += 1

            avg_loss = total_loss / max(1, count)
            scheduler.step(avg_loss)
            est_ratio = 32.0 / (1.0 + (avg_loss * 5.0))
            self.log_stamp.emit(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"Ep {epoch+1} | Loss: {avg_loss:.5f} | Est Ratio: {est_ratio:.2f}x"
            )
            self.progress.emit(int(((epoch + 1) / self.cfg['epochs']) * 100))

        self.finished.emit()

class ValidationWorker(QThread):
    log_stamp = pyqtSignal(str)
    finished = pyqtSignal(float)

    def __init__(self, codec, filepath, device):
        super().__init__()
        self.codec = codec
        self.filepath = filepath
        self.device = device

    def run(self):
        self.codec.to(self.device)
        self.codec.eval()

        orig_size = os.path.getsize(self.filepath)
        compressed_size = 12  # header baseline
        self.log_stamp.emit("--- VALIDATION START (Physical Dry Run) ---")

        with open(self.filepath, 'rb') as f:
            chunk_index = 0
            prev_latent = None
            while True:
                chunk = f.read(1024)
                if not chunk:
                    break
                padded = chunk + b'\x00' * (1024 - len(chunk))
                tensor_in = (
                    torch.tensor(list(padded), dtype=torch.float32, device=self.device)
                    - 128.0
                ) / 50.0
                tensor_in = tensor_in.view(1, 1, 1024)

                latent = self.codec.encode_to_latent(tensor_in)  # [16,12]

                if chunk_index % SYNC_WINDOW_CHUNKS == 0 or prev_latent is None:
                    # Anchor: store 16 indices + residual
                    indices = self.codec.encode_to_indices(tensor_in)
                    approx = self.codec.decode_from_indices(indices)
                    residual = bytes(a ^ b for a, b in zip(padded, approx))
                    comp_res = zlib.compress(residual, level=9)

                    # flag(1) + 16 indices(32) + res_len(4) + residual
                    compressed_size += 1 + 32 + 4 + len(comp_res)
                    prev_latent = latent.detach()
                else:
                    # Mutation: prime-indexed rotation + latent remainder + residual
                    prime = PRIME_CACHE[chunk_index]
                    pred_latent = apply_rotation_to_latent(prev_latent, prime, self.device)
                    eps = (latent - pred_latent).clamp(-1.0, 1.0)
                    eps_q = (eps * 127.0).round().to(torch.int8).cpu().numpy().tobytes()
                    comp_eps = zlib.compress(eps_q, level=9)

                    approx = self.codec.decode_from_latent(latent)
                    residual = bytes(a ^ b for a, b in zip(padded, approx))
                    comp_res = zlib.compress(residual, level=9)

                    # flag(1) + prime(4) + eps_len(4) + eps + res_len(4) + residual
                    compressed_size += 1 + 4 + 4 + len(comp_eps) + 4 + len(comp_res)
                    prev_latent = latent.detach()

                chunk_index += 1

        real_ratio = orig_size / max(1, compressed_size)
        self.finished.emit(real_ratio)

# ==========================================
# 3. IO WORKER (EXPORT / IMPORT) WITH LSC-1
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
        orig_size = os.path.getsize(self.in_path)
        total_chunks = (orig_size + 1023) // 1024
        self.log_stamp.emit(f"Exporting: {os.path.basename(self.out_path)}")

        with open(self.in_path, 'rb') as f_in, open(self.out_path, 'wb') as f_out:
            # Header: magic + original size + sync window
            f_out.write(b'PICO')
            f_out.write(struct.pack('<Q', orig_size))
            f_out.write(struct.pack('<I', SYNC_WINDOW_CHUNKS))

            prev_latent = None
            for i in range(total_chunks):
                chunk = f_in.read(1024)
                if not chunk:
                    break
                padded = chunk + b'\x00' * (1024 - len(chunk))
                t = (
                    torch.tensor(list(padded), dtype=torch.float32, device=self.device)
                    - 128.0
                ) / 50.0
                t = t.view(1, 1, 1024)

                latent = self.codec.encode_to_latent(t)  # [16,12]

                if i % SYNC_WINDOW_CHUNKS == 0 or prev_latent is None:
                    # Anchor record
                    f_out.write(b'\x01')  # flag: anchor
                    indices = self.codec.encode_to_indices(t)
                    approx = self.codec.decode_from_indices(indices)

                    residual = bytes(a ^ b for a, b in zip(padded, approx))
                    comp_res = zlib.compress(residual, level=9)

                    f_out.write(struct.pack('<16H', *indices.cpu().tolist()))
                    f_out.write(struct.pack('<I', len(comp_res)))
                    f_out.write(comp_res)

                    prev_latent = latent.detach()
                else:
                    # Mutation record
                    f_out.write(b'\x02')  # flag: mutation
                    prime = PRIME_CACHE[i]
                    f_out.write(struct.pack('<I', prime))

                    pred_latent = apply_rotation_to_latent(prev_latent, prime, self.device)
                    eps = (latent - pred_latent).clamp(-1.0, 1.0)
                    eps_q = (eps * 127.0).round().to(torch.int8).cpu().numpy().tobytes()
                    comp_eps = zlib.compress(eps_q, level=9)

                    approx = self.codec.decode_from_latent(latent)
                    residual = bytes(a ^ b for a, b in zip(padded, approx))
                    comp_res = zlib.compress(residual, level=9)

                    f_out.write(struct.pack('<I', len(comp_eps)))
                    f_out.write(comp_eps)
                    f_out.write(struct.pack('<I', len(comp_res)))
                    f_out.write(comp_res)

                    prev_latent = latent.detach()

                self.progress.emit(int(((i + 1) / total_chunks) * 100))

    def import_pico(self):
        self.log_stamp.emit(f"Importing/Restoring to: {os.path.basename(self.out_path)}")

        with open(self.in_path, 'rb') as f_in, open(self.out_path, 'wb') as f_out:
            if f_in.read(4) != b'PICO':
                self.log_stamp.emit("Invalid PICO header.")
                return

            orig_size = struct.unpack('<Q', f_in.read(8))[0]
            sync_window = struct.unpack('<I', f_in.read(4))[0]

            prev_latent = None
            chunk_index = 0

            while True:
                flag = f_in.read(1)
                if not flag:
                    break

                if flag == b'\x01':
                    # Anchor
                    idx_data = f_in.read(32)
                    if not idx_data:
                        break
                    indices = torch.tensor(
                        struct.unpack('<16H', idx_data),
                        dtype=torch.long,
                        device=self.device
                    )
                    approx = self.codec.decode_from_indices(indices)

                    res_len_data = f_in.read(4)
                    if not res_len_data:
                        break
                    res_len = struct.unpack('<I', res_len_data)[0]
                    residual = zlib.decompress(f_in.read(res_len))

                    restored = bytes(a ^ b for a, b in zip(approx, residual))
                    f_out.write(restored)

                    # Reconstruct latent from codebook indices
                    with torch.no_grad():
                        latent = self.codec.codebook[indices].reshape(-1, LATENT_DIM)
                        prev_latent = latent.detach()

                elif flag == b'\x02':
                    # Mutation
                    prime_data = f_in.read(4)
                    if not prime_data:
                        break
                    prime = struct.unpack('<I', prime_data)[0]

                    eps_len_data = f_in.read(4)
                    if not eps_len_data:
                        break
                    eps_len = struct.unpack('<I', eps_len_data)[0]
                    eps_q = zlib.decompress(f_in.read(eps_len))

                    res_len_data = f_in.read(4)
                    if not res_len_data:
                        break
                    res_len = struct.unpack('<I', res_len_data)[0]
                    residual = zlib.decompress(f_in.read(res_len))

                    with torch.no_grad():
                        pred_latent = apply_rotation_to_latent(prev_latent, prime, self.device)
                        eps_tensor = torch.frombuffer(
                            eps_q, dtype=torch.int8
                        ).to(self.device).view_as(pred_latent)
                        eps = (eps_tensor.float() / 127.0).clamp(-1.0, 1.0)
                        latent = (pred_latent + eps).clamp(-1.0, 1.0)
                        approx = self.codec.decode_from_latent(latent)

                    restored = bytes(a ^ b for a, b in zip(approx, residual))
                    f_out.write(restored)
                    prev_latent = latent.detach()
                else:
                    self.log_stamp.emit("Unknown record flag, aborting.")
                    break

                chunk_index += 1

            f_out.truncate(orig_size)

# ==========================================
# 4. MAIN GUI
# ==========================================
class PicoCompressor(QWidget):
    def __init__(self):
        super().__init__()
        self.shared_codec = SphericalConvCodec()
        self.selected_file_for_validation = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("PicoCompressor")
        self.setFixedSize(550, 820)
        self.setStyleSheet(
            "QWidget { background-color: #0d1117; color: #c9d1d9; "
            "font-family: 'Segoe UI'; }"
        )

        layout = QVBoxLayout()

        # Model continuity
        state_group = QGroupBox("Model Continuity")
        slayout = QVBoxLayout()
        self.lbl_state = QLabel("RAM State: Fresh")
        self.lbl_state.setStyleSheet("color: #58a6ff; font-weight: bold;")
        slayout.addWidget(self.lbl_state)

        brow = QHBoxLayout()
        b_reset = QPushButton("New Brain")
        b_reset.clicked.connect(self.reset_model)
        b_load = QPushButton("Load Weights")
        b_load.clicked.connect(self.load_weights)
        b_save = QPushButton("Save Weights")
        b_save.clicked.connect(self.save_weights)
        brow.addWidget(b_reset)
        brow.addWidget(b_load)
        brow.addWidget(b_save)
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

        # Tabs
        tabs = QTabWidget()

        # System tab
        res_tab = QWidget()
        rform = QFormLayout()
        self.spin_ram = QSpinBox()
        self.spin_ram.setRange(1, 16)
        self.spin_ram.setValue(4)

        self.spin_threads = QSpinBox()
        self.spin_threads.setRange(1, 32)
        self.spin_threads.setValue(8)

        self.combo_device = QComboBox()
        self.combo_device.addItem("CPU", "cpu")
        if torch.cuda.is_available():
            self.combo_device.addItem("NVIDIA CUDA", "cuda")
        if torch.backends.mps.is_available():
            self.combo_device.addItem("Apple Neural (MPS)", "mps")

        rform.addRow("RAM GB:", self.spin_ram)
        rform.addRow("Threads:", self.spin_threads)
        rform.addRow("Device:", self.combo_device)
        res_tab.setLayout(rform)
        tabs.addTab(res_tab, "System")

        # Neural tab
        ml_tab = QWidget()
        mform = QFormLayout()
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 1000)
        self.spin_epochs.setValue(20)

        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setDecimals(5)
        self.spin_lr.setRange(1e-6, 1.0)
        self.spin_lr.setValue(0.005)

        self.spin_commit = QDoubleSpinBox()
        self.spin_commit.setDecimals(3)
        self.spin_commit.setRange(0.0, 10.0)
        self.spin_commit.setValue(0.25)

        mform.addRow("Epochs:", self.spin_epochs)
        mform.addRow("LR:", self.spin_lr)
        mform.addRow("Beta (commit):", self.spin_commit)
        ml_tab.setLayout(mform)
        tabs.addTab(ml_tab, "Neural")

        layout.addWidget(tabs)

        # Training + validation
        layout.addWidget(QLabel("Step 1: Train / Validate on data"))

        train_row = QHBoxLayout()
        btn_select_val = QPushButton("Select Validation File")
        btn_select_val.clicked.connect(self.select_validation_file)
        self.btn_train = QPushButton("Train Session")
        self.btn_train.setStyleSheet("background-color: #d29922; height: 35px;")
        self.btn_train.clicked.connect(self.train_session)
        self.btn_validate = QPushButton("Validate Ratio")
        self.btn_validate.setStyleSheet("background-color: #1f6feb; height: 35px;")
        self.btn_validate.clicked.connect(self.validate_ratio)
        train_row.addWidget(btn_select_val)
        train_row.addWidget(self.btn_train)
        train_row.addWidget(self.btn_validate)
        layout.addLayout(train_row)

        # Export / Import
        layout.addWidget(QLabel("Step 2: Physical Export / Import (.pico)"))

        arow = QHBoxLayout()
        btn_exp = QPushButton("Export .pico")
        btn_exp.setStyleSheet("background-color: #238636; height: 35px;")
        btn_exp.clicked.connect(self.export_pico)
        btn_imp = QPushButton("Import .pico")
        btn_imp.setStyleSheet("background-color: #1f6feb; height: 35px;")
        btn_imp.clicked.connect(self.import_pico)
        arow.addWidget(btn_exp)
        arow.addWidget(btn_imp)
        layout.addLayout(arow)

        # Progress
        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        self.setLayout(layout)

    # -----------------------------
    # Model controls
    # -----------------------------
    def reset_model(self):
        self.shared_codec = SphericalConvCodec()
        self.lbl_state.setText("RAM State: Fresh")
        self.console.append("> Model reset to fresh weights.")

    def load_weights(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Weights", "", "*.pico_model"
        )
        if path:
            self.shared_codec.load_state_dict(torch.load(path, map_location="cpu"))
            self.lbl_state.setText("RAM State: Weights Loaded")
            self.console.append(f"> Weights loaded from: {os.path.basename(path)}")

    def save_weights(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Weights", "", "*.pico_model"
        )
        if path:
            torch.save(self.shared_codec.state_dict(), path)
            self.console.append(f"> Weights saved to: {os.path.basename(path)}")

    # -----------------------------
    # Training + validation
    # -----------------------------
    def select_validation_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select File for Training/Validation")
        if path:
            self.selected_file_for_validation = path
            self.console.append(f"> Target: {os.path.basename(path)}")

    def train_session(self):
        if not self.selected_file_for_validation:
            QMessageBox.warning(self, "No File", "Select a file for training/validation first.")
            return

        self.btn_train.setEnabled(False)
        self.btn_validate.setEnabled(False)

        config = {
            'epochs': self.spin_epochs.value(),
            'ram_gb': self.spin_ram.value(),
            'threads': self.spin_threads.value(),
            'device': self.combo_device.currentData(),
            'lr': self.spin_lr.value(),
            'commit': self.spin_commit.value()
        }

        self.worker = TrainingWorker(self.shared_codec, self.selected_file_for_validation, config)
        self.worker.log_stamp.connect(self.console.append)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(self._train_finished)
        self.worker.start()

    def _train_finished(self):
        self.btn_train.setEnabled(True)
        self.btn_validate.setEnabled(True)
        self.lbl_state.setText("RAM State: Custom Trained")
        self.console.append("*** Training session complete ***")

    def validate_ratio(self):
        if not self.selected_file_for_validation:
            QMessageBox.warning(self, "No File", "Select a file for training/validation first.")
            return

        self.btn_train.setEnabled(False)
        self.btn_validate.setEnabled(False)

        self.val_worker = ValidationWorker(
            self.shared_codec,
            self.selected_file_for_validation,
            self.combo_device.currentData()
        )
        self.val_worker.log_stamp.connect(self.console.append)
        self.val_worker.finished.connect(self._validation_finished)
        self.val_worker.start()

    def _validation_finished(self, ratio):
        self.btn_train.setEnabled(True)
        self.btn_validate.setEnabled(True)
        self.console.append(f"*** FINAL VALIDATION: {ratio:.2f}x Ratio ***")
        QMessageBox.information(
            self, "Validation Report",
            f"Physical Compression Ratio: {ratio:.2f}x"
        )

    # -----------------------------
    # Export / Import
    # -----------------------------
    def export_pico(self):
        p_in, _ = QFileDialog.getOpenFileName(self, "Select File to Compress")
        if not p_in:
            return
        p_out, _ = QFileDialog.getSaveFileName(
            self, "Save Compressed File", p_in + ".pico", "*.pico"
        )
        if not p_out:
            return

        self.worker = CodecWorker(
            self.shared_codec,
            p_in,
            p_out,
            self.combo_device.currentData(),
            mode="export"
        )
        self.worker.log_stamp.connect(self.console.append)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(
            lambda: self.console.append("*** Export complete ***")
        )
        self.worker.start()

    def import_pico(self):
        p_in, _ = QFileDialog.getOpenFileName(
            self, "Select .pico File to Decompress", "", "*.pico"
        )
        if not p_in:
            return
        p_out, _ = QFileDialog.getSaveFileName(
            self, "Save Restored File", p_in.replace(".pico", ".restored")
        )
        if not p_out:
            return

        self.worker = CodecWorker(
            self.shared_codec,
            p_in,
            p_out,
            self.combo_device.currentData(),
            mode="import"
        )
        self.worker.log_stamp.connect(self.console.append)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.finished.connect(
            lambda: self.console.append("*** Import complete ***")
        )
        self.worker.start()

# ==========================================
# 5. ENTRY POINT
# ==========================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PicoCompressor()
    window.show()
    sys.exit(app.exec_())

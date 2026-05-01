"""ONNX model encryption, obfuscation, fingerprinting, and expiry protection."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import struct
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import onnx

logger = logging.getLogger(__name__)

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    _HAS_CRYPTOGRAPHY = True
except ImportError:
    _HAS_CRYPTOGRAPHY = False
    logger.warning(
        "The 'cryptography' package is not installed. "
        "AES encryption/decryption will be unavailable. "
        "Install it with: pip install cryptography"
    )


@dataclass
class ProtectionResult:
    success: bool
    output_path: str
    method: str
    original_size_mb: float
    protected_size_mb: float
    elapsed_s: float
    error: Optional[str] = None


class ModelProtector:
    """Encrypt, obfuscate, fingerprint, or add expiry to ONNX models."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

    def _load_model(self, path: Path | None = None) -> onnx.ModelProto:
        path = path or self.model_path
        return onnx.load(str(path))

    @staticmethod
    def _file_size_mb(path: Path) -> float:
        return path.stat().st_size / (1024 * 1024)

    @staticmethod
    def _derive_key(password: str, salt: bytes, key_len: int = 32) -> bytes:
        return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations=100_000, dklen=key_len)

    # ------------------------------------------------------------------
    # Encryption / Decryption
    # ------------------------------------------------------------------

    def encrypt(
        self,
        output_path: str | Path,
        password: str,
        algorithm: str = "aes-256-gcm",
    ) -> ProtectionResult:
        if not _HAS_CRYPTOGRAPHY:
            return ProtectionResult(
                success=False,
                output_path=str(output_path),
                method="encrypt",
                original_size_mb=self._file_size_mb(self.model_path),
                protected_size_mb=0.0,
                elapsed_s=0.0,
                error="cryptography package is not installed",
            )

        if algorithm != "aes-256-gcm":
            raise ValueError(f"Unsupported algorithm: {algorithm}. Only 'aes-256-gcm' is supported.")

        t0 = time.monotonic()
        output_path = Path(output_path)
        original_mb = self._file_size_mb(self.model_path)

        try:
            model = self._load_model()
            salt = os.urandom(16)
            key = self._derive_key(password, salt)
            aesgcm = AESGCM(key)

            weight_meta: list[dict[str, Any]] = []

            for init in model.graph.initializer:
                raw = init.raw_data if init.raw_data else np.array(init.float_data, dtype=np.float32).tobytes()
                nonce = os.urandom(12)
                ciphertext = aesgcm.encrypt(nonce, raw, None)

                tag = ciphertext[-16:]
                encrypted_data = ciphertext[:-16]

                weight_meta.append({
                    "name": init.name,
                    "nonce": nonce.hex(),
                    "tag": tag.hex(),
                    "original_size": len(raw),
                    "data_type": int(init.data_type),
                    "dims": list(init.dims),
                })

                init.ClearField("float_data")
                init.ClearField("double_data")
                init.ClearField("int32_data")
                init.ClearField("int64_data")
                init.raw_data = encrypted_data

            onnx.save(model, str(output_path))

            sidecar = output_path.with_suffix(".enc.json")
            sidecar.write_text(json.dumps({
                "algorithm": algorithm,
                "salt": salt.hex(),
                "kdf": "pbkdf2-sha256",
                "kdf_iterations": 100_000,
                "weights": weight_meta,
            }, indent=2))

            logger.info("Encrypted %d weights -> %s (sidecar: %s)", len(weight_meta), output_path, sidecar)

            return ProtectionResult(
                success=True,
                output_path=str(output_path),
                method="encrypt",
                original_size_mb=original_mb,
                protected_size_mb=self._file_size_mb(output_path),
                elapsed_s=time.monotonic() - t0,
            )
        except Exception as exc:
            logger.exception("Encryption failed")
            return ProtectionResult(
                success=False,
                output_path=str(output_path),
                method="encrypt",
                original_size_mb=original_mb,
                protected_size_mb=0.0,
                elapsed_s=time.monotonic() - t0,
                error=str(exc),
            )

    def decrypt(
        self,
        encrypted_path: str | Path,
        output_path: str | Path,
        password: str,
    ) -> ProtectionResult:
        if not _HAS_CRYPTOGRAPHY:
            return ProtectionResult(
                success=False,
                output_path=str(output_path),
                method="decrypt",
                original_size_mb=0.0,
                protected_size_mb=0.0,
                elapsed_s=0.0,
                error="cryptography package is not installed",
            )

        t0 = time.monotonic()
        encrypted_path = Path(encrypted_path)
        output_path = Path(output_path)
        enc_mb = self._file_size_mb(encrypted_path)

        try:
            sidecar = encrypted_path.with_suffix(".enc.json")
            meta = json.loads(sidecar.read_text())

            salt = bytes.fromhex(meta["salt"])
            key = self._derive_key(password, salt)
            aesgcm = AESGCM(key)

            weight_lookup = {w["name"]: w for w in meta["weights"]}
            model = self._load_model(encrypted_path)

            for init in model.graph.initializer:
                wm = weight_lookup[init.name]
                nonce = bytes.fromhex(wm["nonce"])
                tag = bytes.fromhex(wm["tag"])
                ciphertext_with_tag = init.raw_data + tag

                plaintext = aesgcm.decrypt(nonce, ciphertext_with_tag, None)

                init.raw_data = b""
                arr = np.frombuffer(plaintext, dtype=onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[wm["data_type"]])
                init.raw_data = arr.tobytes()
                init.dims[:] = wm["dims"]
                init.data_type = wm["data_type"]

            onnx.save(model, str(output_path))
            logger.info("Decrypted %d weights -> %s", len(weight_lookup), output_path)

            return ProtectionResult(
                success=True,
                output_path=str(output_path),
                method="decrypt",
                original_size_mb=enc_mb,
                protected_size_mb=self._file_size_mb(output_path),
                elapsed_s=time.monotonic() - t0,
            )
        except Exception as exc:
            logger.exception("Decryption failed")
            return ProtectionResult(
                success=False,
                output_path=str(output_path),
                method="decrypt",
                original_size_mb=enc_mb,
                protected_size_mb=0.0,
                elapsed_s=time.monotonic() - t0,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Obfuscation / Deobfuscation
    # ------------------------------------------------------------------

    def obfuscate(self, output_path: str | Path, seed: int | None = None) -> ProtectionResult:
        t0 = time.monotonic()
        output_path = Path(output_path)
        original_mb = self._file_size_mb(self.model_path)

        try:
            model = self._load_model()
            if seed is None:
                seed = int.from_bytes(os.urandom(8), "little")
                logger.info("Generated obfuscation seed: %d", seed)

            for init in model.graph.initializer:
                raw = init.raw_data if init.raw_data else np.array(init.float_data, dtype=np.float32).tobytes()
                rng = np.random.RandomState(seed)
                mask = rng.bytes(len(raw))
                xored = bytes(a ^ b for a, b in zip(raw, mask))

                init.ClearField("float_data")
                init.ClearField("double_data")
                init.ClearField("int32_data")
                init.ClearField("int64_data")
                init.raw_data = xored

            model.metadata_props.append(
                onnx.StringStringEntryProto(key="obfuscated", value="true")
            )

            onnx.save(model, str(output_path))
            logger.info("Obfuscated model -> %s (seed=%d)", output_path, seed)

            return ProtectionResult(
                success=True,
                output_path=str(output_path),
                method="obfuscate",
                original_size_mb=original_mb,
                protected_size_mb=self._file_size_mb(output_path),
                elapsed_s=time.monotonic() - t0,
            )
        except Exception as exc:
            logger.exception("Obfuscation failed")
            return ProtectionResult(
                success=False,
                output_path=str(output_path),
                method="obfuscate",
                original_size_mb=original_mb,
                protected_size_mb=0.0,
                elapsed_s=time.monotonic() - t0,
                error=str(exc),
            )

    def deobfuscate(self, obfuscated_path: str | Path, output_path: str | Path, seed: int) -> ProtectionResult:
        t0 = time.monotonic()
        obfuscated_path = Path(obfuscated_path)
        output_path = Path(output_path)
        enc_mb = self._file_size_mb(obfuscated_path)

        try:
            model = self._load_model(obfuscated_path)

            for init in model.graph.initializer:
                raw = init.raw_data
                rng = np.random.RandomState(seed)
                mask = rng.bytes(len(raw))
                restored = bytes(a ^ b for a, b in zip(raw, mask))
                init.raw_data = restored

            model.metadata_props[:] = [
                p for p in model.metadata_props if p.key != "obfuscated"
            ]

            onnx.save(model, str(output_path))
            logger.info("Deobfuscated model -> %s", output_path)

            return ProtectionResult(
                success=True,
                output_path=str(output_path),
                method="deobfuscate",
                original_size_mb=enc_mb,
                protected_size_mb=self._file_size_mb(output_path),
                elapsed_s=time.monotonic() - t0,
            )
        except Exception as exc:
            logger.exception("Deobfuscation failed")
            return ProtectionResult(
                success=False,
                output_path=str(output_path),
                method="deobfuscate",
                original_size_mb=enc_mb,
                protected_size_mb=0.0,
                elapsed_s=time.monotonic() - t0,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Fingerprinting
    # ------------------------------------------------------------------

    @staticmethod
    def _owner_hash(owner_id: str) -> bytes:
        return hashlib.sha256(owner_id.encode()).digest()

    def fingerprint(
        self,
        output_path: str | Path,
        owner_id: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        t0 = time.monotonic()
        output_path = Path(output_path)
        model = self._load_model()

        owner_digest = self._owner_hash(owner_id)
        fp_bits = "".join(format(b, "08b") for b in owner_digest)

        if metadata:
            meta_bytes = json.dumps(metadata, sort_keys=True).encode()
            meta_digest = hashlib.sha256(meta_bytes).digest()
            fp_bits += "".join(format(b, "08b") for b in meta_digest)

        bit_idx = 0
        total_bits = len(fp_bits)
        embedded = 0

        for init in model.graph.initializer:
            if init.data_type != onnx.TensorProto.FLOAT:
                continue
            raw = init.raw_data if init.raw_data else np.array(init.float_data, dtype=np.float32).tobytes()
            arr = np.frombuffer(raw, dtype=np.float32).copy()

            for i in range(len(arr)):
                if bit_idx >= total_bits:
                    break
                val_bytes = struct.pack("<f", arr[i])
                val_int = int.from_bytes(val_bytes, "little")
                val_int = (val_int & ~1) | int(fp_bits[bit_idx])
                arr[i] = struct.unpack("<f", val_int.to_bytes(4, "little"))[0]
                bit_idx += 1
                embedded += 1

            init.ClearField("float_data")
            init.raw_data = arr.tobytes()

            if bit_idx >= total_bits:
                break

        if bit_idx < total_bits:
            logger.warning(
                "Model too small to embed full fingerprint: %d/%d bits written",
                bit_idx,
                total_bits,
            )

        fp_hash = hashlib.sha256(fp_bits.encode()).hexdigest()

        model.metadata_props.append(
            onnx.StringStringEntryProto(key="fingerprint_hash", value=fp_hash)
        )
        model.metadata_props.append(
            onnx.StringStringEntryProto(key="fingerprint_bits", value=str(total_bits))
        )
        if metadata:
            model.metadata_props.append(
                onnx.StringStringEntryProto(key="fingerprint_metadata", value=json.dumps(metadata, sort_keys=True))
            )

        onnx.save(model, str(output_path))
        logger.info("Fingerprinted model (%d bits embedded) -> %s", embedded, output_path)

        return fp_hash

    def verify_fingerprint(self, model_path: str | Path, owner_id: str) -> tuple[bool, Optional[Dict[str, str]]]:
        model_path = Path(model_path)
        model = self._load_model(model_path)

        meta_lookup = {p.key: p.value for p in model.metadata_props}
        stored_hash = meta_lookup.get("fingerprint_hash")
        if not stored_hash:
            logger.warning("No fingerprint found in model metadata")
            return False, None

        total_bits = int(meta_lookup.get("fingerprint_bits", "256"))
        stored_metadata_json = meta_lookup.get("fingerprint_metadata")
        stored_metadata = json.loads(stored_metadata_json) if stored_metadata_json else None

        owner_digest = self._owner_hash(owner_id)
        expected_bits = "".join(format(b, "08b") for b in owner_digest)

        extracted_bits: list[str] = []
        for init in model.graph.initializer:
            if init.data_type != onnx.TensorProto.FLOAT:
                continue
            raw = init.raw_data if init.raw_data else np.array(init.float_data, dtype=np.float32).tobytes()
            arr = np.frombuffer(raw, dtype=np.float32)

            for i in range(len(arr)):
                if len(extracted_bits) >= total_bits:
                    break
                val_bytes = struct.pack("<f", arr[i])
                val_int = int.from_bytes(val_bytes, "little")
                extracted_bits.append(str(val_int & 1))

            if len(extracted_bits) >= total_bits:
                break

        extracted = "".join(extracted_bits)
        owner_portion = extracted[: len(expected_bits)]
        match = owner_portion == expected_bits

        logger.info("Fingerprint verification: %s", "MATCH" if match else "MISMATCH")
        return match, stored_metadata

    # ------------------------------------------------------------------
    # Expiry
    # ------------------------------------------------------------------

    def add_expiry(
        self,
        output_path: str | Path,
        expiry_date: str | datetime,
        check_code: Optional[str] = None,
    ) -> ProtectionResult:
        t0 = time.monotonic()
        output_path = Path(output_path)
        original_mb = self._file_size_mb(self.model_path)

        try:
            model = self._load_model()

            if isinstance(expiry_date, datetime):
                expiry_str = expiry_date.isoformat()
            else:
                datetime.fromisoformat(expiry_date)
                expiry_str = expiry_date

            model.metadata_props.append(
                onnx.StringStringEntryProto(key="expiry_date", value=expiry_str)
            )

            onnx.save(model, str(output_path))

            wrapper_path = output_path.with_suffix(".loader.py")
            wrapper_code = check_code or _DEFAULT_LOADER_TEMPLATE.format(
                model_path=str(output_path),
                expiry_date=expiry_str,
            )
            wrapper_path.write_text(wrapper_code)

            logger.info("Added expiry %s -> %s (loader: %s)", expiry_str, output_path, wrapper_path)

            return ProtectionResult(
                success=True,
                output_path=str(output_path),
                method="add_expiry",
                original_size_mb=original_mb,
                protected_size_mb=self._file_size_mb(output_path),
                elapsed_s=time.monotonic() - t0,
            )
        except Exception as exc:
            logger.exception("add_expiry failed")
            return ProtectionResult(
                success=False,
                output_path=str(output_path),
                method="add_expiry",
                original_size_mb=original_mb,
                protected_size_mb=0.0,
                elapsed_s=time.monotonic() - t0,
                error=str(exc),
            )


_DEFAULT_LOADER_TEMPLATE = '''\
"""Auto-generated loader with expiry check for a protected ONNX model."""

import sys
from datetime import datetime

import onnx
import onnxruntime as ort

MODEL_PATH = r"{model_path}"
EXPIRY_DATE = "{expiry_date}"


def load_model(model_path: str = MODEL_PATH) -> ort.InferenceSession:
    expiry = datetime.fromisoformat(EXPIRY_DATE)
    if datetime.now() > expiry:
        raise RuntimeError(
            f"Model licence expired on {{EXPIRY_DATE}}. Contact the model owner."
        )

    meta = {{p.key: p.value for p in onnx.load(model_path).metadata_props}}
    if "expiry_date" in meta:
        stored = datetime.fromisoformat(meta["expiry_date"])
        if datetime.now() > stored:
            raise RuntimeError(
                f"Model licence expired on {{meta['expiry_date']}}. Contact the model owner."
            )

    return ort.InferenceSession(model_path)


if __name__ == "__main__":
    session = load_model()
    print(f"Model loaded successfully. Inputs: {{[i.name for i in session.get_inputs()]}}")
'''


# ------------------------------------------------------------------
# Top-level convenience function
# ------------------------------------------------------------------


def protect_model(
    model_path: str,
    output_path: str,
    method: str = "encrypt",
    password: Optional[str] = None,
    **kwargs: Any,
) -> ProtectionResult:
    """CLI-friendly entry point for model protection.

    Args:
        model_path: Path to the source ONNX model.
        output_path: Destination path for the protected model.
        method: One of 'encrypt', 'decrypt', 'obfuscate', 'deobfuscate',
                'fingerprint', 'add_expiry'.
        password: Required for encrypt/decrypt.
        **kwargs: Forwarded to the underlying method.
    """
    protector = ModelProtector(model_path)

    if method == "encrypt":
        if not password:
            raise ValueError("Password is required for encryption")
        return protector.encrypt(output_path, password, **kwargs)

    if method == "decrypt":
        if not password:
            raise ValueError("Password is required for decryption")
        encrypted_path = kwargs.pop("encrypted_path", model_path)
        return protector.decrypt(encrypted_path, output_path, password)

    if method == "obfuscate":
        return protector.obfuscate(output_path, **kwargs)

    if method == "deobfuscate":
        seed = kwargs.pop("seed", None)
        if seed is None:
            raise ValueError("Seed is required for deobfuscation")
        obfuscated_path = kwargs.pop("obfuscated_path", model_path)
        return protector.deobfuscate(obfuscated_path, output_path, seed)

    if method == "fingerprint":
        owner_id = kwargs.pop("owner_id", None)
        if not owner_id:
            raise ValueError("owner_id is required for fingerprinting")
        fp_hash = protector.fingerprint(output_path, owner_id, **kwargs)
        return ProtectionResult(
            success=True,
            output_path=str(output_path),
            method="fingerprint",
            original_size_mb=protector._file_size_mb(protector.model_path),
            protected_size_mb=protector._file_size_mb(Path(output_path)),
            elapsed_s=0.0,
        )

    if method == "add_expiry":
        expiry_date = kwargs.pop("expiry_date", None)
        if not expiry_date:
            raise ValueError("expiry_date is required for add_expiry")
        return protector.add_expiry(output_path, expiry_date, **kwargs)

    raise ValueError(f"Unknown protection method: {method}")

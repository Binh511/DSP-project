"""Validate extension uploads before creating any persisted job files."""
from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import MAX_UPLOAD_BYTES


@dataclass(frozen=True)
class ValidatedUpload:
    filename: str
    suffix: str
    content: bytes
    mode: str
    device: str
    ocr: str


def _filename(payload: dict[str, Any]) -> tuple[str, str]:
    filename = payload.get('filename')
    if (not isinstance(filename, str) or not filename.strip() or len(filename) > 180
            or any(char in filename for char in '/\\:')
            or any(ord(char) < 32 for char in filename)):
        raise ValueError('Tên tài liệu không hợp lệ; chỉ gửi tên file, không gửi đường dẫn.')
    suffix = Path(filename).suffix.lower()
    if suffix not in ('.pdf', '.md', '.txt'):
        raise ValueError('Chỉ hỗ trợ tài liệu PDF, Markdown (.md) hoặc text (.txt).')
    return filename, suffix


def _options(payload: dict[str, Any]) -> dict[str, str]:
    options = {}
    for key, default, choices in (
            ('mode', 'full', ('full', 'fast')),
            ('device', 'auto', ('auto', 'cpu', 'cuda')),
            ('ocr', 'auto', ('auto', 'none', 'always'))):
        value = payload.get(key, default)
        if not isinstance(value, str) or value not in choices:
            raise ValueError(f'{key} phải là một trong: {", ".join(choices)}.')
        options[key] = value
    return options


def _content(payload: dict[str, Any], suffix: str) -> bytes:
    encoded = payload.get('content_base64')
    if not isinstance(encoded, str) or len(encoded) > 4 * ((MAX_UPLOAD_BYTES + 2) // 3):
        raise ValueError('Nội dung tài liệu thiếu hoặc vượt quá giới hạn 20 MiB.')
    try:
        content = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError('Nội dung tài liệu không phải base64 hợp lệ.') from error
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise ValueError('Tài liệu phải có nội dung và không vượt quá 20 MiB.')
    if suffix in ('.md', '.txt'):
        try:
            if not content.decode('utf-8-sig').strip():
                raise ValueError('Tài liệu text không có nội dung đọc được.')
        except UnicodeDecodeError as error:
            raise ValueError('Tài liệu Markdown/text cần mã hóa UTF-8.') from error
    return content


def validate_upload(payload: object) -> ValidatedUpload:
    """Decode a complete, supported document and its strict processing options."""
    if not isinstance(payload, dict):
        raise ValueError('Nội dung yêu cầu phải là một JSON object.')
    filename, suffix = _filename(payload)
    options = _options(payload)
    return ValidatedUpload(filename, suffix, _content(payload, suffix), **options)

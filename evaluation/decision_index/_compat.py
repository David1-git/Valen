"""Keep the pinned upstream CLI usable on Python 3.10. / 兼容 Python 3.10。"""
import hashlib


def ensure_file_digest():
    if hasattr(hashlib, "file_digest"):
        return

    def file_digest(stream, digest):
        result = hashlib.new(digest) if isinstance(digest, str) else digest()
        while block := stream.read(1024 * 1024):
            result.update(block)
        return result

    hashlib.file_digest = file_digest

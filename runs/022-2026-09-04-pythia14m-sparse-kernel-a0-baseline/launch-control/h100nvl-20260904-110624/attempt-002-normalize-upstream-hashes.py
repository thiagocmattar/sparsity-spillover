"""Attempt-local repair for Windows CRLF hashes in Run 022's upstream preflight.

The pinned upstream commit is unchanged.  The original hashes were calculated
from a Windows checkout with CRLF line endings; a clean Linux checkout contains
the byte-identical source text with LF line endings.  This replaces only those
three expected byte hashes in the remote extracted copy of config.yaml.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


CONFIG = Path(
    "/workspace/sparsity-spillover/"
    "runs/022-2026-09-04-pythia14m-sparse-kernel-a0-baseline/config.yaml"
)
REPLACEMENTS = {
    "703c476470138f7e6059ab94e0d1b5669e58c6e6b08fa89d46451f69b6b2b031":
        "527529e33942021c629713ca80e29af336e28155a4787a7f9ed868a406daf389",
    "499e10784908350219e8ab007c4128170c2e3449999c094cc6a6e888a959f9fe":
        "7734162517bcfff4319ccac5cc97abd619262d761c5a61c1058d61b6f3673daa",
    "7a8b6d47bfdcb07e2a579761c3f08a7e52aba611e385552512847c7573994926":
        "170274bb248f611349dd49dfcc9ef36d9a21827f016515d613c20dac3044bbe9",
    "c01fda6953daabf1243ab5c2a2e6da8d0f4400c6cf490595884548dc44389be6":
        "2f96866ee9227eefaf4707524a4df0f51a7a44ef7275d9a24a7e6a3ba6b302c4",
    "05ccbc11db17d647420e7e8f49cbcbe8b338e25cd500b623bfb7d68aaf9a9f98":
        "ffc3a7517d56b3f1fa3ab0f9e1a1be6c570005bd72250c4a67107415eec65698",
    "aa450d5d351abc4eb3229c9136292fabf7533b39cfb24b177a603127b1875ff4":
        "916d5d32e5ea632b23f6f1ea3afd35030ae501004aaa1daf18942963ebb9d704",
    "bbc6a1ad7155d44fd35e8f0b26abaedcb8e4417856d4ad2f387eefe02963ca35":
        "2721ff05f39ac348c885734b23ac680da9f0781c5d06e4ca1678d9af78719961",
    "39f1137f710da20580ac47cb5dba72cb51f41a8599e9abdf0130f36df6db71c7":
        "6cef5514c5dad7482461f38033e214821149ffe9aff622aef7147c51eec60145",
}


text = CONFIG.read_text(encoding="utf-8")
for windows_crlf_sha256, linux_lf_sha256 in REPLACEMENTS.items():
    occurrences = text.count(windows_crlf_sha256)
    if occurrences == 1:
        text = text.replace(windows_crlf_sha256, linux_lf_sha256)
    elif occurrences == 0 and text.count(linux_lf_sha256) == 1:
        continue
    else:
        raise RuntimeError(
            f"Expected one occurrence of {windows_crlf_sha256}, found {occurrences}."
        )

with CONFIG.open("w", encoding="utf-8", newline="\n") as handle:
    handle.write(text)

print(f"updated={CONFIG}")
print(f"config_sha256={hashlib.sha256(CONFIG.read_bytes()).hexdigest()}")

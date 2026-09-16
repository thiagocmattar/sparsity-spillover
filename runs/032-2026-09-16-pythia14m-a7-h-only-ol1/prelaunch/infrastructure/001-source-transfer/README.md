# Source transfer retry, before scientific execution

The first full-history source bundle was 97,421,856 bytes, SHA-256
7137b57c432d27ffa3713aa02e03503c01d5caa3f6980e17b1cae9a1ea34db0c,
from actual source commit fb72ee91e6192d280a7ed1533d822a779926a698.
At 20:12 UTC only about 14 MiB had reached the first Pod. All five scoped
controller upload processes were stopped before any checkout/setup or science.
The partial remote bundle files were replaced with a 95,973-byte deployment
subset, an actual new local Git commit 772c7ccd07122744e444307089fe5bdf2ff70f5a.
No historical commit/date was fabricated. Every selected file came from the
named source commit. No dataset or model weights were included in Git.

Before science, a Windows CRLF-versus-Linux LF discrepancy in the historical
Run 013 comparator JSON motivated explicit text normalization in Run 032's
source inventory. Original Run 013 bytes were left unchanged. Full tests passed
253/253 again. Source commit 2a546dc01c3475ec5c0e60d5193523ccdb7ded15 and
actual deployment commit 240eb34490b40056a3d1dfe86ad0cc4de2f882c7 supersede
the initial source packet. All five clean remote checkouts were updated before
preflight/science, and all 36 normalized code-inventory files reconcile.
The final source receipt identifies the 98,073-byte bundle and its hash.

Controller SSH parsing was corrected to use the observed `runpodctl ssh info`
ip/port fields. Early connection attempts created no remote scientific state.

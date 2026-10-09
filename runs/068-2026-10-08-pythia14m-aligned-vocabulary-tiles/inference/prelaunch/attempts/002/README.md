# Infrastructure upload retry002

The initial benchmark upload used Paramiko's default SFTP window/packet sizes
and reached only about0.055 MiB/s. Its two scoped local Python processes were
stopped before extraction or any model process started. Retry002 resumes the
same remote partial archive, using the64 MiB window,1 MiB packet and8 MiB socket
send buffer that transferred the training inputs at about10 MiB/s.

The original81-file inventory,157,475,311-byte packet, checkpoints, runtime
source, GPU, deadlines and all scientific parameters are unchanged. Archive
SHA256 must match before extraction, and every extracted member is verified
before setup or benchmark execution. The published source commit remains
`35a4c2b7d470191f52e439df961ab77c9b5daec2`.

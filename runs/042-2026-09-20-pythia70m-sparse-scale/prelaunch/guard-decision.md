# Credential-free deadline enforcement

Automatic approval review rejected bootstrap_001.py before execution because
its original remote stop guard would upload the local RunPod API key to the
pod. No credential was transferred. The bootstrap was revised to eliminate
all credential reads/uploads. The existing scoped local guard stops the Pod
at the absolute deadline, and each remote scientific command has a local
timeout ending before that deadline. Monitoring and verified early teardown
remain the primary budget controls. No permission request is needed for this
safer alternative. The inherited unused12_deadline_guard.py is not launched.

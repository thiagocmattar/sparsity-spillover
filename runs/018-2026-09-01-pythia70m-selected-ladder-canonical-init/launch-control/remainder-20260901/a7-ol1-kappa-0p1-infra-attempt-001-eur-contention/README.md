# A7-OL1 kappa 0.1 infrastructure attempt 001

- Pod: `n1rfg6vn09box1`
- GPU/location: NVIDIA H200, secure cloud, EUR-IS-4
- Scientific attempt: `011-20260901-164011-cbff1de2`
- Source commit: `7f26226b18a3191446ddb37785d4faaa5814c010`
- Outcome: infrastructure retry; not scientific evidence
- Reason: after all inputs and the runtime passed exact hash verification, the worker reached step 5 at 137,215 tokens/s (15.284 seconds/step). Concurrent EUR-IS-4 workers also lost throughput, making this location materially slower than the sentinel and non-EUR H200 observations and projecting completion outside the cost guard.
- Action: retain the small partial control/manifest/event files, delete the Pod, and restart the unchanged condition on a non-EUR H200.

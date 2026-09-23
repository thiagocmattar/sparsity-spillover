"""Pin the follow-up registry and source identity for unchanged measurement code."""
import runpy,sys
import bootstrap
import primitives_v3
import compiler_evidence_v3
import work_counters_v3
from support import RUN
sys.modules['primitives']=primitives_v3
sys.modules['compiler_evidence']=compiler_evidence_v3
sys.modules['work_counters']=work_counters_v3
script=sys.argv.pop(1)
assert script in ('03_operator_checks.py','04_screen.py','06_benchmark.py','08_diagnostics.py')
sys.argv[0]=str(RUN/script)
runpy.run_path(str(RUN/script),run_name='__main__')

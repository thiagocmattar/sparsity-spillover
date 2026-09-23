"""Add CTA-local payload reads and writes to the existing independent count oracle."""
from work_counters import Work as PreviousWork

class Work(PreviousWork):
    def add(self,key,op,x):
        super().add(key,op,x)
        if op is None or op.spec.get('strategy')!='cta_compact':return
        row=self.rows[key]
        used=x.numel() if op.spec.get('no_skip',False) else int((x!=0).sum())
        row['activation_payload_values_read']=row.get('activation_payload_values_read',0)+used
        row['local_index_words_written']=row.get('local_index_words_written',0)+used
        row['local_value_words_written']=row.get('local_value_words_written',0)+used

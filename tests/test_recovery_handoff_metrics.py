"""A brief upright detection must not count as durable recovery."""
from pathlib import Path
import sys
import numpy as np


def test_durable_gate_rejects_refalls_and_invalid_trials():
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    sys.path.insert(0,str(scripts))
    try:
        from check_recovery_handoff import summarize
    finally:
        sys.path.pop(0)
    shape=(450,128)
    data={k:np.zeros(shape,dtype=bool) for k in ['handoff','entered','invalid','upright']}
    data['position']=np.zeros((*shape,3))
    data['handoff'][59,:]=True
    data['upright'][30:,:]=True
    data['upright'][100,0]=False
    data['entered'][100,0]=True
    data['invalid'][100:,1]=True
    result=summarize(data)
    assert result['football_handoff']['stand_within_5s_and_upright_next_3s']==62
    assert result['football_handoff']['refall_within_3s']==1
    assert result['football_handoff']['numerically_invalid_trials']==1
    assert result['keep_recovery']['stand_within_5s_and_upright_next_3s']==64

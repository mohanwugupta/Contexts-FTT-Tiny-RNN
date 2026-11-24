"""Single intervention: unchanged native outer step, then actual-floor projection."""
from .common import *
from .protocol import projection
from experiments.high_scale_meta_progression.runtime import meta_step as native_step
from experiments.high_scale_constrained_meta.runtime import pack,scale_metrics
from experiments.high_scale_constrained_meta.initialization import groups,rms

def advance(m,o,e,arm,condition,k,f):
 before=pack(m,o);initial_rms={g:rms(x) for g,x in groups(m).items()};native=native_step(m,o,e,arm,k);unprojected=pack(m,o)
 if not np.isfinite(unprojected).all():raise FloatingPointError('Nonfinite unprojected native outer state')
 rows=projection(m,f,o,condition);after=pack(m,o)
 if not np.isfinite(after).all():raise FloatingPointError('Nonfinite post-projection state')
 np.testing.assert_array_equal(after[1:],unprojected[1:])
 if condition=='U':np.testing.assert_array_equal(after,unprojected)
 for row in rows:row['before_outer_rms']=initial_rms[row['group']]
 return dict(native=native,before=before,unprojected=unprojected,after=after,projection=rows,outer_displacement=float(np.linalg.norm(unprojected[0].astype(float)-before[0])),projection_displacement=float(np.linalg.norm(after[0].astype(float)-unprojected[0])),net_displacement=float(np.linalg.norm(after[0].astype(float)-before[0])))

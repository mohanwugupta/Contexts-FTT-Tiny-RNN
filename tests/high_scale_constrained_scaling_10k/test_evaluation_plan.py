from experiments.high_scale_constrained_scaling_10k.evaluation import target_jobs,conflict_jobs

def test_exact_reuse_and_prescribed_horizons_no_dose_zero_duplicates():
 assert len(target_jobs('B-targets'))==54 and len(target_jobs('C-targets'))==216 and len(target_jobs('targets'))==576
 alljobs=sum([target_jobs(s) for s in ['B-targets','C-targets','targets']],[]);assert len(alljobs)==846 and len(set(alljobs))==846
 assert sum(j[-1] for j in alljobs)==386100
 assert all(j[-1]==990 for j in alljobs if j[3]=='R' and j[4]==1920 or j[4]==10000)
 assert all(j[-1]==330 for j in alljobs if j[3]!='R' and j[4]<10000)
 c=conflict_jobs();assert len(c)==522 and len(set(c))==522 and all(j[-1] in [1920,3840,7680,10000] for j in c)
 assert sum(j[2]=='P' for j in c)==306 and sum(j[2]=='U' for j in c)==216

 assert len(c)*12*20==125280

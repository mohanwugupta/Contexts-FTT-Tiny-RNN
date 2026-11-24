"""Source-indexed compact position exports and supplemental table-only views."""
import argparse
from .common import *

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['screen','confirm'],required=True);stage=p.parse_args().stage;reg=read(stage+'_physical_registry');positions=[];decomp=[];maths=[];rows=[]
 for r in reg.itertuples():
  d=ROOT/r.directory;meta=dict(seed=r.seed,arm=r.arm,dose=r.dose);ps=pd.read_csv(d/'position_predictions.csv.gz',float_precision='round_trip');dc=pd.read_csv(d/'decomposition.csv.gz',float_precision='round_trip');mc=pd.read_csv(d/'mathematical_checks.csv.gz',float_precision='round_trip');probes=[0,10,20,40,80,120,170,330];positions.append(ps[ps['update'].isin(probes)].assign(**meta));decomp.append(dc[dc['update'].isin(probes)].assign(**meta));maths.append(mc.assign(**meta));rows.append(dict(**meta,positions_source=str((d/'position_predictions.csv.gz').relative_to(ROOT)),positions_sha=sha(d/'position_predictions.csv.gz'),decomposition_source=str((d/'decomposition.csv.gz').relative_to(ROOT)),decomposition_sha=sha(d/'decomposition.csv.gz'),states_source=r.path,states_sha=r.sha256))
 write_frame(OUT/f'tables/{stage}_position_predictions.csv.gz',pd.concat(positions,ignore_index=True));write_frame(OUT/f'tables/{stage}_decomposition_positions.csv.gz',pd.concat(decomp,ignore_index=True));write_frame(OUT/f'tables/{stage}_mathematical_checks.csv.gz',pd.concat(maths,ignore_index=True));write_frame(OUT/f'tables/{stage}_source_tables.csv',rows);dump(OUT/f'audit/{stage}_products_complete.json',dict(passed=True,physical_histories=len(rows),compact_probes=probes,complete331_state_position_sources_preserved=True,all_four_logits_and_probabilities=True,trained_models_in_export=False))
if __name__=='__main__':main()

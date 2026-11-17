#!/usr/bin/env python3
"""Create the post-freeze held-out mapping audit without inventing a midpoint."""

from pathlib import Path
import argparse,json,hashlib
import pandas as pd


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--selection",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    raw=args.selection.read_bytes(); payload=json.loads(raw)
    if "payload_sha256" not in payload: raise ValueError("Selection artifact is not frozen/hash-recorded.")
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for model in ["Lump","Cross","Split","M3a","M3b","M4","M5"]:
        rows.append({"model":model,"probe_context":"Mixed","prediction":float("nan"),
            "status":"undefined without an additional context-generalization rule",
            "midpoint_assumed":False,"parameters_refit":False,
            "selection_file_sha256":hashlib.sha256(raw).hexdigest()})
    pd.DataFrame(rows).to_csv(args.output/"heldout_mixed_predictions.csv",index=False)
    pd.DataFrame(columns=pd.DataFrame(rows).columns).to_csv(args.output/"heldout_novel_predictions.csv",index=False)
    print(f"Held-out mapping audit written to {args.output}",flush=True)


if __name__=="__main__": main()

"""Multiobjective ancestry and retained tradeoffs. No scalar research reward."""
import math,json

def frontier(rows,directions):
    good=[r for r in rows if all(isinstance(r.get(k),(float,int)) and math.isfinite(r[k]) for k in directions)]
    def dominates(a,b):
        diffs=[(a[k]-b[k])*(1 if d=='max' else -1) for k,d in directions.items()]
        return all(x>=0 for x in diffs) and any(x>0 for x in diffs)
    return sorted(r['id'] for r in good if not any(dominates(q,r) for q in good if q['id']!=r['id']))

def export(store):
    return dict(nodes=[dict(r,proposal=json.loads(r['proposal'])) for r in store.nodes()],events=[dict(r,data=json.loads(r['data'])) for r in store.db.execute('SELECT * FROM events ORDER BY n')],results=[dict(id=r['id'],value=json.loads(r['value'])) for r in store.db.execute('SELECT * FROM results')])

"""Provider event validation reuses the repaired native assessor unchanged."""
from experiments.research_search_pilot_v2.events import assess as native_assess

def assess(events,returncode,reservation,timed_out=False):
 return native_assess(dict(returncode=returncode,timed_out=timed_out,cancelled=False),events,reservation)

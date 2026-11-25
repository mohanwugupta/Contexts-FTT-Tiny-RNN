import hashlib,json,re,time
from pathlib import Path
_parents=Path(__file__).resolve().parents
ROOT=_parents[min(2,len(_parents)-1)]
HOSTS={'chatgpt.com','auth.openai.com'}
ROLES={'coordinator','architecture','learning','experience','initialization','critic','specialist_initialization','specialist_learning','coordinator_select','coordinator_review'}
DISABLED=['multi_agent','multi_agent_v2','apps','plugins','hooks','browser_use','browser_use_external','browser_use_full_cdp_access','computer_use','image_generation','in_app_browser','in_app_local_automation','shell_tool','unified_exec','workspace_dependencies','memories','skill_search','skill_mcp_dependency_install','remote_plugin','code_mode_host','code_mode','sleep_tool','view_image','unbounded_connection_retries']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');p.chmod(0o644)
def authority(s):
 if not isinstance(s,str) or not re.fullmatch(r'[a-z.]+:443',s):raise ValueError('CONNECT authority rejected')
 host=s[:-4]
 if host not in HOSTS:raise ValueError('CONNECT destination rejected')
 return host,443

def remaining(start,ceiling,now=None):
 now=time.time() if now is None else now
 if type(start) not in [int,float] or start>now or type(ceiling) not in [int,float] or ceiling<=0:raise ValueError('Existing valid rehearsal clock required; never reset it')
 left=start+ceiling-now
 if left<=0:raise ValueError('Existing rehearsal deadline elapsed')
 return left

def auth_bytes(data):
 a=json.loads(data)
 if a.get('OPENAI_API_KEY') or a.get('CODEX_API_KEY') or a.get('auth_mode') not in [None,'chatgpt']:raise ValueError('ChatGPT subscription authentication only; no API fallback')
 if not isinstance(a.get('tokens'),dict) or not a['tokens'].get('access_token'):raise ValueError('File-based ChatGPT authentication unavailable')
 return data

def redact(text,data):
 def strings(v):
  if isinstance(v,dict):return [x for a in v.values() for x in strings(a)]
  if isinstance(v,list):return [x for a in v for x in strings(a)]
  return [v] if isinstance(v,str) and len(v)>=8 else []
 for value in strings(json.loads(data or b'{}')):text=text.replace(value,'[REDACTED_AUTH]')
 return text

def codex_argv(mode):
 values={'forced_login_method':'chatgpt','cli_auth_credentials_store':'file','model_provider':'openai','sqlite_home':'/tmp/state','log_dir':'/tmp/logs','analytics.enabled':False,'feedback.enabled':False,'check_for_update_on_startup':False,'mcp_servers':{},'project_doc_max_bytes':0,'model_reasoning_effort':'low','web_search':'disabled'}
 args=['/runtime/codex']
 if mode=='exec':args+=['exec','--ignore-user-config','--ignore-rules','--ephemeral','--skip-git-repo-check','--sandbox','read-only','--json','-C','/work','--output-schema','/request/schema.json']
 elif mode=='quota':args+=['app-server','--stdio']
 elif mode=='login':args+=['login','status']
 else:raise ValueError('Unregistered CLI mode')
 for k,v in values.items():args+=['-c',k+'='+('{}' if v=={} else json.dumps(v,separators=(',',':')))]
 for k in DISABLED:args+=['--disable',k]
 return args+(['-'] if mode=='exec' else [])

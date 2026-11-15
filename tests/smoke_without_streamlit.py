"""Offline logic/figure exercise using a minimal Streamlit stub.
This does NOT replace a browser/UI smoke test with actual Streamlit installed.
"""
import sys,types,runpy,os
from pathlib import Path

class Context:
    def __enter__(self): return self
    def __exit__(self,*args): return False
    def __getattr__(self,key): return getattr(stub,key)

class Stub(types.ModuleType):
    def __init__(self):
        super().__init__('streamlit')
        self.sidebar=Context(); self.figure_count=0
    def cache_data(self,**kwargs): return lambda f:f
    def columns(self,n,*args,**kwargs):
        if isinstance(n,int):return [Context() for _ in range(n)]
        return [Context() for _ in n]
    def tabs(self,labels): return [Context() for _ in labels]
    def expander(self,*a,**k):return Context()
    def radio(self,label,options,index=0,**kwargs):return options[-1] if os.environ.get("FIRELENS_SMOKE_CUSTOM")=="1" else options[index]
    def selectbox(self,label,options,index=0,**kwargs):return options[index]
    def select_slider(self,label,options,value=None,**kwargs):return options[-1] if os.environ.get("FIRELENS_SMOKE_CUSTOM")=="1" else (value if value is not None else options[0])
    def slider(self,label,*args,**kwargs):return kwargs.get('value',args[2] if len(args)>2 else None)
    def checkbox(self,label,value=False,**kwargs):return True if os.environ.get("FIRELENS_SMOKE_CUSTOM")=="1" else value
    def plotly_chart(self,fig,**kwargs):
        d=fig.to_plotly_json()
        assert d.get('data')
        self.figure_count+=1
    def stop(self):raise RuntimeError('stop() called')
    def error(self,msg,*args,**kwargs):raise RuntimeError('Streamlit error: '+str(msg))
    def __getattr__(self,name):return lambda *args,**kwargs:None
stub=Stub()
sys.modules['streamlit']=stub
path=Path(__file__).parents[1]/'app.py'
runpy.run_path(str(path),run_name='__main__')
assert stub.figure_count>=6,stub.figure_count
print('PASS: V9 application logic completed with',stub.figure_count,'plotly figures (mock UI only).')

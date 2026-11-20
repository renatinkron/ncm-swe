import ast,codeop,importlib,json,os,pickle,sys,time,types
import numpy as np
import torch
hydra=types.ModuleType("hydra")
class _U:
    @staticmethod
    def get_class(p):
        m,_,n=p.rpartition(".")
        return getattr(importlib.import_module(m),n)
hydra.utils=_U;sys.modules["hydra"]=hydra
ROOT="/repo/vendor/ncm-upstream";sys.path.insert(0,ROOT);sys.path.insert(0,"/repo/src");os.chdir(ROOT)
from omegaconf import OmegaConf
from tokenization.bpe_tokenizer import BPETokenizer
from tokenization.vqvae_tokenizer import VQVAETokenizer
from ncm_swe.model.generator import LevelGenerator
torch.set_num_threads(int(os.environ.get("THREADS","4")))
CK="/work/artifacts/ncm_official";OUTD="/work/artifacts/ncm_code"
cfg=OmegaConf.load(ROOT+"/config/eval/tinystories.yaml")
cfg.tokenizer.vqvae.checkpoint_path=os.path.join(CK,cfg.checkpoints.tokenizer)
cfg.tokenizer.pre_tokenizer.vocab_file_path=os.path.join(CK,cfg.checkpoints.bpe_vocab)
cfg.tokenizer.pre_tokenizer.merges_file_path=os.path.join(CK,cfg.checkpoints.bpe_merges)
tk=VQVAETokenizer.from_config(cfg.tokenizer);vq=tk.vqvae.eval()
tune=torch.load(OUTD+"/vq_code_tune.pt",weights_only=False)
vq.encoder.wte.weight.data.copy_(tune["wte"].to(vq.encoder.wte.weight.dtype))
if tune.get("decoder") is not None: vq.decoder.load_state_dict(tune["decoder"])
EOS=chr(60)+"|endoftext|>";PAD=chr(60)+"|pad|>"
bpe=BPETokenizer(pickle.load(open(OUTD+"/bpe_code_vocab.pickle","rb")),
                 pickle.load(open(OUTD+"/bpe_code_merges.pickle","rb")),
                 {"eos":EOS,"pad":PAD})

def viable(txt):
    # True se txt e prefixo viavel de programa Python
    try:
        codeop.compile_command(txt)
        return True                      # completo ou incompleto: viavel
    except Exception:
        pass
    for suf in (" 0\n","\n    pass\n","\npass\n"):
        try:
            codeop.compile_command(txt+suf)
            return True                  # fecha expressao/bloco: viavel
        except Exception:
            continue
    return False

def n_real(text):
    # stmts reais excluindo docstrings/expr de constante
    try: tree=ast.parse(text)
    except Exception: return -1
    return sum(1 for n in tree.body
               if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant)))

def argmax_decode(codes_row):
    with torch.no_grad():
        _,zl=vq.vq.idx_to_zhats(torch.tensor(codes_row).long().unsqueeze(0),return_zlevels=True)
        ids=vq.decoder(zl[-1]).argmax(-1)[0].tolist()
    if 0 in ids: ids=ids[:ids.index(0)]
    return bpe.decode(ids,skip_special_tokens=True)

def constr_decode(codes_row,K=16):
    with torch.no_grad():
        _,zl=vq.vq.idx_to_zhats(torch.tensor(codes_row).long().unsqueeze(0),return_zlevels=True)
        logits=vq.decoder(zl[-1])[0]
        top=logits.topk(K,-1)
    ids,forced= [],0
    for pos in range(128):
        picked=None
        for k in range(K):
            cand=int(top.indices[pos,k])
            if cand==0:
                picked=0;break            # pad = fim da linha
            if viable(bpe.decode(ids+[cand],skip_special_tokens=True)):
                picked=cand;break
        if picked is None:
            picked=int(top.indices[pos,0]);forced+=1
        if picked==0: break
        ids.append(picked)
    return bpe.decode(ids,skip_special_tokens=True),forced

def eval_rows(rows,tag):
    r={"tag":tag,"n":len(rows)}
    for mode in ("argmax","constrained"):
        np_=nr=nf=0;t0=time.time();outs=[]
        for row in rows:
            if mode=="argmax":
                t=argmax_decode(row);f=0
            else:
                t,f=constr_decode(row,K=int(os.environ.get("K","16")))
            outs.append(t)
            try:
                ast.parse(t);np_+=1
                if n_real(t)>=1: nr+=1
            except Exception: pass
            nf+=f
        r[mode]={"parse":np_,"real":nr,"forced_avg":round(nf/len(rows),1),
                 "secs":round(time.time()-t0,1)}
        json.dump(outs,open(OUTD+f"/constr_{tag}_{mode}.json","w"),indent=1)
        print(tag,mode,json.dumps(r[mode]),flush=True)
    return r

mode=os.environ.get("MODE","both")
codes=np.load(OUTD+"/codes_code.npy")
res={}
if mode in ("both","gold"):
    gold=[c for c in codes[8000:8000+int(os.environ.get("N","32"))]]
    res["gold"]=eval_rows(gold,"gold")
if mode in ("both","gen"):
    ck=torch.load("/work/artifacts/gen_ncm_v2/final.pt",map_location="cpu",weights_only=False)
    c=ck.get("config",{})
    gen=LevelGenerator(vq,n_embd=c.get("n_embd",896),n_layer=c.get("n_layer",4),
                       n_head_q=14,n_head_kv=2,mlp_hidden=4864,
                       intra=c.get("intra",False),corrupt=c.get("corrupt",0.0))
    sd=torch.load(os.environ.get("GEN_CK",OUTD+"/gen_code2.pt"),map_location="cpu",weights_only=False)
    gen.load_state_dict(sd,strict=True);gen.eval()
    torch.manual_seed(7)
    idx=gen.generate(batch_size=int(os.environ.get("N","32")),
                     temperature=float(os.environ.get("TEMP","0.6")),
                     top_k=int(os.environ.get("TOPK","24")),top_p=1.0)
    rows=[r.tolist() for r in idx]
    res["gen"]=eval_rows(rows,"gen")
json.dump(res,open(OUTD+"/constr_decode_results.json","w"),indent=2)
print(json.dumps(res))

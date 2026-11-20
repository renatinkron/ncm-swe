import importlib,os,sys,types,time,json
import torch,torch.nn.functional as F
hydra=types.ModuleType("hydra")
class _U:
    @staticmethod
    def get_class(p):
        m,_,n=p.rpartition(".")
        return getattr(importlib.import_module(m),n)
hydra.utils=_U;sys.modules["hydra"]=hydra
ROOT="/repo/vendor/ncm-upstream";sys.path.insert(0,ROOT);os.chdir(ROOT)
from omegaconf import OmegaConf
from tokenization.vqvae_tokenizer import VQVAETokenizer
from tokenizers import Tokenizer as HFTokenizer
import glob
torch.set_num_threads(int(os.environ.get("THREADS","14")))
CK="/work/artifacts/ncm_official";OUTD="/work/artifacts/ncm_code"
CORPUS="/repo/data/corpus_code";CTX=128;PAD_ID=1
cfg=OmegaConf.load(ROOT+"/config/eval/tinystories.yaml")
cfg.tokenizer.vqvae.checkpoint_path=os.path.join(CK,cfg.checkpoints.tokenizer)
cfg.tokenizer.pre_tokenizer.vocab_file_path=os.path.join(CK,cfg.checkpoints.bpe_vocab)
cfg.tokenizer.pre_tokenizer.merges_file_path=os.path.join(CK,cfg.checkpoints.bpe_merges)
tk=VQVAETokenizer.from_config(cfg.tokenizer);vq=tk.vqvae.eval()
tune=torch.load(OUTD+"/vq_code_tune.pt",weights_only=False)
vq.encoder.wte.weight.data.copy_(tune["wte"].to(vq.encoder.wte.weight.dtype))
if tune.get("decoder") is not None: vq.decoder.load_state_dict(tune["decoder"])

fast=HFTokenizer.from_file(OUTD+"/bpe_code_tokenizer.json")
ids_all=[]
for f in sorted(glob.glob(CORPUS+"/shard_*.txt")):
    ids_all.extend(fast.encode(open(f).read()).ids)
    if len(ids_all)>576*CTX: break
rows=torch.tensor([ids_all[i*CTX:(i+1)*CTX] for i in range(min(576,len(ids_all)//CTX))])
tiny=rows[:64]  # the memorization target

def recovery(vq,b):
    vq.eval();hits=tot=0
    with torch.no_grad():
        codes=vq.encode(b)
        _,zl=vq.vq.idx_to_zhats(codes,return_zlevels=True)
        rec=vq.decoder(zl[-1]).argmax(-1)
        pad=(b==PAD_ID)
        hits=((rec==b)&~pad).sum().item();tot=(~pad).sum().item()
    return hits/max(tot,1)

def run_arm(name,params):
    for p in vq.parameters(): p.requires_grad_(False)
    for p in params: p.requires_grad_(True)
    opt=torch.optim.AdamW(params,lr=3e-4)
    vq.eval();t0=time.time()
    rec0=recovery(vq,tiny)
    print(f"[{name}] rec@0={rec0:.4f}",flush=True)
    hist=[round(rec0,4)]
    for step in range(1,1501):
        logits,vq_loss,*_=vq(tiny)
        ce=F.cross_entropy(logits.reshape(-1,logits.size(-1)),tiny.reshape(-1),ignore_index=PAD_ID)
        loss=ce+0.25*vq_loss
        opt.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(params,1.0);opt.step()
        if step%200==0:
            rec=recovery(vq,tiny);hist.append(round(rec,4))
            print(f"[{name}] step {step} ce={ce.item():.4f} rec={rec:.4f}",flush=True)
    res={"arm":name,"hist":hist,"final":hist[-1],"minutes":round((time.time()-t0)/60,1)}
    json.dump(res,open(OUTD+f"/overfit_{name}.json","w"),indent=2)
    print(json.dumps(res),flush=True)
    return hist[-1]

# Arm 1: decoder only — do the FIXED codes determine the tokens?
run_arm("dec_only",list(vq.decoder.parameters()))

import ast,importlib,json,os,pickle,sys,time,types
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
ck=torch.load("/work/artifacts/gen_ncm_v2/final.pt",map_location="cpu",weights_only=False)
c=ck.get("config",{})
gen=LevelGenerator(vq,n_embd=c.get("n_embd",896),n_layer=c.get("n_layer",4),
                   n_head_q=14,n_head_kv=2,mlp_hidden=4864,
                   intra=c.get("intra",False),corrupt=c.get("corrupt",0.0))
sd=torch.load(os.environ.get("GEN_CK",OUTD+"/gen_code2.pt"),map_location="cpu",weights_only=False)
gen.load_state_dict(sd,strict=True);gen.eval()

def decode(codes_row):
    with torch.no_grad():
        _,zl=vq.vq.idx_to_zhats(torch.tensor(codes_row).unsqueeze(0),return_zlevels=True)
        ids=vq.decoder(zl[-1]).argmax(-1)[0].tolist()
    if 0 in ids: ids=ids[:ids.index(0)]
    return bpe.decode(ids,skip_special_tokens=True)

N=int(os.environ.get("N","32"));TEMP=float(os.environ.get("TEMP","0.9"));TK_=int(os.environ.get("TOPK","32"))
res={}
for tag,(temp,tk_) in {"main":(TEMP,TK_)}.items():
    torch.manual_seed(7)
    t0=time.time()
    idx=gen.generate(batch_size=N,temperature=temp,top_k=tk_,top_p=1.0)
    texts=[decode(r.tolist()) for r in idx]
    n_parse=n_comp=0
    for t in texts:
        try: ast.parse(t); n_parse+=1
        except Exception: pass
        try: compile(t,"<s>","exec"); n_comp+=1
        except Exception: pass
    res[tag]={"n":N,"temp":temp,"top_k":tk_,"parse":n_parse,"compile":n_comp,
              "secs":round(time.time()-t0,1)}
    json.dump(texts,open(OUTD+f"/sample_eval_{tag}.json","w"),indent=1)
    print(tag,json.dumps(res[tag]),flush=True)
print(json.dumps(res))

import importlib,os,sys,types,torch
import numpy as np
hydra=types.ModuleType("hydra")
class _U:
    @staticmethod
    def get_class(p):
        m,_,n=p.rpartition(".")
        return getattr(importlib.import_module(m),n)
hydra.utils=_U;sys.modules["hydra"]=hydra
ROOT="/repo/vendor/ncm-upstream";sys.path.insert(0,ROOT);sys.path.insert(0,"/repo/src");os.chdir(ROOT)
from omegaconf import OmegaConf
from tokenization.vqvae_tokenizer import VQVAETokenizer
from ncm_swe.model.generator import LevelGenerator
torch.set_num_threads(10)
CK="/work/artifacts/ncm_official";OUTD="/work/artifacts/ncm_code"
cfg=OmegaConf.load(ROOT+"/config/eval/tinystories.yaml")
cfg.tokenizer.vqvae.checkpoint_path=os.path.join(CK,cfg.checkpoints.tokenizer)
cfg.tokenizer.pre_tokenizer.vocab_file_path=os.path.join(CK,cfg.checkpoints.bpe_vocab)
cfg.tokenizer.pre_tokenizer.merges_file_path=os.path.join(CK,cfg.checkpoints.bpe_merges)
tk=VQVAETokenizer.from_config(cfg.tokenizer);vq=tk.vqvae.eval()
tune=torch.load(OUTD+"/vq_code_tune.pt",weights_only=False)
vq.encoder.wte.weight.data.copy_(tune["wte"].to(vq.encoder.wte.weight.dtype))
if tune.get("decoder") is not None: vq.decoder.load_state_dict(tune["decoder"])
ck=torch.load("/work/artifacts/gen_ncm_v2/final.pt",map_location="cpu",weights_only=False)
c=ck.get("config",{})
gen=LevelGenerator(vq,n_embd=c.get("n_embd",896),n_layer=c.get("n_layer",4),n_head_q=14,n_head_kv=2,mlp_hidden=4864,intra=c.get("intra",False),corrupt=c.get("corrupt",0.0))
sd=torch.load(os.environ.get("GEN_CK",OUTD+"/gen_lessons2.pt"),map_location="cpu",weights_only=False)
gen.load_state_dict(sd,strict=True);gen.eval()
_off=int(os.environ.get("OFFSET","0"))
codes=torch.tensor(np.load(OUTD+"/"+os.environ.get("CODES","codes_lessons.npy")).astype(np.int64)[_off:_off+64])
with torch.no_grad():
    logits_l0,outs=gen(codes)
l0=gen.levels[0]
acc_l0=(logits_l0.argmax(-1)==codes[:,:l0]).float().mean().item()
print(f"l0 (coarse) top1 acc: {acc_l0:.4f}")
ofs=l0;tot=0;ok=0
for i,out in enumerate(outs):
    L=gen.chunk_levels[i]
    t=codes[:,ofs:ofs+L]
    a=(out.argmax(-1)==t).float()
    tk=(out.topk(5,-1).indices==t.unsqueeze(-1)).any(-1).float()
    print(f"  l{i+1} ({L} codes): top1 acc={a.mean().item():.4f} top5 acc={tk.mean().item():.4f}")
    ok+=a.sum().item();tot+=t.numel();ofs+=L
    globals().setdefault("_k5",[0,0]);_k5[0]+=tk.sum().item();_k5[1]+=t.numel()
print(f"fine levels overall: {ok/tot:.4f}  top5: {_k5[0]/_k5[1]:.4f}")
# whole-row exactness: % rows where ALL 641 codes predicted right
allok=(logits_l0.argmax(-1)==codes[:,:l0]).all(1)
ofs=l0
for i,out in enumerate(outs):
    L=gen.chunk_levels[i]
    allok=allok&(out.argmax(-1)==codes[:,ofs:ofs+L]).all(1)
    ofs+=L
print(f"rows with ALL 641 codes correct: {allok.float().mean().item():.3f}")

import importlib,os,sys,types,time
import numpy as np,torch
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
torch.set_num_threads(int(os.environ.get("THREADS","14")))
CK="/work/artifacts/ncm_official";OUTD="/work/artifacts/ncm_code"
cfg=OmegaConf.load(ROOT+"/config/eval/tinystories.yaml")
cfg.tokenizer.vqvae.checkpoint_path=os.path.join(CK,cfg.checkpoints.tokenizer)
cfg.tokenizer.pre_tokenizer.vocab_file_path=os.path.join(CK,cfg.checkpoints.bpe_vocab)
cfg.tokenizer.pre_tokenizer.merges_file_path=os.path.join(CK,cfg.checkpoints.bpe_merges)
tk=VQVAETokenizer.from_config(cfg.tokenizer);vq=tk.vqvae.eval()
tune=torch.load(OUTD+"/vq_code_tune.pt",weights_only=False)
vq.encoder.wte.weight.data.copy_(tune["wte"].to(vq.encoder.wte.weight.dtype))
if tune.get("decoder") is not None: vq.decoder.load_state_dict(tune["decoder"])
for p in vq.parameters(): p.requires_grad_(False)
ck=torch.load("/work/artifacts/gen_ncm_v2/final.pt",map_location="cpu",weights_only=False)
c=ck.get("config",{})
gen=LevelGenerator(vq,n_embd=c.get("n_embd",896),n_layer=c.get("n_layer",4),n_head_q=14,n_head_kv=2,mlp_hidden=4864,intra=c.get("intra",False),corrupt=c.get("corrupt",0.0))
sd=torch.load(OUTD+"/gen_lessons2.pt",map_location="cpu",weights_only=False)
gen.load_state_dict(sd,strict=True)
tiny=torch.tensor(np.load(OUTD+"/codes_lessons.npy").astype(np.int64)[:64])

def acc():
    gen.eval()
    with torch.no_grad():
        logits_l0,outs=gen(tiny)
    ok=tot=0
    ofs=gen.levels[0]
    ok+=(logits_l0.argmax(-1)==tiny[:,:gen.levels[0]]).sum().item();tot+=gen.levels[0]*len(tiny)
    for i,out in enumerate(outs):
        L=gen.chunk_levels[i]
        ok+=(out.argmax(-1)==tiny[:,ofs:ofs+L]).sum().item();tot+=L*len(tiny);ofs+=L
    return ok/tot

gen.train()
opt=torch.optim.AdamW(gen.parameters(),lr=1e-4)
print(f"acc@0={acc():.4f}",flush=True)
for step in range(1,801):
    out=gen.loss(tiny)
    loss=out[0] if isinstance(out,(tuple,list)) else out
    opt.zero_grad();loss.backward()
    torch.nn.utils.clip_grad_norm_(gen.parameters(),1.0);opt.step()
    if step%100==0:
        print(f"step {step} loss={float(loss):.4f} acc={acc():.4f}",flush=True)

import importlib,os,sys,types,torch,ast,json
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
for p in vq.parameters(): p.requires_grad_(False)
import pickle
from tokenization.bpe_tokenizer import BPETokenizer
EOS=chr(60)+"|endoftext|>";PAD=chr(60)+"|pad|>"
bpe=BPETokenizer(pickle.load(open(OUTD+"/bpe_code_vocab.pickle","rb")),pickle.load(open(OUTD+"/bpe_code_merges.pickle","rb")),{"eos":EOS,"pad":PAD})
ck=torch.load("/work/artifacts/gen_ncm_v2/final.pt",map_location="cpu",weights_only=False)
c=ck.get("config",{})
gen=LevelGenerator(vq,n_embd=c.get("n_embd",896),n_layer=c.get("n_layer",4),n_head_q=14,n_head_kv=2,mlp_hidden=4864,intra=c.get("intra",False),corrupt=c.get("corrupt",0.0))
sd=torch.load(OUTD+"/gen_lessons2.pt",map_location="cpu",weights_only=False)
gen.load_state_dict(sd,strict=True);gen.eval()
for T in [0.3,0.6]:
    torch.manual_seed(7)
    idx=gen.generate(batch_size=16,temperature=T,top_k=32,top_p=1.0)
    ok=0;ex=None
    with torch.no_grad():
        _,zl=vq.vq.idx_to_zhats(idx,return_zlevels=True)
        ids_all=vq.decoder(zl[-1]).argmax(-1).tolist()
    for row in ids_all:
        r=row[:row.index(0)] if 0 in row else row
        t=bpe.decode(r,skip_special_tokens=True)
        try: ast.parse(t); ok+=1; ex=ex or t
        except Exception: pass
    print(f"temp={T}: parse {ok}/16")
    if ex: print("EX:",repr(ex[:150]))

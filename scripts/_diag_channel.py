import importlib,os,sys,types,torch,ast
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
import pickle
from tokenization.bpe_tokenizer import BPETokenizer
EOS=chr(60)+"|endoftext|>";PAD=chr(60)+"|pad|>"
bpe=BPETokenizer(pickle.load(open(OUTD+"/bpe_code_vocab.pickle","rb")),pickle.load(open(OUTD+"/bpe_code_merges.pickle","rb")),{"eos":EOS,"pad":PAD})
lessons=[l.strip() for l in open("/repo/data/corpus_teach/lessons.txt").read().split("\n\n") if l.strip()]
parse_ok=0;tok_ok_sum=0.0;first_bad=None
with torch.no_grad():
    for lesson in lessons:
        ids=bpe.encode(lesson)[:127]+[0]
        ids=ids+[0]*(128-len(ids))
        x=torch.tensor([ids])
        codes=vq.encode(x)
        _,zl=vq.vq.idx_to_zhats(codes,return_zlevels=True)
        rec=vq.decoder(zl[-1]).argmax(-1)[0].tolist()
        ref=ids
        n=min(len(ref),len(rec))
        tok_ok=sum(1 for a,b in zip(ref[:n],rec[:n]) if a==b)/n
        tok_ok_sum+=tok_ok
        r=rec[:rec.index(0)] if 0 in rec else rec
        t=bpe.decode(r,skip_special_tokens=True)
        try:
            ast.parse(t);parse_ok+=1
        except Exception:
            if first_bad is None: first_bad=t
print(f"channel-only: lessons={len(lessons)} ast.parse={parse_ok} mean_tok_rec={tok_ok_sum/len(lessons):.4f}")
if first_bad: print("FIRST_BAD:",repr(first_bad[:200]))

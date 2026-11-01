"""Finding dataset and benchmark names in abstracts.

Two parts:
1. A seed list of well-known datasets and benchmarks (high precision, limited recall).
2. Patterns that find names the list does not know ("the Stanford Cars dataset",
   "evaluated on Foo-Bench and Bar-QA"). These can also be wrong, which is why the
   evaluation compares each part against a hand-labeled set.
"""

import re

# canonical name | aliases ... (matching is case-sensitive, longest name first)
SEED = """
ImageNet | ImageNet-1K | ImageNet-1k | ImageNet1K | ILSVRC | Imagenet
ImageNet-21K | ImageNet-21k
Tiny-ImageNet | TinyImageNet | Tiny ImageNet
ImageNet-C | ImageNet-A | ImageNet-R | ImageNet-V2 | ImageNet-O
CIFAR-10 | CIFAR10
CIFAR-100 | CIFAR100
MNIST
Fashion-MNIST | FashionMNIST
SVHN
STL-10 | STL10
COCO | MS COCO | MS-COCO | MSCOCO | COCO 2017
LVIS
Objects365
Open Images | OpenImages
Pascal VOC | PASCAL VOC | VOC2007 | VOC2012 | VOC 2007 | VOC 2012
Cityscapes
KITTI
nuScenes
Waymo Open Dataset | Waymo
ADE20K
CelebA | CelebA-HQ
FFHQ
LSUN
LAION-5B | LAION-400M | LAION
Places365
Kinetics | Kinetics-400 | Kinetics-600 | Kinetics-700
UCF101 | UCF-101
HMDB51
Something-Something | Something-Something V2
ActivityNet
MSR-VTT | MSRVTT
VQAv2 | VQA v2
GQA
TextVQA
OK-VQA
Flickr30k | Flickr30K
CLEVR
ScanNet
ShapeNet
ModelNet40 | ModelNet
S3DIS
Human3.6M
NYU Depth | NYUv2 | NYU-Depth V2
DAVIS
MOT17
MOT20
DanceTrack
Market-1501 | Market1501
WIDER FACE
LFW
GLUE
SuperGLUE
SQuAD | SQuAD1.1 | SQuAD2.0 | SQuAD 2.0 | SQuAD 1.1
MNLI
SNLI
SST-2
CoNLL-2003 | CoNLL03 | CoNLL 2003
OntoNotes
WMT | WMT14 | WMT16 | WMT19 | WMT22 | WMT23
IWSLT | IWSLT14
CNN/DailyMail | CNN/Daily Mail | CNN-DM
XSum
Gigaword
MMLU
MMLU-Pro
GSM8K
MATH
MATH-500
HumanEval
MBPP
BIG-bench | BIG-Bench | BBH
HellaSwag
WinoGrande
ARC-Challenge | ARC-Easy
TruthfulQA
PIQA
BoolQ
DROP
Natural Questions | NaturalQuestions
TriviaQA
HotpotQA
MS MARCO | MSMARCO
BEIR
MTEB
LongBench
RULER
AlpacaEval
MT-Bench
Chatbot Arena
SWE-bench | SWE-Bench
GPQA
AIME
LiveCodeBench
Codeforces
CodeContests
APPS
FLORES | FLORES-200
XNLI
XTREME
LibriSpeech
Common Voice
VoxCeleb | VoxCeleb1 | VoxCeleb2
AudioSet
LJSpeech
Atari
MuJoCo
D4RL
Meta-World | MetaWorld
Procgen
DeepMind Control Suite | DMControl
Cora
Citeseer | CiteSeer
PubMed
OGB
ogbn-arxiv
ogbn-products
ZINC
QM9
MUTAG
MIMIC-III
MIMIC-IV
ChestX-ray14 | ChestX-ray
CheXpert
BraTS
ISIC
LIDC-IDRI
TCGA
UK Biobank
WikiText-103 | WikiText | Wikitext-103
Penn Treebank | PTB
C4
The Pile | Pile
RedPajama
FineWeb
Common Crawl
OpenWebText
BookCorpus
Dolma
SlimPajama
"""

# Capitalised words that are not dataset names.
_NOT_NAMES = {
    "This", "That", "These", "Those", "The", "Our", "We", "Their", "Its", "A", "An", "Each",
    "Such", "Existing", "Large", "Recent", "New", "Public", "Real", "Synthetic", "Standard",
    "Several", "Many", "Both", "Other", "Another", "Common", "Popular", "Multiple", "Various",
    "Benchmark", "Dataset", "Datasets", "Corpus", "Training", "Test", "Evaluation", "In", "On",
    "And", "For", "With", "Using", "Across", "Moreover", "However", "Furthermore", "Finally",
    "Experiments", "Experimental", "Extensive", "Results", "Compared", "Unlike", "Despite",
}
# Common acronyms that are not datasets.
_NOT_DATASETS = {
    "CPU", "GPU", "TPU", "API", "LLM", "LLMs", "AI", "ML", "NLP", "GAN", "GANs", "CNN", "CNNs",
    "RNN", "LSTM", "VAE", "RL", "SGD", "IoT", "SOTA", "MLP", "GNN", "GNNs", "VLM", "VLMs",
    "MLLM", "MLLMs", "RAG", "SFT", "RLHF", "DPO", "PPO", "LoRA", "NLU", "NLG", "OCR", "UAV",
    "SNN", "SNNs", "ViT", "BERT", "GPT", "GPT-4", "GPT-3", "CLIP", "LLaMA", "Llama", "ResNet",
    "AUC", "BLEU", "ROUGE", "FLOPs", "FLOPS", "MSE", "PCA", "KL", "EM", "IID", "OOD",
}

_TRIGGER = re.compile(
    r"\b(?:evaluat\w*|tested|testing|trained|training|experiments?|benchmark\w*|results?|"
    r"performance|outperform\w*)\b[^.;:]{0,40}?\b(?:on|across|using|over)\s+"
)
_NAME_TOKEN = re.compile(r"^[A-Z][A-Za-z0-9\-\.]*[A-Za-z0-9]$|^[A-Z][A-Za-z0-9]$")
_KIND = re.compile(r"\b(?:dataset|datasets|benchmark|benchmarks|corpus|corpora)\b")


def normalize_key(name: str) -> str:
    """'MS-COCO' and 'ms coco' both become 'mscoco'."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _parse_seed():
    canonical, alias_to_canonical = [], {}
    for line in SEED.strip().splitlines():
        names = [n.strip() for n in line.split("|") if n.strip()]
        canon = names[0]
        canonical.append(canon)
        for alias in names:
            alias_to_canonical[alias] = canon
    return canonical, alias_to_canonical


CANONICAL_NAMES, ALIASES = _parse_seed()
_KEY_TO_CANONICAL = {normalize_key(a): c for a, c in ALIASES.items()}
_SEED_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:" + "|".join(re.escape(a) for a in sorted(ALIASES, key=len, reverse=True))
    + r")(?![A-Za-z0-9])"
)


def canonicalize(name: str) -> str:
    """Map an alias to its canonical name ('MS COCO' -> 'COCO'). Unknown names are returned as typed."""
    return _KEY_TO_CANONICAL.get(normalize_key(name), name.strip())


def find_seed(text: str) -> list[str]:
    """Known datasets mentioned in the text (canonical names, in order, no duplicates)."""
    found = []
    for m in _SEED_RE.finditer(text or ""):
        canon = ALIASES[m.group(0)]
        if canon not in found:
            found.append(canon)
    return found


def _plausible(name: str, followed_by_kind: bool = False) -> bool:
    tokens = name.split()
    if not tokens or any(t in _NOT_NAMES for t in tokens[:1]):
        return False
    if name in _NOT_DATASETS or len(name) < 2:
        return False
    if any(not _NAME_TOKEN.match(t) for t in tokens):
        return False
    has_shape = bool(re.search(r"[0-9\-]", name)) or len(re.findall(r"[A-Z]", name)) >= 2
    # "the Stanford Cars dataset": two Capitalised words directly before "dataset" are accepted.
    return has_shape or (followed_by_kind and len(tokens) == 2)


_MASK = "ZZSEEDZZ"


def find_patterns(text: str) -> list[str]:
    """Names found by patterns. May contain mistakes. Known datasets are returned canonicalized."""
    # Hide the known names first, so the patterns cannot return pieces of them ("MS" from "MS COCO").
    text = _SEED_RE.sub(_MASK, text or "")
    found = []

    def add(name: str, kind: bool = False):
        name = name.strip(" ,.;()[]\"'")
        if _MASK in name:
            return
        if _plausible(name, kind):
            name = canonicalize(name)
            if name not in found:
                found.append(name)

    # 1. "<Name> dataset/benchmark/corpus"
    for m in _KIND.finditer(text):
        words = text[: m.start()].split()[-2:]
        if words:
            if len(words) == 2 and _NAME_TOKEN.match(words[0].strip(",.;()")) and _NAME_TOKEN.match(words[1].strip(",.;()")):
                add(" ".join(words), kind=True)
            add(words[-1], kind=True)

    # 2. "evaluated on A, B and C"
    for m in _TRIGGER.finditer(text):
        for token in text[m.end():].split()[:14]:
            bare = token.strip(",.;()")
            if bare.lower() in ("and", "&", "or"):
                continue
            if _NAME_TOKEN.match(bare):
                add(bare)  # the mask is ignored in add(), but the loop keeps reading the list
                if token.endswith((".", ";", ")")):
                    break
                continue
            break
    return found


def extract_datasets(text: str) -> dict:
    """{'seed': [...], 'pattern': [...], 'all': [...]} with canonical names."""
    seed = find_seed(text)
    pattern = [n for n in find_patterns(text) if n not in seed]
    return {"seed": seed, "pattern": pattern, "all": seed + pattern}

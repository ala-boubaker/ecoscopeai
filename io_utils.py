import io, math, time, os, hashlib
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# --- ESG & NLP helpers ---
from dataclasses import dataclass
# PDF
from pdfminer.high_level import extract_text as pdf_extract_text
# HF transformers
from transformers import pipeline

# Optional OCR fallback (Tesseract + pdf2image)
try:
    import pytesseract
    _HAVE_TESSERACT = True
except Exception:
    _HAVE_TESSERACT = False

try:
    from pdf2image import convert_from_bytes
    _HAVE_PDF2IMG = True
except Exception:
    _HAVE_PDF2IMG = False


# ---------------------------------------------------------
# 1. Arabic-aware normalization
# ---------------------------------------------------------

ARABIC_DIACRITICS = [
    "َ","ً","ُ","ٌ","ِ","ٍ","ْ","ّ"
]

ARABIC_STOPWORDS = set([
    "من","على","الى","في","عن","مع","و","أو","ثم","هذا","هذه",
    "ذلك","تلك","هو","هي","هم","هن","كما","لها","له","لهم"
])

def _has_arabic(text: str) -> bool:
    return any('\u0600' <= ch <= '\u06FF' for ch in text)

def arabic_normalize(text: str) -> str:
    """Remove diacritics, normalize shapes, trim stopwords for Arabic."""
    if not isinstance(text, str):
        return ""
    # remove diacritics
    for d in ARABIC_DIACRITICS:
        text = text.replace(d, "")
    # unify characters
    text = (
        text.replace("أ", "ا")
            .replace("إ", "ا")
            .replace("آ", "ا")
            .replace("ى", "ي")
            .replace("ة", "ه")
    )
    text = text.strip().lower()
    tokens = [t for t in text.split() if t not in ARABIC_STOPWORDS]
    return " ".join(tokens)

def normalize_series(s: pd.Series) -> pd.Series:
    """
    Unified normalization for mixed Arabic + English text.

    Behaviour:
    - If string contains Arabic characters → arabic_normalize()
    - Else → original EcoScopeAI ASCII normalization (NFKD, ASCII, lower, strip)
    """
    if s is None:
        return pd.Series([], dtype="object")

    out = []
    for v in s.astype(str):
        raw = v.strip()
        if _has_arabic(raw):
            out.append(arabic_normalize(raw))
        else:
            eng = (
                raw
                .encode("ascii", errors="ignore")
                .decode("ascii", errors="ignore")
                .strip()
                .lower()
            )
            out.append(eng)
    return pd.Series(out, dtype="object")


# -------------------------
# File helpers / loaders
# -------------------------

def _is_streamlit_file(obj) -> bool:
    """True if this looks like a Streamlit UploadedFile."""
    return hasattr(obj, "read") and hasattr(obj, "name")

def load_table_any(file_or_path) -> pd.DataFrame:
    """
    Load single CSV/XLSX into a DataFrame.
    Returns empty DataFrame if None or unreadable.
    """
    if file_or_path is None:
        return pd.DataFrame()

    try:
        if _is_streamlit_file(file_or_path):
            name = getattr(file_or_path, "name", "")
            data = file_or_path.read()
            bio = io.BytesIO(data)
            if name.lower().endswith(".csv"):
                bio.seek(0)
                return pd.read_csv(bio)
            elif name.lower().endswith(".xlsx"):
                bio.seek(0)
                xl = pd.ExcelFile(bio)
                first = xl.sheet_names[0]
                return xl.parse(first)
            return pd.DataFrame()
        else:
            path = str(file_or_path)
            if not os.path.exists(path):
                return pd.DataFrame()
            if path.lower().endswith(".csv"):
                return pd.read_csv(path)
            elif path.lower().endswith(".xlsx"):
                xl = pd.ExcelFile(path)
                first = xl.sheet_names[0]
                return xl.parse(first)
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def read_multisheet(file_or_path) -> Dict[str, pd.DataFrame]:
    """
    Read CSV (single sheet) or XLSX (multi-sheet) into dict of DataFrames.
    Expected sheet names: Procurement_Transport, Waste, Business_Travel,
    Employee_Commute, Energy, CBAM.
    """
    sheets: Dict[str, pd.DataFrame] = {
        "Procurement_Transport": pd.DataFrame(),
        "Waste": pd.DataFrame(),
        "Business_Travel": pd.DataFrame(),
        "Employee_Commute": pd.DataFrame(),
        "Energy": pd.DataFrame(),
        "CBAM": pd.DataFrame(),
    }
    if file_or_path is None:
        return sheets

    try:
        if _is_streamlit_file(file_or_path):
            name = getattr(file_or_path, "name", "")
            data = file_or_path.read()
            bio = io.BytesIO(data)
            if name.lower().endswith(".csv"):
                bio.seek(0)
                sheets["Procurement_Transport"] = pd.read_csv(bio)
            elif name.lower().endswith(".xlsx"):
                bio.seek(0)
                xl = pd.ExcelFile(bio)
                for k in sheets.keys():
                    if k in xl.sheet_names:
                        sheets[k] = xl.parse(k)
            return sheets
        else:
            path = str(file_or_path)
            if not os.path.exists(path):
                return sheets
            if path.lower().endswith(".csv"):
                sheets["Procurement_Transport"] = pd.read_csv(path)
            elif path.lower().endswith(".xlsx"):
                xl = pd.ExcelFile(path)
                for k in sheets.keys():
                    if k in xl.sheet_names:
                        sheets[k] = xl.parse(k)
            return sheets
    except Exception:
        return sheets


# -------------------------
# EF validation & struct builders
# -------------------------

def validate_ef(ef_df: pd.DataFrame) -> bool:
    """
    EF file must have at least: type, key, ef_value, unit.
    """
    if ef_df is None or ef_df.empty:
        return False
    need = {"type", "key", "ef_value", "unit"}
    return need.issubset(set(map(str.lower, ef_df.columns)))

def _normalize_ef_columns(ef_df: pd.DataFrame) -> pd.DataFrame:
    """Make sure EF columns are lowercased as expected."""
    cols = {c: c.lower() for c in ef_df.columns}
    ef = ef_df.rename(columns=cols).copy()
    ef["type"] = normalize_series(ef["type"])
    ef["key"] = normalize_series(ef["key"])
    ef["ef_value"] = pd.to_numeric(ef["ef_value"], errors="coerce").fillna(0.0)
    ef["unit"] = ef["unit"].astype(str)
    return ef

def build_ef_structs(ef_df: pd.DataFrame):
    """
    Returns:
      - spend_efs (DataFrame indexed by category key)
      - mode_ef (dict), ef_waste (dict), ef_travel (dict),
        ef_comm (dict), ef_energy (dict of key -> {ef_value, unit}),
        ef_cbam (dict), spend_unit (str)
    """
    ef = _normalize_ef_columns(ef_df)

    # Spend EFs
    spend = ef[ef["type"] == "spend"][["key", "ef_value", "unit"]].drop_duplicates()
    spend_efs = spend.set_index("key")

    # Transport mode dict
    mode = ef[ef["type"] == "transport"][["key", "ef_value"]]
    mode_ef = dict(zip(mode["key"], mode["ef_value"]))

    # Waste / travel / commute dicts
    w = ef[ef["type"] == "waste"][["key", "ef_value"]]
    t = ef[ef["type"] == "travel"][["key", "ef_value"]]
    c = ef[ef["type"] == "commute"][["key", "ef_value"]]
    ef_waste = dict(zip(w["key"], w["ef_value"]))
    ef_travel = dict(zip(t["key"], t["ef_value"]))
    ef_comm = dict(zip(c["key"], c["ef_value"]))

    # Energy
    e = ef[ef["type"] == "energy"][["key", "ef_value", "unit"]]
    ef_energy = {k: {"ef_value": v, "unit": u}
                 for k, v, u in zip(e["key"], e["ef_value"], e["unit"])}

    # CBAM product EF
    cb = ef[ef["type"] == "cbam_product"][["key", "ef_value"]]
    ef_cbam = dict(zip(cb["key"], cb["ef_value"]))

    # Spend unit (assume uniform)
    spend_unit = spend["unit"].iloc[0] if not spend.empty else "kgCO2e_per_GBP"
    return spend_efs, mode_ef, ef_waste, ef_travel, ef_comm, ef_energy, ef_cbam, spend_unit


# -------------------------
# Simple NLP (TF-IDF) for key suggestion
# -------------------------

def build_ef_corpora(ef_df: pd.DataFrame) -> Dict[str, List[Tuple[str, str]]]:
    """
    Build a small “corpus” per EF type.
    Returns dict[type] -> list of (key, text_for_matching)
    """
    ef = _normalize_ef_columns(ef_df)
    corpora: Dict[str, List[Tuple[str, str]]] = {}
    for t in ef["type"].unique():
        sub = ef[ef["type"] == t]
        items = [(row["key"], f"{row['key']} {row['unit']}") for _, row in sub.iterrows()]
        if items:
            corpora[t] = items
    return corpora

class EmbeddingEngine:
    """
    Lightweight TF-IDF engine per EF type. Offline, fast, good-enough.
    """
    def __init__(self):
        self.vectorizer: TfidfVectorizer = None
        self.doc_matrix = None
        self.keys: List[str] = []
        self.docs: List[str] = []

    def fit(self, pairs: List[Tuple[str, str]]):
        self.keys = [k for k, _ in pairs]
        self.docs = [txt for _, txt in pairs]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)
        self.doc_matrix = self.vectorizer.fit_transform(self.docs)

    def suggest(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        if not query or self.vectorizer is None or self.doc_matrix is None or not self.keys:
            return []
        qv = self.vectorizer.transform([str(query)])
        sims = linear_kernel(qv, self.doc_matrix).ravel()
        if sims.size == 0:
            return []
        idx = sims.argsort()[::-1][:top_k]
        return [(self.keys[i], float(sims[i])) for i in idx]


# ---------- hashing ----------
def file_bytes_sha1(b: bytes) -> str:
    h = hashlib.sha1()
    h.update(b)
    return h.hexdigest()

def filelike_to_bytes(f) -> bytes:
    if hasattr(f, "getvalue"):
        return f.getvalue()
    if hasattr(f, "read"):
        pos = f.tell() if hasattr(f, "tell") else None
        data = f.read()
        if pos is not None:
            try: f.seek(pos)
            except Exception: pass
        return data
    # assume path
    with open(f, "rb") as fh:
        return fh.read()


# ---------- PDF / chunk ----------
def _pdf_to_images(pdf_bytes: bytes):
    """Optional PDF→image conversion for OCR fallback."""
    if not _HAVE_PDF2IMG:
        return []
    try:
        return convert_from_bytes(pdf_bytes, dpi=300)
    except Exception:
        return []

def extract_text_from_pdf(file_like) -> str:
    """
    Hybrid extractor:
    1) Try pdfminer text extraction
    2) If almost nothing extracted and Tesseract+pdf2image are available → OCR fallback
    """
    try:
        b = filelike_to_bytes(file_like)
        # 1) pdfminer
        text = pdf_extract_text(io.BytesIO(b)) or ""
        if len(text.strip()) > 50:
            return text

        # 2) OCR fallback (optional)
        if _HAVE_TESSERACT and _HAVE_PDF2IMG:
            images = _pdf_to_images(b)
            if images:
                ocr_text = ""
                for img in images:
                    ocr_text += pytesseract.image_to_string(img, lang="ara+eng") + "\n"
                if len(ocr_text.strip()) > 0:
                    return ocr_text.strip()

        return text
    except Exception:
        return ""

def extract_text_from_image(file_like) -> str:
    """PNG/JPG → OCR (if Tesseract installed)."""
    if not _HAVE_TESSERACT:
        return ""
    try:
        b = filelike_to_bytes(file_like)
        from PIL import Image
        img = Image.open(io.BytesIO(b))
        return pytesseract.image_to_string(img, lang="ara+eng")
    except Exception:
        return ""

def split_into_chunks(text: str, max_chars: int = 900, overlap: int = 120) -> List[str]:
    if not text: return []
    chunks, i, n = [], 0, len(text)
    step = max_chars - overlap
    while i < n and len(chunks) < 400:
        chunks.append(text[i:i+max_chars].strip())
        i += max(step, 1)
    return [c for c in chunks if c]


# ---------- models ----------
def load_sentiment_pipeline(model_name: str = "ProsusAI/finbert"):
    return pipeline("text-classification", model=model_name, tokenizer=model_name, return_all_scores=True, truncation=True)

def load_zeroshot_pipeline(model_name: str = "valhalla/distilbart-mnli-12-1"):
    return pipeline("zero-shot-classification", model=model_name, tokenizer=model_name)

ESG_LABELS = ["Environment", "Social", "Governance"]

@dataclass
class ChunkResult:
    supplier: str
    chunk_id: int
    text: str
    sent_pos: float
    sent_neu: float
    sent_neg: float
    topic_env: float
    topic_soc: float
    topic_gov: float

def score_sentiment(outputs) -> Tuple[float,float,float]:
    label_map = {"positive":"pos","negative":"neg","neutral":"neu",
                 "POSITIVE":"pos","NEGATIVE":"neg","NEUTRAL":"neu"}
    pos = neu = neg = 0.0
    for d in outputs:
        k = label_map.get(d["label"])
        if k == "pos": pos = float(d["score"])
        elif k == "neu": neu = float(d["score"])
        elif k == "neg": neg = float(d["score"])
    s = pos + neu + neg
    if s > 0:
        pos, neu, neg = pos/s, neu/s, neg/s
    return pos, neu, neg

# ---------- FAST topic tagging (rule-based fallback) ----------
_KEYWORDS = {
    "Environment": ["emission","carbon","co2","energy","waste","water","biodiversity","renewable","climate","scope 1","scope 2","scope 3"],
    "Social": ["labor","diversity","safety","injury","community","training","human rights","inclusion","wellbeing"],
    "Governance": ["board","audit","bribery","ethics","compliance","whistleblowing","corruption","governance","privacy","risk management"],
}
def rule_based_esg_scores(text: str) -> Tuple[float,float,float]:
    t = text.lower()
    def score(words):
        hits = sum(1 for w in words if w in t)
        return min(1.0, hits/3.0)
    return score(_KEYWORDS["Environment"]), score(_KEYWORDS["Social"]), score(_KEYWORDS["Governance"])

# ---------- Batched inference ----------
def analyze_chunks_batched(supplier: str, chunks: List[str], sent_pipe, zshot_pipe=None, fast_topics=False, batch_size=16) -> List[ChunkResult]:
    results: List[ChunkResult] = []
    if not chunks: return results

    # 1) sentiment in batches
    sent_out = sent_pipe(chunks, truncation=True, batch_size=batch_size)
    sent_triplets = [score_sentiment(o) for o in sent_out]

    # 2) topics
    if fast_topics or zshot_pipe is None:
        topic_scores = [rule_based_esg_scores(ch) for ch in chunks]
    else:
        topic_scores = []
        for ch in chunks:
            z = zshot_pipe(ch, candidate_labels=ESG_LABELS, multi_label=True)
            mp = {lab: sc for lab, sc in zip(z["labels"], z["scores"])}
            topic_scores.append((mp.get("Environment",0.0), mp.get("Social",0.0), mp.get("Governance",0.0)))

    # 3) assemble
    for i, ch in enumerate(chunks):
        p,n,e = sent_triplets[i][0], sent_triplets[i][2], sent_triplets[i][1]
        env, soc, gov = topic_scores[i]
        results.append(ChunkResult(
            supplier=supplier, chunk_id=i, text=ch,
            sent_pos=p, sent_neu=n, sent_neg=e,
            topic_env=env, topic_soc=soc, topic_gov=gov
        ))
    return results

# ---------- aggregate ----------
def _norm01(x, lo, hi):
    if hi <= lo: return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))

def aggregate_supplier_esg(results: List[ChunkResult]) -> pd.DataFrame:
    import numpy as np
    if not results:
        return pd.DataFrame(columns=["supplier","sentiment_score","coverage","balance","controversy","chunks","neg_share","ESG_Index_0_100"])
    df = pd.DataFrame([r.__dict__ for r in results])
    rows=[]
    for supp, g in df.groupby("supplier"):
        pos_mean = float(g["sent_pos"].mean())
        neg_mean = float(g["sent_neg"].mean())
        sent_score = _norm01(pos_mean - neg_mean, -1.0, 1.0)
        env = float(g["topic_env"].mean()); soc = float(g["topic_soc"].mean()); gov = float(g["topic_gov"].mean())
        coverage = (env + soc + gov)/3.0
        balance = 1.0 - float(np.std([env,soc,gov])); balance = max(0.0, min(1.0, balance))
        neg_share = float((g["sent_neg"] > 0.5).mean())
        controversy = 1.0 - neg_share
        w_sent, w_cov, w_bal, w_cont = 0.45, 0.25, 0.15, 0.15
        esg_idx = (w_sent*sent_score + w_cov*coverage + w_bal*balance + w_cont*controversy) * 100.0
        rows.append({
            "supplier": supp, "sentiment_score": round(sent_score,3), "coverage": round(coverage,3),
            "balance": round(balance,3), "controversy": round(controversy,3),
            "chunks": int(len(g)), "neg_share": round(neg_share,3), "ESG_Index_0_100": round(esg_idx,1)
        })
    return pd.DataFrame(rows).sort_values("ESG_Index_0_100", ascending=False).reset_index(drop=True)

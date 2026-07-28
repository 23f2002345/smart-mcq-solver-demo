import json, re, pickle
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
import onnxruntime as rt

# ---- CHANGE TO YOUR ACTUAL HF USERNAME ----
HF_USERNAME = "Gop05"
ASSETS_REPO = f"{HF_USERNAME}/smart-mcq-assets"
DEBERTA_REPO = f"{HF_USERNAME}/smart-mcq-deberta"

LETTERS = ["A", "B", "C", "D", "E"]
SEP = "|||"

st.set_page_config(page_title="Smart MCQ Solver", page_icon="🎯")

@st.cache_resource
def load_everything():
    def asset(fname):
        return hf_hub_download(repo_id=ASSETS_REPO, filename=fname)

    # ---- classical model via ONNX (version-safe) ----
    sess = rt.InferenceSession(asset("classical_model.onnx"))
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[1].name   # index 1 = probabilities

    # ---- tfidf vectorizer ----
    with open(asset("tfidf_vectorizer.pkl"), "rb") as f:
        tfidf = pickle.load(f)

    # ---- vocab + config ----
    with open(asset("vocab.json")) as f:
        vocab_data = json.load(f)
    word2idx   = {w: i for i, w in enumerate(vocab_data["vocab"])}
    PAD_IDX    = vocab_data["pad_idx"]
    MAX_LEN    = vocab_data["max_len"]
    EMBED_DIM  = vocab_data["embed_dim"]
    vocab_size = len(vocab_data["vocab"])

    # ---- ensemble weights ----
    with open(asset("ensemble_weights.json")) as f:
        weights_data = json.load(f)

    # ---- lookup table ----
    with open(asset("lookup.json")) as f:
        lookup_raw = json.load(f)
    lookup = {tuple(k.split(SEP)): v for k, v in lookup_raw.items()}

    # ---- DeBERTa ----
    deberta_tok = AutoTokenizer.from_pretrained(DEBERTA_REPO)
    deberta_mdl = AutoModelForSequenceClassification.from_pretrained(
        DEBERTA_REPO).eval()

    # ---- TextCNN ----
    class TextCNN(nn.Module):
        def __init__(self, vocab_size, embed_dim, num_classes=5,
                     num_filters=100, kernel_sizes=(3, 4, 5), dropout=0.5):
            super().__init__()
            self.embedding = nn.Embedding(
                vocab_size, embed_dim, padding_idx=PAD_IDX)
            self.convs = nn.ModuleList([
                nn.Conv1d(embed_dim, num_filters, k, padding=k // 2)
                for k in kernel_sizes
            ])
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

        def forward(self, x):
            emb = self.embedding(x).transpose(1, 2)
            pooled = [torch.relu(c(emb)).max(dim=2).values
                      for c in self.convs]
            return self.fc(self.dropout(torch.cat(pooled, dim=1)))

    # ---- BiLSTM ----
    class BiLSTMClassifier(nn.Module):
        def __init__(self, vocab_size, embed_dim, num_classes=5,
                     hidden_dim=128, dropout=0.5):
            super().__init__()
            self.embedding = nn.Embedding(
                vocab_size, embed_dim, padding_idx=PAD_IDX)
            self.lstm = nn.LSTM(embed_dim, hidden_dim,
                                batch_first=True, bidirectional=True)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim * 2, num_classes)

        def forward(self, x):
            emb = self.embedding(x)
            _, (h_n, _) = self.lstm(emb)
            return self.fc(
                self.dropout(torch.cat([h_n[0], h_n[1]], dim=1)))

    textcnn = TextCNN(vocab_size, EMBED_DIM)
    textcnn.load_state_dict(
        torch.load(asset("textcnn.pt"), map_location="cpu"))
    textcnn.eval()

    bilstm = BiLSTMClassifier(vocab_size, EMBED_DIM)
    bilstm.load_state_dict(
        torch.load(asset("bilstm.pt"), map_location="cpu"))
    bilstm.eval()

    return {
        "sess": sess,
        "input_name": input_name,
        "output_name": output_name,
        "tfidf": tfidf,
        "word2idx": word2idx,
        "PAD_IDX": PAD_IDX,
        "MAX_LEN": MAX_LEN,
        "weights": weights_data["weights"],
        "lookup": lookup,
        "deberta_tok": deberta_tok,
        "deberta_mdl": deberta_mdl,
        "textcnn": textcnn,
        "bilstm": bilstm,
    }

with st.spinner("Loading models... (first load takes ~30 seconds)"):
    assets = load_everything()

# ---- helpers ----
def clean(s):
    return re.sub(r"\s+", " ", str(s)).strip()

def build_text(prompt, a, b, c, d, e):
    return (f"Question: {prompt}\n"
            f"A) {a}\nB) {b}\nC) {c}\nD) {d}\nE) {e}")

def tokenize(text):
    return re.findall(r"[a-zA-Z]+", text.lower())

def encode(text):
    w2i = assets["word2idx"]
    PAD = assets["PAD_IDX"]
    ML  = assets["MAX_LEN"]
    ids = [w2i.get(w, PAD) for w in tokenize(text)][:ML]
    ids += [PAD] * (ML - len(ids))
    return torch.tensor([ids])

@torch.no_grad()
def deberta_probs(text):
    tok, mdl = assets["deberta_tok"], assets["deberta_mdl"]
    enc = tok(text, truncation=True, max_length=384, return_tensors="pt")
    return F.softmax(mdl(**enc).logits, dim=-1)[0].numpy()

@torch.no_grad()
def custom_probs(model, text):
    return F.softmax(model(encode(text)), dim=-1)[0].numpy()

def classical_probs(prompt, options):
    tfidf = assets["tfidf"]
    pv    = tfidf.transform([prompt])
    sims  = {
        L: cosine_similarity(pv, tfidf.transform([options[L]]))[0][0]
        for L in LETTERS
    }
    srow  = np.array([sims[L] for L in LETTERS])
    total = srow.sum() + 1e-9
    order = np.argsort(-srow)
    rank  = {LETTERS[order[r]]: r + 1 for r in range(5)}
    lens  = [len(options[L]) for L in LETTERS]

    probs = []
    for L in LETTERS:
        ot  = options[L]
        avg = np.mean([sims[o] for o in LETTERS if o != L])
        feat = np.array([[
            sims[L], rank[L], sims[L] / total, srow.max() - sims[L],
            len(ot), len(ot.split()), 5 - rank[L] + 1,
            avg, sims[L] - avg, len(ot) / (sum(lens) + 1e-9),
        ]], dtype=np.float32)

        # run through ONNX session
        result = assets["sess"].run(
            [assets["output_name"]],
            {assets["input_name"]: feat}
        )[0]
        # result is shape (1, 2) → prob of class 1
        probs.append(float(result[0][1]))

    probs = np.array(probs)
    return probs / (probs.sum() + 1e-9)

def predict(prompt, a, b, c, d, e):
    prompt, a, b, c, d, e = map(clean, [prompt, a, b, c, d, e])
    options = {"A": a, "B": b, "C": c, "D": d, "E": e}
    text    = build_text(prompt, a, b, c, d, e)

    # exact match override
    key = (prompt, a, b, c, d, e)
    if key in assets["lookup"]:
        return assets["lookup"][key], None, True

    p1 = classical_probs(prompt, options)
    p2 = deberta_probs(text)
    p4 = custom_probs(assets["textcnn"], text)
    p5 = custom_probs(assets["bilstm"],  text)

    w        = assets["weights"]
    w_active = np.array([w[0], w[1], w[3], w[4]])
    w_active = w_active / w_active.sum()
    final    = sum(wt * p for wt, p in zip(w_active, [p1, p2, p4, p5]))

    ranked    = sorted(zip(LETTERS, final), key=lambda x: -x[1])
    top3      = " ".join(l for l, _ in ranked[:3])
    breakdown = {l: round(float(p), 4) for l, p in ranked}
    return top3, breakdown, False

# ---- UI ----
st.title(" Smart MCQ Solver")
st.caption("Ensemble: TF-IDF+GBM · DeBERTa-v3-small · Text-CNN · BiLSTM")
st.markdown("---")

prompt = st.text_area(" Question / Prompt", height=100,
                      placeholder="Paste your question here...")
col1, col2 = st.columns(2)
with col1:
    a = st.text_input("Option A")
    c = st.text_input("Option C")
    e = st.text_input("Option E")
with col2:
    b = st.text_input("Option B")
    d = st.text_input("Option D")

if st.button("🔍 Predict", type="primary"):
    if not prompt.strip():
        st.warning("Please enter a question first.")
    elif not all([a, b, c, d, e]):
        st.warning("Please fill in all 5 options.")
    else:
        with st.spinner("Predicting..."):
            top3, breakdown, exact = predict(prompt, a, b, c, d, e)
        if exact:
            st.success(f"**Top-3 Answers: {top3}**")
            st.info("⚡ Exact match found in training data")
        else:
            st.success(f"**Top-3 Predicted Answers: {top3}**")
            st.markdown("**Probability breakdown:**")
            for letter, prob in breakdown.items():
                st.progress(prob, text=f"{letter}: {prob:.4f}")

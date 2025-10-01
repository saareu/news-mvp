import argparse
import pathlib
import json
import glob
import re
import time
import math
from collections import Counter, defaultdict
from typing import List

import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import umap
import hdbscan
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer


def precompute_token_strings(docs: list[str]) -> dict[int, str]:
    # Use fast analyzer for all docs
    return {i: " ".join(he_analyzer(d)) for i, d in enumerate(docs)}


def bigram_df_in_docs(
    bg: str, doc_ids: list[int], token_strings: dict[int, str]
) -> int:
    w = " " + bg + " "
    return sum(1 for i in doc_ids if w in (" " + token_strings[i] + " "))


def accept_phrase(
    phrase: str,
    topic_doc_ids: list[int],
    other_doc_ids: list[int],
    token_strings: dict[int, str],
    min_df_abs: int,
    min_df_ratio: float = 0.06,
    max_out_ratio: float = 0.02,
) -> bool:
    toks = phrase.split()
    if len(toks) < 2:
        return False
    bgs = [" ".join(toks[i : i + 2]) for i in range(len(toks) - 1)]
    td = len(topic_doc_ids)
    od = max(1, len(other_doc_ids))
    for bg in bgs:
        df_t = bigram_df_in_docs(bg, topic_doc_ids, token_strings)
        df_o = bigram_df_in_docs(bg, other_doc_ids, token_strings)
        if df_t < max(min_df_abs, int(min_df_ratio * td)):
            return False
        if (df_o / od) > max_out_ratio:
            return False
    return True


def mmr_select(
    candidates: List[str],
    cand_vecs: "np.ndarray",
    centroid: "np.ndarray",
    k: int = 3,
    lambda_div: float = 0.6,
) -> List[str]:
    # Normalize
    cand_vecs = cand_vecs.astype("float32")
    cand_vecs /= np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-12
    c = centroid.astype("float32")
    c /= np.linalg.norm(c) + 1e-12

    sim_to_centroid = cand_vecs @ c
    selected = []
    selected_idx = []

    mask = np.arange(len(candidates)).tolist()
    while mask and len(selected) < k:
        if not selected_idx:
            i = int(np.argmax(sim_to_centroid[mask]))
            pick = mask[i]
        else:
            # diversity = max cosine to already selected
            S = cand_vecs[mask] @ cand_vecs[selected_idx].T
            max_sim = S.max(axis=1)
            scores = (1 - lambda_div) * sim_to_centroid[mask] - lambda_div * max_sim
            pick = mask[int(np.argmax(scores))]
        selected.append(candidates[pick])
        selected_idx.append(pick)
        mask.remove(pick)
    return selected


# --- Hebrew label-only normalization and verbish filtering ---
def norm_he_token(tok: str) -> str:
    if len(tok) >= 6 and tok[0] in HE_PREFIXES:
        tok = tok[1:]
    return tok


def looks_verbish(tok: str) -> bool:
    return tok.endswith(("ו", "נו", "תי", "ת"))


HE_ONLY = re.compile(r"^[א-ת\"׳״–\-]{2,}$")


def is_good_he_token(tok: str) -> bool:
    return bool(HE_ONLY.match(tok)) and tok not in HE_STOP


def heb_frac(s: str) -> float:
    if not s:
        return 0.0
    he = sum("א" <= ch <= "ת" for ch in s)
    return he / max(1, len(s))


def clean_phrase_pos(s: str) -> str:
    toks = []
    for t in s.split():
        t = norm_he_token(t)
        if not is_good_he_token(t):
            continue
        if looks_verbish(t):
            continue  # drop likely verbs in labels
        if t in BANNED_LABEL_TOKENS or len(t) < 3:
            continue
        toks.append(t)
    return " ".join(toks[:3])


def clean_phrase(s: str) -> str:
    toks = [t for t in s.split() if is_good_he_token(t)]
    toks = [t for t in toks if t not in BANNED_LABEL_TOKENS and len(t) >= 3]
    return " ".join(toks[:3])


def topic_docs_by_id(docs: List[str], topics: np.ndarray) -> dict[int, List[str]]:
    buckets = defaultdict(list)
    for d, t in zip(docs, topics):
        if t != -1:
            buckets[int(t)].append(d)
    return buckets


def tokens(doc: str) -> List[str]:
    return list(he_analyzer(doc))


# --- Best bigram by lift for fallback labeling ---
def best_lift_bigram(docs: List[str], global_counts: Counter) -> str:
    # Find the best bigram by lift in the given docs
    toks = [tok for d in docs for tok in he_analyzer(d)]  # fast analyzer
    if len(toks) < 2:
        return ""
    bigrams = [" ".join(pair) for pair in zip(toks, toks[1:])]
    c = Counter(bigrams)

    def lift(bg):
        try:
            w1, w2 = bg.split()
        except ValueError:
            return 0.0
        return c[bg] / math.sqrt(
            (global_counts.get(w1, 1)) * (global_counts.get(w2, 1))
        )

    for bg, _ in sorted(c.items(), key=lambda kv: (lift(kv[0]), kv[1]), reverse=True)[
        :20
    ]:
        ph = clean_phrase_pos(bg)
        if heb_frac(ph) >= 0.7 and len(ph.split()) >= 2:
            return ph
    return ""


# Hebrew text processing
HE_PREFIXES = ("ו", "ב", "ל", "כ", "מ", "ה", "ש")  # vav/bet/lamed/kaf/mem/he/shin


# Hebrew stopwords (merged + expanded)
HE_STOP = set(
    """
של על עם זה זו גם כל הוא היא הם הן האם כי יש אין לא כן עוד עד אבל מאוד כאן שם יותר פחות כך וכן ואת אותו אחרי וידאו צפו תמונה
אני אתה את אתם אתן אנו אנחנו שלי שלך שלו שלה שלנו שלכם שלכן שלהן אותם אותן לכן לכם לנו אצל בתוך אתהם
זהו זוהי אלו אלה יהיה יהיו היתה היו הייתי היית הייתם הייתן היינו להיות
את לכם שלכם הללו הזה הזאת בעבר לשעבר מצב חשוב היום אמש אתמול מחר כעת עכשיו בקרוב בשעה בשעות הבוקר הערב הלילה השבוע החודש השנה
איתם ניתן לגבי חייב צריך אפשר אולי ייתכן ככל הנראה כנראה אמור אמורים צפוי צפויה
אליהם מהם ולא בהם עליהם להם בלי ידי אמר אומר לפי לעומת אומרים לפיכך בגלל בשל לאור עימם באמצעות בעקבות במסגרת תחת נגד בין לבין מול עבור לגבי
ככל כיום היינו תוך בנוסף אלא בכל באופן מספר רבים מעט פחות יותר למשל כגון למשל כמו כן מנגד מאידך לעומת זאת מצד שני בסך הכל
משרד ממשלת הממשלה הרשויות רשויות מערכת כתבת כתב דיווח דיווחים נמסר מסרה מסרו מסר ציין ציינו ציינה הוסיף הוסיפה הוסיפו טען טענה טענו הדגיש הדגישה הדגישו קבע קבעה קבעו
קרוב רחוק נוסף נוסיף בהמשך קודם קודם לכן מוקדם מאוחר בשלב בשלבו בשלב זה מאז בזמן בזמן שה בעת כאשר כשי בזמן אמת
מדיני חרדים חרדי לכניסה יציאה בתוך מחוץ פנימה החוצה סביב אצל מעל מתחת סמוך קרוב רחוק
""".split()
)

# extra high-frequency discourse fillers & newsroom boilerplate
HE_STOP.update(
    {
        "מאז",
        "בשלב",
        "היה",
        "כתב",
        "מדיני",
        "לשעבר",
        "בהמשך",
        "מתי",
        "חוסר",
        "לדברי",
        "לדבריו",
        "לדבריה",
        "לפי",
        "כאמור",
        "כפי",
        "כך",
        "כזה",
        "כאלה",
        "עודכן",
        "עודכנו",
        "עדכן",
        "עדכנה",
        "פרסם",
        "פורסם",
        "פורסמה",
        "פרסמו",
        "דווח",
        "דווחו",
        "מדווח",
        "מדווחים",
        "מדווחת",
        "דיווח",
        "הודיע",
        "הודיעה",
        "הודיעו",
        "הבהיר",
        "הבהירה",
        "נגד",
        "בעד",
        "ללא",
        "עם זאת",
        "יחד עם זאת",
        "כמו",
        "כמו כן",
        "כלל",
        "כולל",
        "וכן",
        "ובכל זאת",
        "באשר",
        "אשר",
        "שבו",
        "שבה",
        "שבהם",
        "שבהן",
        "שלהם",
        "שלהן",
        "שלה",
        "שלו",
        "מאות",
        "אלפים",
        "עשרות",
        "כמה",
        "מספר",
        "מסוימים",
        "מסוימות",
        # time/recency
        "בקרוב",
        "מזה",
        "כבר",
        "כמעט",
        "עת",
        "כעת",
        "כרגע",
        "בינתיים",
        "בתוך",
        "במהלך",
        "בעת",
        "כש",
    }
)

# Add additional stopwords to prevent common leaks into labels
HE_STOP.update(
    {
        "גרסה",
        "גרסת",
        "פרויקט",
        "סדרה",
        "וידיאו",
        "קליפ",
        "תמונת",
        "סטייל",
        "סטייליסטית",
        "סלב",
        "כוכבת",
        "מאדאם",
        "בארי",
        "לוטוס",
        "גרנד",
        "סוויס",
    }
)
# Use this to scrub weird/junk label terms you’ve observed.
BANNED_LABEL_TOKENS = {
    "בחלק",
    "ביג",
    "שיק",
    "לרוב",
    "צד",
    "שמנהל",
    "מאז",
    "בשלב",
    "חשוב",
    "כתב",
    "מדיני",
    "לשעבר",
    "בהמשך",
    "מתי",
    "חוסר",
    "אמר",
    "אומר",
    "דיווח",
    "דיווחים",
    "פורסם",
    "עודכן",
    "נמסר",
    "טען",
    "טענה",
    "ציין",
    "ציינה",
    "קבע",
    "קבעה",
}

# Notes
#
# Keep ngram_range=(1,3) so informative bigrams/trigrams can surface even if one token is in HE_STOP.
#
# Use BANNED_LABEL_TOKENS only when post-processing labels (don’t feed it into the vectorizer), e.g., drop any label token that’s in this set.
#
# Iterate: when you see a new generic token leaking into labels, add it to BANNED_LABEL_TOKENS first; only promote to HE_STOP if it’s truly non-topical across the corpus.


HE_TOKEN_RE = re.compile(r"[א-ת]{2,}", re.UNICODE)


def he_analyzer(text: str):
    for tok in HE_TOKEN_RE.findall(text or ""):
        # strip prefix only for long tokens to avoid "ית/ות" fragments
        if len(tok) >= 6 and tok[0] in HE_PREFIXES:
            tok = tok[1:]
        if tok in HE_STOP or len(tok) < 2:
            continue
        yield tok


# ---- Paths ----
ART = pathlib.Path("artifacts")
ART.mkdir(exist_ok=True)
IDMAP_PATH = ART / "idmap.txt"
MMAP_PATH = ART / "embeddings.mmap"
META_PATH = ART / "embeddings.meta.json"

OUT_DIR = ART / "topics"
OUT_DIR.mkdir(exist_ok=True)


# ---- Data loading ----
def load_window_from_parquet(parquet_glob: str, lang: str, min_len: int = 10):
    matches = glob.glob(parquet_glob, recursive=True)
    if not matches:
        raise FileNotFoundError(
            f"No parquet files matched glob '{parquet_glob}'. "
            "Check the pattern (e.g., use 'data/parquet/**/*.parquet')."
        )

    scan = pl.scan_parquet(parquet_glob).select(
        ["article_id", "description", "language"]
    )
    if lang:
        scan = scan.filter(pl.col("language") == lang)
    scan = scan.filter(pl.col("description").str.len_chars() >= min_len)
    df = scan.collect()
    ids = df["article_id"].to_list()
    texts = df["description"].to_list()
    return ids, texts


def load_embeddings_for_ids(ids):
    if not (IDMAP_PATH.exists() and MMAP_PATH.exists() and META_PATH.exists()):
        raise RuntimeError(
            "Embeddings artifacts missing (idmap/mmap/meta). Run embeddings first."
        )
    idmap = IDMAP_PATH.read_text(encoding="utf-8").splitlines()
    pos = {i: idx for idx, i in enumerate(idmap)}
    meta = json.loads(META_PATH.read_text())
    mmap = np.memmap(
        MMAP_PATH, dtype="float32", mode="r", shape=(meta["rows"], meta["dim"])
    )

    keep_positions, keep_ids = [], []
    for idx, i in enumerate(ids):
        p = pos.get(i)
        if p is not None:
            keep_positions.append((idx, p))
            keep_ids.append(i)
    if not keep_positions:
        raise RuntimeError(
            "No requested IDs found in idmap; check your parquet_glob/window."
        )
    window_idx, embed_idx = zip(*keep_positions)
    X = mmap[list(embed_idx), :]  # (N, dim) float32 normalized vectors
    return keep_ids, X, list(window_idx)


# ---- Topic ID stability (match to previous centroids) ----
def load_prev_centroids():
    path = OUT_DIR / "topic_centroids.npy"
    meta = OUT_DIR / "topic_meta.json"
    if not (path.exists() and meta.exists()):
        return None, None

    try:
        centroids = np.load(path)
    except Exception:
        return None, None

    raw = meta.read_text(encoding="utf-8").strip()
    if not raw:
        return centroids, {}

    try:
        meta_obj = json.loads(raw)
    except json.JSONDecodeError:
        meta_obj = {}

    return centroids, meta_obj


def match_stable_ids(
    prev_vecs,
    prev_meta,
    uniq_topics: list[int],
    new_centroids: np.ndarray,
    sim_thr: float = 0.85,
):
    if not uniq_topics or new_centroids.size == 0:
        return {}, new_centroids, {}

    if (
        prev_vecs is None
        or prev_vecs.size == 0
        or prev_meta is None
        or len(prev_meta) == 0
    ):
        stable_map = {t: t for t in uniq_topics}
        return (
            stable_map,
            new_centroids,
            {t: {"prev": None, "sim": None} for t in uniq_topics},
        )

    S = cosine_similarity(new_centroids, prev_vecs)
    stable_map: dict[int, int] = {}
    info: dict[int, dict[str, float | int | None]] = {}
    used_prev: set[int] = set()
    prev_keys = list(prev_meta.keys())
    idx_map = {t: idx for idx, t in enumerate(uniq_topics)}

    for t in uniq_topics:
        idx = idx_map[t]
        if idx >= S.shape[0]:
            # Skip topics not present in S (degenerate case)
            stable_map[t] = t
            info[t] = {"prev": None, "sim": None}
            continue
        row = S[idx]
        j = int(np.argmax(row))
        sim = float(row[j])
        if sim >= sim_thr and j not in used_prev:
            prev_tid_raw = prev_keys[j] if j < len(prev_keys) else j
            prev_tid = (
                int(prev_tid_raw)
                if isinstance(prev_tid_raw, str)
                else int(prev_tid_raw)
            )
            stable_map[t] = prev_tid
            used_prev.add(j)
            info[t] = {"prev": prev_tid, "sim": sim}
        else:
            stable_map[t] = int(1_000_000 + t)
            info[t] = {"prev": None, "sim": sim}
    return stable_map, new_centroids, info


# ---- Hebrew stopwords are defined at the top of the file ----
# HE_STOP is a set of Hebrew stopwords


# ---- Main clustering ----
def run_clustering(
    parquet_glob: str,
    lang: str = "he",
    n_neighbors: int = 25,
    n_components: int = 8,
    min_cluster_size: int = 15,
    top_n_words: int = 10,
    random_state: int = 42,
):

    # 1) Load texts & embeddings for the window

    t0 = time.time()
    ids, texts = load_window_from_parquet(parquet_glob, lang=lang, min_len=10)
    print(f"Loaded {len(texts)} texts. Time: {time.time()-t0:.1f}s")
    ids, X, keep_indices = load_embeddings_for_ids(ids)
    texts = [texts[idx] for idx in keep_indices]
    print(
        f"Kept {len(texts)} texts after embedding filter. Time: {time.time()-t0:.1f}s"
    )

    # 2) BERTopic: UMAP + HDBSCAN + vectorizer
    umap_model = umap.UMAP(
        n_neighbors=n_neighbors,  # use CLI arg for local connectivity
        n_components=n_components,
        min_dist=0.0,
        metric="cosine",
        random_state=random_state,
    )
    hdb = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,  # CLI arg
        min_samples=1,  # allow small dense regions
        metric="euclidean",
        cluster_selection_epsilon=0.0,  # tighter separation
        cluster_selection_method="leaf",  # produce more fine-grained clusters
        prediction_data=True,  # Enable outlier reassignment
    )
    # Use a very permissive min_df=1 to avoid issues when topics have few docs
    vectorizer = CountVectorizer(
        analyzer=he_analyzer,  # custom analyzer
        ngram_range=(1, 3),  # allow unigrams and phrases
        min_df=1,  # allow terms in at least 1 doc (robust for small topics)
        max_df=1.0,  # allow terms in up to 100% of docs
    )

    # c-TF-IDF with frequent-word reduction
    ctfidf = ClassTfidfTransformer(reduce_frequent_words=True)
    rep = KeyBERTInspired()  # KeyBERT-inspired representation model
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdb,
        vectorizer_model=vectorizer,
        ctfidf_model=ctfidf,  # reduce frequent words
        representation_model=rep,  # diverse KeyBERT
        embedding_model=embedding_model,
        top_n_words=top_n_words,
        language="multilingual",
        calculate_probabilities=True,
        verbose=False,
    )

    topics, _ = topic_model.fit_transform(texts, embeddings=X)
    # Reassign outliers using embeddings if supported. Be defensive: some BERTopic
    # versions raise TypeError or ValueError (e.g., when there are no outliers).
    try:
        topics = topic_model.reduce_outliers(
            texts, topics, strategy="embeddings", threshold=0.1
        )
    except Exception:
        try:
            topics = topic_model.reduce_outliers(texts, topics)
        except Exception:
            # If reduce_outliers isn't available or there are no outliers, continue
            pass
    topics = np.array(topics, dtype=int)

    # ---- Topic ID stability (match to previous centroids) ----
    prev_vecs, prev_meta = load_prev_centroids()
    uniq = sorted([t for t in set(topics) if t != -1])
    centroids = np.stack([X[topics == t].mean(axis=0) for t in uniq], axis=0).astype(
        "float32"
    )
    stable_map, new_centroids, info = match_stable_ids(
        prev_vecs, prev_meta, uniq_topics=uniq, new_centroids=centroids, sim_thr=0.85
    )

    stable_topics = np.array(
        [stable_map[t] if t != -1 else -1 for t in topics], dtype=int
    )

    # 4) Build outputs
    # Assignments
    assignments = pl.DataFrame({"article_id": ids, "topic_id": stable_topics.tolist()})
    assignments_path = OUT_DIR / "assignments.parquet"
    assignments.write_parquet(assignments_path)

    # === Candidate terms from BERTopic c-TF-IDF ===
    candidates_per_topic: dict[int, List[str]] = {}
    for t in uniq:
        words = topic_model.get_topic(t)
        if not words or not isinstance(words, list):
            words = []
        raw_terms = [w for (w, _score) in words]
        # Clean, keep up to, say, 30
        cand = []
        for w in raw_terms:
            w = clean_phrase_pos(w)
            if w and heb_frac(w) >= 0.7:
                cand.append(w)
        candidates_per_topic[t] = list(dict.fromkeys(cand))[:30]

    # === Embedding-based ranking with MMR ===
    emb = embedding_model  # you already created it
    # Build label_docs just for stats (no trimming of embeddings):
    label_docs = texts  # if you have titles later: use title + lead for better labels

    global_counts = Counter(tok for d in label_docs for tok in tokens(d))
    rep = topic_docs_by_id(label_docs, topics)

    print("Precomputing token strings for all docs (fast analyzer)...")
    t_pos = time.time()
    token_strings = precompute_token_strings(label_docs)
    print(
        f"Token strings ready. Time: {time.time()-t_pos:.1f}s (total {time.time()-t0:.1f}s)"
    )
    doc_idx_by_topic = {t: [i for i, tt in enumerate(topics) if tt == t] for t in uniq}
    all_doc_indices = set(range(len(label_docs)))

    # We'll produce a list of top keywords per topic (3-5) and a concise meta-label
    final_labels: dict[int, str] = {}
    final_keywords: dict[int, list[str]] = {}
    print(f"Selecting labels for {len(uniq)} topics...")
    t_labels = time.time()
    # Use a lightweight local keyphrase extractor (CountVectorizer over topic docs)
    from sklearn.feature_extraction.text import CountVectorizer as LocalCountVectorizer

    for t_idx, t in enumerate(uniq):
        if t_idx % 10 == 0 and t_idx > 0:
            print(f"  ...{t_idx} topics done, elapsed {time.time()-t_labels:.1f}s")
        topic_docs = rep[t]
        topic_doc_ids = doc_idx_by_topic[t]
        other_doc_ids = list(all_doc_indices - set(topic_doc_ids))
        centroid = centroids[t_idx].astype("float32")
        keywords = []
        meta_label = None
        # 1. Local keyphrase extraction with CountVectorizer over this topic's docs
        if topic_docs:
            try:
                local_cv = LocalCountVectorizer(
                    analyzer=he_analyzer, ngram_range=(1, 3), min_df=1
                )
                X_loc = local_cv.fit_transform(topic_docs)
                # convert sparse matrix to dense then sum columns
                freqs = np.asarray(X_loc.toarray().sum(axis=0)).ravel()
                terms = np.array(local_cv.get_feature_names_out(), dtype=str)
                if len(terms) > 0:
                    top_n = min(50, len(terms))
                    top_idx = np.argsort(freqs)[-top_n:][::-1]
                    keyphrases = [terms[i] for i in top_idx]
                    # Clean and keep unique
                    keyphrases = [clean_phrase_pos(str(kp)) for kp in keyphrases]
                    keyphrases = [kp for kp in dict.fromkeys(keyphrases) if kp]
                    if keyphrases:
                        kp_vecs = emb.encode(keyphrases, convert_to_numpy=True)
                        kp_vecs = np.asarray(kp_vecs, dtype="float32")
                        # Rank by similarity to topic centroid and filter by coverage
                        sims = (
                            kp_vecs
                            @ centroid
                            / (
                                np.linalg.norm(kp_vecs, axis=1)
                                * np.linalg.norm(centroid)
                                + 1e-12
                            )
                        )
                        # keep top candidates by sim, but enforce some coverage rules
                        cand_idx = np.argsort(sims)[::-1]
                        selected = []
                        for ii in cand_idx:
                            if len(selected) >= 5:
                                break
                            phrase = keyphrases[int(ii)]
                            # require phrase length >=2 and reasonable Hebrew fraction
                            if len(phrase.split()) < 1:
                                continue
                            if heb_frac(phrase) < 0.6:
                                continue
                            # accept if phrase appears in enough of topic docs and not too often outside
                            if accept_phrase(
                                phrase,
                                topic_doc_ids,
                                other_doc_ids,
                                token_strings,
                                min_df_abs=3,
                                min_df_ratio=0.02,
                                max_out_ratio=0.05,
                            ):
                                selected.append((phrase, float(sims[int(ii)])))
                        # If we got selected keyphrases, use them
                        if selected:
                            # keep top 3-5 as keywords
                            kws = [p for p, s in selected][:5]
                            keywords = kws[:5]
                            meta_label = " ".join(keywords[:5])
            except Exception:
                # If local extraction fails for any reason, continue to fallbacks
                keywords = []
                meta_label = None
        # 2. Fallback: if no keywords, use representative sentence -> extract top nouns/phrases via local cv
        if not keywords and topic_docs:
            try:
                doc_vecs = emb.encode(topic_docs, convert_to_numpy=True)
                sims = (
                    doc_vecs
                    @ centroid
                    / (
                        np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(centroid)
                        + 1e-12
                    )
                )
                idx = int(np.argmax(sims)) if len(sims) else 0
                rep_doc = topic_docs[idx]
                # derive keyphrases from representative sentence
                local_cv = LocalCountVectorizer(
                    analyzer=he_analyzer, ngram_range=(1, 3), min_df=1
                )
                X_loc = local_cv.fit_transform([rep_doc])
                freqs = np.asarray(X_loc.toarray().sum(axis=0)).ravel()
                terms = np.array(local_cv.get_feature_names_out(), dtype=str)
                if len(terms) > 0:
                    top_idx = np.argsort(freqs)[-10:][::-1]
                    keyphrases = [clean_phrase_pos(str(terms[i])) for i in top_idx]
                    keyphrases = [kp for kp in dict.fromkeys(keyphrases) if kp]
                    if keyphrases:
                        keywords = keyphrases[:5]
                        meta_label = " ".join(keywords[:5])
            except Exception:
                pass
        # 3. Fallback: best bigram by lift
        if not keywords:
            fb = best_lift_bigram(topic_docs, global_counts)
            if fb:
                keywords = [fb]
                meta_label = fb
            else:
                keywords = ["נושא"]
                meta_label = "נושא"

        # Finalize: ensure concise meta label (max 6 words) and never None
        if meta_label:
            meta_label = " ".join(meta_label.split()[:6])
        else:
            meta_label = "נושא"
        final_labels[t] = meta_label
        final_keywords[t] = keywords

    print(
        f"Label selection done. Time: {time.time()-t_labels:.1f}s (total {time.time()-t0:.1f}s)"
    )
    # Map raw -> stable id
    raw_to_stable = {t: stable_map[t] for t in uniq}
    # Write labels and keyword lists into metadata
    label_rows = [
        {
            "topic_id": raw_to_stable[t],
            "label": final_labels[t],
            "keywords": final_keywords[t],
        }
        for t in uniq
    ]
    labels_df = pl.DataFrame(label_rows)

    # Keep top_terms (for debugging)
    topic_terms = []
    for t in uniq:
        words = topic_model.get_topic(t)
        if not words or not isinstance(words, list):
            words = []
        terms = [w for w, _ in words][:15]
        # attach our final keywords as well for easier inspection
        kws = final_keywords.get(t, [])
        topic_terms.append(
            {"topic_id": raw_to_stable[t], "top_terms": terms, "keywords": kws}
        )

    meta_df = (
        pl.DataFrame(
            [
                {"topic_id": raw_to_stable[t], "size": int((topics == t).sum())}
                for t in uniq
            ]
        )
        .join(labels_df, on="topic_id")
        .join(pl.DataFrame(topic_terms), on="topic_id", how="left")
    )
    meta_path = OUT_DIR / "topics.parquet"
    meta_df.write_parquet(meta_path)

    # Save centroids for stability next runs
    # Align order to meta_df.topic_id
    order = meta_df.select("topic_id").to_series().to_list()
    cent_map = {raw_to_stable[t]: X[topics == t].mean(axis=0) for t in uniq}
    if order:
        C = np.stack([cent_map[t] for t in order], axis=0).astype("float32")
        np.save(OUT_DIR / "topic_centroids.npy", C)
        # store small json meta (topic_id -> label/size)
        tmp = {
            int(tid): {"label": lbl, "size": int(sz)}
            for tid, lbl, sz in zip(order, meta_df["label"], meta_df["size"])
        }
        (OUT_DIR / "topic_meta.json").write_text(
            json.dumps(tmp, ensure_ascii=False), encoding="utf-8"
        )
    else:
        # no topics found
        np.save(
            OUT_DIR / "topic_centroids.npy", np.zeros((0, X.shape[1]), dtype="float32")
        )
        (OUT_DIR / "topic_meta.json").write_text(json.dumps({}), encoding="utf-8")

    # Coverage metric
    coverage = float((stable_topics != -1).mean())
    print(
        {
            "topics": len(order),
            "coverage": coverage,
            "assignments": str(assignments_path),
            "topics_meta": str(meta_path),
        }
    )

    # ---- Combined CSV: full article metadata + cluster info ----
    try:
        # Read the original parquet window and filter to kept ids (ids is keep_ids from earlier)
        full_scan = pl.scan_parquet(parquet_glob)
        full_df = full_scan.filter(pl.col("article_id").is_in(ids)).collect()

        assignments_df = pl.read_parquet(assignments_path)
        # meta_df already contains topic_id and label
        meta_small = meta_df.select(["topic_id", "label"]).unique()

        combined = full_df.join(assignments_df, on="article_id", how="left").join(
            meta_small, on="topic_id", how="left"
        )
        # rename label -> cluster_label and drop original label column
        if "label" in combined.columns:
            # rename label -> cluster_label for clarity
            combined = combined.rename({"label": "cluster_label"})

        # reorder columns so cluster info is near the front
        cols = combined.columns
        front = [c for c in ["article_id", "topic_id", "cluster_label"] if c in cols]
        rest = [c for c in cols if c not in front]
        combined = combined.select(front + rest)

        csv_path = OUT_DIR / "assignments_with_meta.csv"
        # Write using pandas to ensure utf-8-sig BOM is included (works across polars versions)
        try:
            pdf = combined.to_pandas()
            pdf.to_csv(str(csv_path), index=False, encoding="utf-8-sig")
        except Exception:
            # As a last resort, try polars write_csv (utf-8 without BOM)
            combined.write_csv(csv_path)
        print(f"Wrote combined CSV with metadata and cluster labels: {csv_path}")
    except Exception as e:
        print("Failed to write combined CSV:", repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parquet_glob",
        default="data/parquet/**/**/*.parquet",
        help="Window of validated parquet files to cluster (e.g., data/parquet/date=2025-09-*/source=*/**/*.parquet)",
    )
    ap.add_argument("--lang", default="he")
    ap.add_argument("--n_neighbors", type=int, default=25)
    ap.add_argument("--n_components", type=int, default=8)
    ap.add_argument("--min_cluster_size", type=int, default=15)
    ap.add_argument("--top_n_words", type=int, default=10)
    args = ap.parse_args()
    run_clustering(
        parquet_glob=args.parquet_glob,
        lang=args.lang,
        n_neighbors=args.n_neighbors,
        n_components=args.n_components,
        min_cluster_size=args.min_cluster_size,
        top_n_words=args.top_n_words,
    )


if __name__ == "__main__":
    main()

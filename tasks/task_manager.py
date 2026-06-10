"""
DS-43 — Semantik Arama Motoru (sentence-transformers)
Bir bilgi tabanı (knowledge base) üzerinde ANLAM tabanlı arama motoru
kuruyorsun. Klasik anahtar-kelime (TF-IDF) aramasının aksine, burada metni
embedding'e (anlam vektörüne) çevirip cosine similarity ile karşılaştıracaksın.
Böylece "şifremi unuttum" sorgusu, hiçbir ortak kelime olmasa bile
"parola sıfırlama" dokümanını bulabilir.

Teknoloji: sentence-transformers `all-MiniLM-L6-v2` (384 boyutlu cümle
embedding'leri) + cosine similarity (util.cos_sim).

Her fonksiyonun pass kısmını doldur. Testleri çalıştır, hepsi geçene kadar
iterate et: `python watch.py` veya `pytest tests/test_question.py -v`
"""


# 1. Doküman koleksiyonunu yükle (koda gömülü)
def load_documents():
    """
    Arama yapılacak doküman koleksiyonunu döndür (Python listesi, str'ler).

    En az 15 kısa İngilizce doküman olmalı. Bazı dokümanlar AYNI konuyu
    FARKLI kelimelerle anlatmalı (parafraz çiftleri) — embedding'in kelime
    örtüşmesi olmadan anlamı yakaladığını göstermek için.

    Örn:
      "How do I reset my account password?"
      "Steps to recover access if you forgot your login credentials."
      (ikisi de aynı konu, ortak kelime neredeyse yok)

    Returns:
        list[str]: doküman metinleri
    """
    pass


# 2. SentenceTransformer modelini yükle
def build_model():
    """
    'all-MiniLM-L6-v2' modelini yükleyip döndür.

    Returns:
        SentenceTransformer: yüklenmiş model

    İpucu:
    - from sentence_transformers import SentenceTransformer
    - return SentenceTransformer("all-MiniLM-L6-v2")
    """
    pass


# 3. Dokümanları embedding'e çevir
def embed_documents(model, docs):
    """
    Doküman listesini embedding matrisine çevir.

    Args:
        model: build_model'den dönen model
        docs: list[str]

    Returns:
        np.ndarray: (n_docs, 384) boyutlu embedding matrisi

    İpucu:
    - model.encode(docs, convert_to_numpy=True)
    """
    pass


# 4. Sorguyu embedding'e çevir
def embed_query(model, q):
    """
    Tek bir sorgu metnini embedding vektörüne çevir.

    Args:
        model: model
        q: str (sorgu)

    Returns:
        np.ndarray: (384,) boyutlu vektör

    İpucu:
    - model.encode(q, convert_to_numpy=True)
    """
    pass


# 5. Semantik arama — en benzer top_k dokümanı bul
def semantic_search(model, docs, doc_emb, query, top_k=3):
    """
    Sorguya anlamca en yakın top_k dokümanı bul.

    Args:
        model: model
        docs: list[str] (orijinal doküman metinleri)
        doc_emb: embed_documents'ten dönen (n_docs, 384) matris
        query: str (arama sorgusu)
        top_k: kaç sonuç döndürülecek (default 3)

    Returns:
        list[tuple[str, float]]: [(doc, score), ...]
        Skora göre AZALAN sırada (en benzer ilk).

    İpucu:
    - from sentence_transformers import util
    - q_emb = embed_query(model, query)
    - scores = util.cos_sim(q_emb, doc_emb)[0].cpu().numpy()  # (n_docs,)
    - top_idx = np.argsort(scores)[::-1][:top_k]
    - [(docs[i], float(scores[i])) for i in top_idx]
    """
    pass


# 6. Tüm dokümanlar arasında en benzer çifti bul
def most_similar_pair(model, docs):
    """
    Koleksiyondaki en anlamca benzer (doc_i, doc_j) çiftini bul.

    Args:
        model: model
        docs: list[str]

    Returns:
        tuple[str, str, float]: (doc_a, doc_b, similarity_score)

    İpucu:
    - emb = embed_documents(model, docs)
    - sim = util.cos_sim(emb, emb).cpu().numpy()  # (n, n) matris
    - Köşegeni yok say (kendisiyle benzerlik = 1):
        np.fill_diagonal(sim, -1.0)
    - i, j = np.unravel_index(np.argmax(sim), sim.shape)
    - return (docs[i], docs[j], float(sim[i, j]))
    """
    pass


# 7. Bir skor yeterince ilgili mi?
def is_relevant(score, threshold=0.4):
    """
    Cosine similarity skoru eşik üstündeyse 'ilgili' kabul et.

    Args:
        score: float (cosine similarity)
        threshold: float (eşik, default 0.4)

    Returns:
        bool: score >= threshold

    İpucu: return bool(score >= threshold)
    """
    pass


if __name__ == "__main__":
    docs = load_documents()
    model = build_model()
    doc_emb = embed_documents(model, docs)

    print("Doküman sayısı     :", len(docs))
    print("Embedding boyutu   :", doc_emb.shape)

    q = "I forgot my password, how can I log in again?"
    results = semantic_search(model, docs, doc_emb, q, top_k=3)
    print(f"\nSorgu: {q}")
    for doc, score in results:
        print(f"  {score:.3f}  {doc}")

    pair = most_similar_pair(model, docs)
    print(f"\nEn benzer çift ({pair[2]:.3f}):")
    print(f"  - {pair[0]}")
    print(f"  - {pair[1]}")

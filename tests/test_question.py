import pytest
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tasks.task_manager import (
    load_documents, build_model, embed_documents, embed_query,
    semantic_search, most_similar_pair, is_relevant,
)


# ──────────────────────────────────────────────────────
# Modül-seviye cache — testler arası tekrar model yükleme/embed yok
# ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def docs():
    return load_documents()


@pytest.fixture(scope="module")
def model():
    """İlk testte modeli yükler (~80MB indirir/cache'ler), sonraki testler tekrar kullanır."""
    return build_model()


@pytest.fixture(scope="module")
def doc_emb(model, docs):
    return embed_documents(model, docs)


# 1. load_documents — koleksiyon doğru mu
def test_load_documents(docs):
    assert isinstance(docs, list)
    assert len(docs) >= 15
    assert all(isinstance(d, str) and len(d) > 0 for d in docs)


# 2. embed_documents — shape (n, 384)
def test_embed_documents_shape(model, docs, doc_emb):
    assert isinstance(doc_emb, np.ndarray)
    assert doc_emb.shape[0] == len(docs)
    # all-MiniLM-L6-v2 → 384 boyutlu embedding
    assert doc_emb.shape[1] == 384


# 3. embed_query — tek sorgu (384,) vektör
def test_embed_query_shape(model):
    q_emb = embed_query(model, "How do I reset my password?")
    assert isinstance(q_emb, np.ndarray)
    assert q_emb.shape[-1] == 384


# 4. semantic_search — top_k kadar sonuç döner
def test_semantic_search_returns_top_k(model, docs, doc_emb):
    results = semantic_search(model, docs, doc_emb, "free delivery", top_k=3)
    assert isinstance(results, list)
    assert len(results) == 3
    # Her sonuç (doc, score) ikilisi
    for doc, score in results:
        assert isinstance(doc, str)
        assert isinstance(score, float)


# 5. semantic_search — skorlar azalan sıralı
def test_semantic_search_scores_descending(model, docs, doc_emb):
    results = semantic_search(model, docs, doc_emb, "mobile application download", top_k=5)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


# 6. semantik arama — anlamca ilgili dokümanı bulur (kelime örtüşmesi olmadan)
def test_semantic_search_finds_relevant_doc(model, docs, doc_emb):
    # "log in again" → "reset password" dokümanı; kelime örtüşmesi yok ama anlam aynı
    results = semantic_search(
        model, docs, doc_emb,
        "I forgot my password, how can I log in again?",
        top_k=3,
    )
    top_doc = results[0][0].lower()
    # En ilgili doküman parola/giriş ile ilgili olmalı
    assert "password" in top_doc or "login" in top_doc or "credentials" in top_doc
    # Top-1 skoru anlamlı şekilde yüksek (gerçek model ~0.81)
    assert results[0][1] > 0.6


# 7. ilgili sorgu skoru > ilgisiz sorgu skoru
def test_relevant_score_higher_than_irrelevant(model, docs):
    # Hedef doküman: parola sıfırlama
    target = "How do I reset my account password?"
    t_emb = embed_query(model, target)
    from sentence_transformers import util

    relevant_q = "How can I change my password?"
    irrelevant_q = "What is the capital of France?"

    rel = float(util.cos_sim(embed_query(model, relevant_q), t_emb)[0][0])
    irr = float(util.cos_sim(embed_query(model, irrelevant_q), t_emb)[0][0])

    # İlgili sorgu, ilgisiz sorgudan belirgin şekilde daha benzer olmalı
    assert rel > irr
    assert rel > 0.6   # gerçek: ~0.84
    assert irr < 0.4   # gerçek: ~0.09


# 8. most_similar_pair — anlamca aynı çifti bulur
def test_most_similar_pair(model, docs):
    a, b, score = most_similar_pair(model, docs)
    assert isinstance(a, str) and isinstance(b, str)
    assert a != b
    # En benzer çift birbirine çok yakın olmalı (parafraz çiftleri)
    assert score > 0.7


# 9. is_relevant — eşik mantığı
def test_is_relevant():
    assert is_relevant(0.85) is True
    assert is_relevant(0.1) is False
    # Özel eşik
    assert is_relevant(0.5, threshold=0.6) is False
    assert is_relevant(0.65, threshold=0.6) is True


# ──────────────────────────────────────────────────────
# Kaizu skor gönderimi — bu kısma DOKUNMA
# ──────────────────────────────────────────────────────

import requests


def _send_score(user_score):
    """Kaizu API'sine skor gönder. user_id ve project_id kaizu_config'ten gelir."""
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        from kaizu_config import USER_ID, PROJECT_ID
    except ImportError:
        print("⚠️  kaizu_config.py bulunamadı — skor gönderilmeyecek.")
        return

    if USER_ID == 0:
        print("⚠️  kaizu_config.py'de USER_ID=0 — kendi ID'ni yazmadın, skor gönderilmeyecek.")
        return

    url = "https://kaizu-api-8cd10af40cb3.herokuapp.com/projectLog"
    payload = {
        "user_id": USER_ID,
        "project_id": PROJECT_ID,
        "user_score": user_score,
        "is_auto": True,
    }
    try:
        r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
        if r.status_code in (200, 201):
            print(f"✅ Skor gönderildi: {user_score}")
        else:
            print(f"⚠️  Skor gönderilemedi (HTTP {r.status_code})")
    except Exception as e:
        print(f"⚠️  Skor gönderilirken hata: {e}")


class _ResultCollector:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1


def run_tests():
    """Tüm testleri çalıştır + skoru Kaizu'ya gönder."""
    collector = _ResultCollector()
    pytest.main([os.path.dirname(__file__), "-q"], plugins=[collector])
    total = collector.passed + collector.failed
    if total == 0:
        print("Hiç test çalışmadı.")
        return
    user_score = round((collector.passed / total) * 100, 2)
    print(f"\n📊 Toplam başarılı : {collector.passed}/{total}")
    print(f"📊 Skor            : {user_score}")
    _send_score(user_score)


if __name__ == "__main__":
    run_tests()

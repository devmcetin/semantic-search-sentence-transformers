# DS-43 — Semantik Arama Motoru (sentence-transformers)

**Modül**: ML (NLP / Embeddings) • **Süre**: 2-3 saat

## 🎯 Proje Senaryosu

Bir SaaS şirketinde **data scientist** olarak çalışıyorsun. Şirketin yardım merkezinde (help center) yüzlerce doküman var ve kullanıcılar arama yaptığında doğru cevaba ulaşamıyor. Problem şu: kullanıcılar dokümanlardaki kelimeleri **birebir yazmıyor**. Biri "şifremi unuttum, nasıl girerim?" diye arıyor ama doküman "parola sıfırlama adımları" diyor — **ortak kelime neredeyse yok**, klasik anahtar-kelime araması (TF-IDF) bunu kaçırıyor.

Senin görevin: **anlam tabanlı (semantik) bir arama motoru** kurmak. Kelimeleri saymak yerine, her metni bir **embedding**'e (anlam vektörüne) çevireceksin. Anlamca benzer iki cümle, ortak kelimeleri olmasa bile vektör uzayında **birbirine yakın** olur. Sonra **cosine similarity** ile sorguyu en yakın dokümanlarla eşleştireceksin.

Bunun için **sentence-transformers** kütüphanesinin **`all-MiniLM-L6-v2`** modelini kullanacaksın — küçük, hızlı ve cümle benzerliğinde çok güçlü. Her metni **384 boyutlu** bir vektöre çevirir.

> **TF-IDF ile farkı:** TF-IDF kelime örtüşmesine bakar ("password" geçiyor mu?). Embedding **anlama** bakar — "forgot my login" ile "reset password" arasındaki bağı kelime paylaşmadan yakalar. Bu projede tam da bunu göstereceğiz.

Bu projede şunları uygulayacaksın:
- ✅ **Embedding** kavramı (metin → anlam vektörü)
- ✅ **sentence-transformers** ile model yükleme ve `encode`
- ✅ **Cosine similarity** (`util.cos_sim`) ile benzerlik ölçme
- ✅ **Semantik arama** (top-k en yakın doküman)
- ✅ **Parafraz tespiti** (anlamca aynı, kelimece farklı çiftler)
- ✅ **Eşik (threshold)** ile ilgili / ilgisiz ayrımı

## 📦 Proje Kurulumu

```bash
# Fork + clone
git clone <your-fork-url>
cd data-science-project-43

# Virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate          # Windows

# Dependencies
pip install -r requirements.txt

# Auto test runner (dosya değişince çalışır)
python watch.py

# Manuel test
pytest tests/test_question.py -v
```

> **Not — ilk çalıştırma:** `all-MiniLM-L6-v2` modeli (~80MB) ilk testte internetten indirilip cache'lenir. Sonraki çalıştırmalar lokal cache'den okur.

## 🔑 Kaizu Bağlantısı — `kaizu_config.py`

Skorunun Kaizu hesabına yazılması için **`kaizu_config.py`** dosyasını aç ve **`USER_ID`** alanını kendi user_id'nle değiştir:

```python
USER_ID = 0      # ← Kaizu profilinden alıp buraya yaz
PROJECT_ID = 723 # ← Bu projeye ait, dokunma
```

User_id'ni Kaizu profilinden bulabilirsin (Profile → Settings → User ID).

Skor göndermek için tüm testleri toplu çalıştırmalısın:

```bash
python tests/test_question.py
```

Bu komut tüm testleri çalıştırır, **passed/total oranını otomatik Kaizu'ya gönderir**. Geliştirme sırasında `pytest -v` kullanmaya devam edebilirsin (skor göndermez).

## 📚 Veri — Koda Gömülü Doküman Koleksiyonu

Bu projede dış veri seti indirmiyorsun. Arama yapılacak **~20 kısa İngilizce doküman** doğrudan `load_documents()` içinde Python listesi olarak tutulur (FAQ / yardım merkezi tarzı kısa metinler).

Önemli: koleksiyondaki bazı dokümanlar **aynı konuyu farklı kelimelerle** anlatır (parafraz çiftleri). Örn:

```
"How do I reset my account password?"
"Steps to recover access if you forgot your login credentials."
```

İkisi de aynı anlama gelir ama **neredeyse hiç ortak kelimeleri yok**. Embedding tabanlı arama bu çiftleri yakalayabilir — anahtar-kelime araması kaçırır. Tam da bu yüzden böyle çiftler eklenmiştir.

## 📋 Görevler (`tasks/task_manager.py`)

`task_manager.py` dosyasındaki **7 fonksiyonu** sırayla doldur.

1. **`load_documents()`** — koda gömülü ~20 kısa dokümanı liste olarak döndür
2. **`build_model()`** — `SentenceTransformer("all-MiniLM-L6-v2")` yükle
3. **`embed_documents(model, docs)`** — dokümanları `(n, 384)` ndarray'e çevir
4. **`embed_query(model, q)`** — tek sorguyu `(384,)` vektöre çevir
5. **`semantic_search(model, docs, doc_emb, query, top_k=3)`** — cosine similarity ile en yakın top-k doküman, `[(doc, score), ...]` azalan sırada
6. **`most_similar_pair(model, docs)`** — koleksiyondaki en benzer çift `(doc_a, doc_b, score)`
7. **`is_relevant(score, threshold=0.4)`** — skor eşik üstünde mi (bool)

## 🎓 Öğrenme Hedefleri

Bu projeyi bitirdiğinde:
- [x] **Embedding** ile metni anlam vektörüne çevirebileceksin
- [x] **sentence-transformers** (`all-MiniLM-L6-v2`) kullanabileceksin
- [x] **Cosine similarity** ile iki metnin anlamca benzerliğini ölçebileceksin
- [x] **Semantik arama** (top-k retrieval) kurabileceksin
- [x] Anlamsal aramanın **anahtar-kelime aramasından** neden farklı olduğunu açıklayabileceksin
- [x] **Eşik** ile ilgili/ilgisiz sonuçları ayırabileceksin

## 🧪 Testler

Test dosyası: `tests/test_question.py` (9 test)

Tümü pass olmalı:
- `load_documents` ≥ 15 doküman döner
- `embed_documents` çıktısı `(n, 384)` (384 = MiniLM embedding boyutu)
- `embed_query` çıktısı `(384,)`
- `semantic_search` tam `top_k` sonuç döner ve skorlar **azalan** sıralı
- Anlamca ilgili doküman top-1'de bulunur ("forgot my password" → reset/login dokümanı), kelime örtüşmesi olmadan
- **İlgili sorgu skoru > ilgisiz sorgu skoru** (örn. parola sorgusu ~0.84 vs Fransa sorgusu ~0.09)
- `most_similar_pair` parafraz çiftini bulur (skor > 0.7)
- `is_relevant` eşik mantığı doğru

## 📊 Beklenen Sonuçlar

```
Embedding boyutu : (20, 384)
"forgot my password" sorgusu → top-1: parola/giriş dokümanı (~0.81)
İlgili sorgu skoru  : ~0.65 - 0.84
İlgisiz sorgu skoru : ~0.05 - 0.15
En benzer çift      : bir parafraz çifti (~0.79)
```

## 💡 İpuçları

- Modeli **bir kez** yükle (testler modül-seviye cache'liyor); `encode` görece pahalı.
- `model.encode(docs, convert_to_numpy=True)` → numpy ndarray döndürür.
- `util.cos_sim(a, b)` bir **tensör** döner; `.cpu().numpy()` ile numpy'a çevir.
- Tek sorgu için: `util.cos_sim(q_emb, doc_emb)[0]` → tüm dokümanlara skor vektörü.
- En yüksek skorları sıralamak için: `np.argsort(scores)[::-1][:top_k]`.
- `most_similar_pair`'da köşegeni (kendisiyle benzerlik = 1) yok say: `np.fill_diagonal(sim, -1.0)`.
- Embedding skorları **TF-IDF/Naive Bayes olasılıkları değildir** — cosine similarity [-1, 1] aralığında; ilgili çiftler tipik olarak 0.5+ olur.

## 🚫 Dikkat

- `tests/test_question.py` dosyasını **değiştirme**
- `all-MiniLM-L6-v2` model adını değiştirme (embedding boyutu testi 384 bekler)
- `_solution/` klasörü yok (DB'de saklanır, dersin haftası geçince açılır)
- Dokunabileceğin **2 dosya**: `tasks/task_manager.py` (kodu yaz) + `kaizu_config.py` (sadece USER_ID)

# NLP — objedinjena skripta za kolokvijum

## 1. Analiza podataka

Pre treniranja proveravamo raspodelu klasa i karakteristika.

```python
data['sentiment'].value_counts(normalize=True)
```

- `value_counts()` broji primere svake kategorije; `normalize=True` vraća njihove udele.
- Ako je jedna klasa dominantna, accuracy može zavarati. Zato pratimo **macro F1** i rezultate po klasama.
- `Branch` sa malo kategorija pogodan je za `OneHotEncoder`; `Reviewer_Location` sa mnogo kategorija može proizvesti mnogo kolona.
- Broj komentara po parku nije isto što i raspodela sentimenta unutar parka:

```python
pd.crosstab(data['Branch'], data['sentiment'], normalize='index')
```

**Grafici:** `countplot` prikazuje broj pojavljivanja kategorija; `kdeplot` procenjenu gustinu raspodele numeričkih, uglavnom kontinualnih podataka.

## 2. spaCy preprocessing

```python
def spacy_preprocessing(text):
    tokens = spacy_model(text)
    tokens = [tok for tok in tokens if
              not tok.is_punct and not tok.is_space
              and not tok.like_num and not tok.like_url]
    return [tok.lemma_.lower() for tok in tokens
            if not tok.is_stop or tok.lemma_ == 'not']
```

**Šta imamo → šta radimo → šta dobijamo:** Ulazni tekst → tokenizacija, uklanjanje interpunkcije/razmaka/brojeva/URL-ova, lematizacija i mala slova → lista obrađenih reči. Negaciju `not` čuvamo jer menja sentiment. Ova funkcija **ne pravi numeričke vektore**.

## 3. Vektorizacija — ključne razlike

| Metoda | Šta radi | Rezultat | Primena u vežbama |
|---|---|---|---|
| `CountVectorizer` | Broji pojavljivanja reči i n-grama | Matrica brojeva | BoW + Decision Tree |
| `TfidfVectorizer` | Dodeljuje težine prema zastupljenosti u dokumentu i korpusu | Matrica TF-IDF težina | VSM + Logistic Regression, ključne reči |
| spaCy Word Embeddings | Koristi naučene vektore reči i prosečno ih kombinuje | Gust vektor dokumenta | ReviewVectorizer + LR/SVM |
| `SentenceTransformer` | Pravi naučenu semantičku reprezentaciju celog teksta | Embedding teksta | Sličnost i klasterizacija |

Primer za `['movie', 'good']`:
- CountVectorizer sa rečnikom `['movie','good','bad']` → `[1,1,0]` (brojanja).
- TF-IDF → npr. `[0.6,0.8,0]` (ilustrativne težine).
- Word Embeddings → prosečan vektor reči, npr. `[0.2,-0.4,...]` (ilustrativno).
- SentenceTransformer → vektor celog teksta naučen za semantičko poređenje.

### CountVectorizer

```python
count_vectoriser = CountVectorizer(
    tokenizer=spacy_preprocessing,
    token_pattern=None,
    ngram_range=(1, 2),
    min_df=0.01,
    max_df=0.95
)
```

`tokenizer` koristi našu spaCy obradu; `token_pattern=None` isključuje podrazumevani obrazac; `ngram_range=(1,2)` uključuje unigram i bigram; `min_df=0.01` zahteva pojavljivanje u najmanje 1% dokumenata; `max_df=0.95` uklanja termine u više od 95% dokumenata. Opcioni `max_features` ograničava broj karakteristika. **Ne zna značenje reči**, samo ih broji.

### TfidfVectorizer

```python
tfidf_vectorizer = TfidfVectorizer(
    tokenizer=spacy_preprocessing,
    token_pattern=None,
    ngram_range=(1, 2),
    min_df=0.01
)
```

**TF** opisuje učestalost termina u dokumentu; **IDF** smanjuje značaj termina čestih u mnogim dokumentima. Za razliku od CountVectorizer-a, izlaz nisu sirovi brojevi pojavljivanja već težine. Podrazumevano se u scikit-learn-u izlazni vektori L2 normalizuju.

### ReviewVectorizer — Word Embeddings

```python
class ReviewVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, spacy_pipeline, vector_dim=300):
        self.spacy_pipeline = spacy_pipeline
        self.vector_dim = vector_dim

    def spacy_tokenizer(self, doc):
        tokens = self.spacy_pipeline(doc)
        tokens = [token for token in tokens if
                  not token.is_punct and not token.like_num
                  and not token.like_url and not token.is_space
                  and not token.is_stop]
        return [token.lemma_.lower() for token in tokens]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = np.zeros((len(X), self.vector_dim))
        for i, text in enumerate(X):
            words = self.spacy_tokenizer(text)
            X_transformed[i] = self.spacy_pipeline(' '.join(words)).vector
        return X_transformed
```

- `BaseEstimator`: kompatibilnost sa scikit-learn parametrima i alatima.
- `TransformerMixin`: obezbeđuje `fit_transform()`.
- `fit()` ne uči rečnik ili nove vektore, zato vraća `self`.
- `transform()` pravi matricu `(broj_recenzija, 300)`, obrađuje svaku recenziju, spaja reči u tekst i uzima `doc.vector` (kod modela sa vektorima tokena podrazumevano njihov prosek).
- **Razlika u preprocessing-u:** ova verzija uklanja sve stop reči, a prethodna čuva `not`.
- **Ključna razlika od CountVectorizer-a:** ne broji reči, već koristi već naučene numeričke reprezentacije reči.

### SentenceTransformer

```python
model = SentenceTransformer('all-MiniLM-L12-v2')
embeddings = model.encode(np.array(texts), normalize_embeddings=True)
```

SentenceTransformer je treniran da embedding celog teksta bude koristan za semantičko poređenje. `normalize_embeddings=True` normalizuje nenulte vektore na dužinu 1; kosinusno poređenje tada zavisi od smera, a ne od veličine vektora.

## 4. Klasifikatori

```python
DecisionTreeClassifier(random_state=RAND_STATE,
                       min_impurity_decrease=0.001)
LogisticRegression(random_state=RAND_STATE, max_iter=2000)
SVC(kernel='linear', C=1, random_state=RAND_STATE,
    decision_function_shape='ovo')
```

- **Decision Tree:** gradi stablo uslova; `min_impurity_decrease` određuje najmanje smanjenje nečistoće za novu podelu.
- **Logistic Regression:** uči linearnu granicu klasifikacije, optimizujući logistički gubitak; `max_iter` ograničava iteracije.
- **Linear SVM:** traži granicu sa velikom marginom; `kernel='linear'` bira linearni kernel; `C` kontroliše kompromis između margine i grešaka. `decision_function_shape='ovo'` određuje oblik izlaza decision funkcije za više klasa.
- LR i linearni SVM mogu dati linearne granice, ali koriste različite funkcije gubitka.

## 5. Pipeline

```python
pipe = Pipeline([
    ('vectorizer', tfidf_vectorizer),
    ('classifier', lr_classifier)
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

`fit()` prvo uči vektorizator i transformiše trening tekstove, zatim trenira klasifikator. `predict()` koristi **isti naučeni vektorizator**, pa predviđa klase. Tako izbegavamo da na test podacima učimo novi rečnik.

```python
pipe.named_steps
pipe.named_steps['classifier'].get_params()
tfidf_step = pipe.named_steps['vectorizer']
tfidf_output = tfidf_step.transform(X_test)
```

`named_steps` pristupa koracima po nazivu. `transform()` koristi već naučeno stanje.

## 6. GridSearchCV i RandomizedSearchCV

```python
param_grid = {'classifier__C': [0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(svm_pipe, param_grid=param_grid,
                           scoring='f1_macro', cv=5)
grid_search.fit(X_train, y_train)
```

**GridSearchCV** ispituje sve kombinacije. Pet vrednosti `C` × pet CV foldova = 25 treniranja za ocenjivanje. Svaka kombinacija se trenira na četiri dela, proverava na petom, pet puta; bira se najbolji prosečan macro F1. Podrazumevano `refit=True` dodatno trenira najbolju kombinaciju na celom trening skupu.

```python
grid_search.best_params_      # najbolji hiperparametri
grid_search.best_score_       # najbolji prosečan CV rezultat
grid_search.best_estimator_   # već refitovan najbolji Pipeline
predictions = grid_search.predict(X_test)
# ili:
best_pipe = grid_search.best_estimator_
predictions = best_pipe.predict(X_test)
```

Ne moramo ponovo ručno trenirati najbolji model. Ali ako promenimo parametre **originalnog** `svm_pipe`, moramo ponovo pozvati njegov `fit()`.

```python
random_search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_distributions,
    n_iter=10,
    scoring='f1_macro',
    random_state=RAND_STATE
)
```

**RandomizedSearchCV** bira nasumične kombinacije umesto svih, korisno kada je prostor parametara velik. `n_iter=10` znači deset pokušaja (ako prostor pretrage to dozvoljava). I ovde podrazumevano postoji `best_estimator_` i `refit=True`.

Nazivi parametara u ugnježdenom pipeline-u koriste `__`, npr. `vectorizer__tf_idf__min_df` ili `classifier__C`, prema stvarnim imenima koraka.

## 7. Evaluacija klasifikacije

- **Accuracy:** udeo svih tačnih predikcija.
- **Precision:** od predviđenih primera neke klase, koliko je stvarno te klase.
- **Recall:** od stvarnih primera klase, koliko smo prepoznali.
- **F1:** harmonijska sredina precision i recall.
- **Macro F1:** običan prosek F1 po klasama; svaka klasa jednako važna.
- **Weighted F1:** prosek ponderisan brojem primera po klasama.
- **Balanced accuracy:** prosečan recall po klasama; u standardnoj višeklasnoj klasifikaciji jednak macro recall-u.
- **Confusion matrix:** tabela stvarnih i predviđenih klasa; pokazuje gde model greši.

Kod nebalansiranih klasa gledamo macro F1, balanced accuracy i rezultate po klasama, ne samo accuracy.

## 8. Sličnost tekstova

```python
similarities = cosine_similarity(embeddings)
```

Kosinusna sličnost poredi smer vektora: 1 znači isti smer, 0 ortogonalne, -1 suprotne smerove. Za četiri teksta dobijamo matricu `(4,4)`. Kosinusno rastojanje je `1 - cosine_similarity`.

```python
n_news = similarities.shape[0]
pairs_scores = [
    {'pair': i, 'score': similarities[sample_index, i]}
    for i in range(n_news) if i != sample_index
]
sorted_pair_scores = sorted(pairs_scores,
                            key=lambda pair: pair['score'], reverse=True)
```

`shape[0]` je broj redova, a `shape[1]` broj kolona. `sorted(..., reverse=True)` stavlja najveću sličnost prvu. Indeksi NumPy embeddinga i DataFrame-a odgovaraju samo ako je redosled usklađen; nakon filtriranja je često koristan `reset_index(drop=True)`.

## 9. Hijerarhijsko klasterovanje

```python
distances = pdist(embeddings, metric='cosine')
review_linkage = sch.linkage(distances, method='ward')
```

- `pdist()` računa rastojanja za svaki jedinstveni par i vraća jednodimenzionalni **condensed** niz, ne punu kvadratnu matricu.
- `linkage()` vrši aglomerativno hijerarhijsko klasterovanje: počinjemo sa po jednim dokumentom u svakom klasteru i postepeno spajamo klastere.
- Ward metoda minimizuje porast varijanse unutar klastera **za euklidska rastojanja**. Kombinacija kosinusnih rastojanja i `ward` može se izvršiti, ali nema standardno Wardovo tumačenje; uz kosinusna rastojanja razmotriti `average` ili `complete`.

```python
labels = sch.fcluster(review_linkage, t=3, criterion='maxclust')
```

`fcluster()` uzima istoriju spajanja i dodeljuje oznake klastera; `t=3` uz `maxclust` traži najviše tri klastera. Oznake standardno počinju od 1, npr. `[1, 2, 1, 3]`.

## 10. Silhouette score

Silhouette ocenjuje koliko je primer blizu sopstvenom klasteru u odnosu na najbliži drugi klaster:

\[s = \frac{b-a}{\max(a,b)}\]

- `a`: prosečno rastojanje do ostalih primera u istom klasteru.
- `b`: najmanje prosečno rastojanje do primera drugog klastera.
- Blizu 1 = dobro uklapanje; blizu 0 = granica; negativno = moguće pogrešan klaster.

```python
silhouette_score(cos_dist_clipped, labels, metric='precomputed')
```

`metric='precomputed'` koristimo jer prosleđujemo **već izračunatu kvadratnu matricu rastojanja**, ne originalne vektore. `cos_dist_clipped` uklanja sitne negativne numeričke greške u rastojanjima.

## 11. Ključne reči klastera

```python
cluster_reviews = data.loc[data.clust == cluster, 'review_content']
tfidf_vectoriser = TfidfVectorizer(
    ngram_range=(2, 2), stop_words='english',
    min_df=0.01, max_df=0.95
)
reviews_dtm = tfidf_vectoriser.fit_transform(cluster_reviews)
candidate_kwds = tfidf_vectoriser.get_feature_names_out()
avg_tfidf_score = np.array(reviews_dtm.mean(axis=0)).flatten()
```

1. `data.clust == cluster` pravi Boolean masku; `loc` izdvaja recenzije iz izabranog klastera i vraća Series.
2. TF-IDF sa `(2,2)` koristi samo bigrame.
3. `get_feature_names_out()` vraća nazive karakteristika, tj. bigrama.
4. `mean(axis=0)` računa prosečan TF-IDF **svakog bigrama kroz sve recenzije**.
5. `np.array()` pretvara matricu u NumPy niz; `.flatten()` pretvara oblik `(1,n)` u `(n,)`.

Primer: `[[0.2, 0.5, 0.1]]` → `[0.2, 0.5, 0.1]`.

## 12. NumPy / pandas podsetnik

| Operacija | Značenje |
|---|---|
| `shape[0]` / `shape[1]` | Broj redova / kolona |
| `mean(axis=0)` | Prosek po kolonama |
| `mean(axis=1)` | Prosek po redovima |
| `np.asarray()` | Pretvara ulaz u NumPy niz po potrebi |
| `.flatten()` | Pravi jednodimenzionalnu kopiju |
| `.iloc` | Pristup po poziciji |
| `.loc` | Pristup po oznaci ili Boolean maski |
| `.reset_index(drop=True)` | Novi uzastopni indeks, bez čuvanja starog |
| `enumerate(X)` | Parovi `(indeks, element)` |
| `.nunique()` | Broj različitih vrednosti |
| `.value_counts()` | Broj pojavljivanja svake vrednosti |

## 13. Brzo ponavljanje — šta biramo?

| Zahtev zadatka | Metoda |
|---|---|
| Brojanje reči | `CountVectorizer` |
| Težine reči | `TfidfVectorizer` |
| Lematizacija | spaCy `lemma_` |
| Prosečan vektor reči | `ReviewVectorizer` |
| Embedding celog teksta | `SentenceTransformer` |
| Stablo odluka | `DecisionTreeClassifier` |
| Linearna logistička klasifikacija | `LogisticRegression` |
| Linearni SVM | `SVC(kernel='linear')` |
| Povezivanje koraka | `Pipeline` |
| Sve kombinacije parametara | `GridSearchCV` |
| Nasumične kombinacije | `RandomizedSearchCV` |
| Najbolji trenirani model | `best_estimator_` |
| Kosinusna sličnost | `cosine_similarity` |
| Rastojanja parova | `pdist` |
| Hijerarhijsko spajanje | `linkage` |
| Konkretne oznake klastera | `fcluster` |
| Kvalitet podele na klastere | `silhouette_score` |
| Ključne reči klastera | TF-IDF + prosečne težine |

### Cela priča u dve rečenice

**Klasifikacija:** imamo poznate oznake klasa; obrađujemo tekst, vektorizujemo ga, treniramo klasifikator i merimo kvalitet predikcije.

**Klasterizacija:** nemamo unapred zadate klase; pravimo embeddinge, merimo rastojanja, grupišemo slične tekstove, proveravamo kvalitet klastera i izdvajamo ključne reči.

# KNN Testing and Evaluation

Ovaj direktorijum sadrži test programe i alate za evaluaciju KNN algoritama.

## Struktura

- `test_knn_basic.cpp` - Test program za osnovni KNN klasifikator
- Metrike implementirane u `../include/utils/metrics.h`
- Python vizualizacija u `../visualization/visualize_metrics.py`

## Kako koristiti

### 1. Kompajliranje

```bash
# Dodaj u CMakeLists.txt:
add_executable(test_knn_basic tests/test_knn_basic.cpp)
target_link_libraries(test_knn_basic knn_lib utils_lib)

# Build
mkdir build && cd build
cmake ..
make
```

### 2. Pokretanje testa

```bash
# Osnovno korišćenje
./test_knn_basic dataset.csv 5

# Sa opcijama
./test_knn_basic iris.csv 5 --auto-encode --distance manhattan --output metrics.json

# Kategorički podaci
./test_knn_basic titanic.csv 7 --auto-encode --distance euclidean
```

### 3. Opcije

| Opcija | Opis | Primer |
|--------|------|--------|
| `--no-header` | CSV nema header red | `--no-header` |
| `--auto-encode` | Automatski one-hot enkoduj kategoričke kolone | `--auto-encode` |
| `--distance <type>` | Metrika: euclidean, manhattan, hamming, minkowski | `--distance manhattan` |
| `--minkowski-p <p>` | Parametar p za Minkowski (default: 2.0) | `--minkowski-p 3.0` |
| `--test-ratio <r>` | Procenat test skupa (default: 0.2) | `--test-ratio 0.3` |
| `--output <file>` | JSON fajl za metrike (default: metrics.json) | `--output my_metrics.json` |

### 4. Metrike koje se izračunavaju

Program izračunava sledeće metrike:

- **Accuracy** - Ukupna tačnost klasifikacije
- **Precision** - Preciznost po klasama (TP / (TP + FP))
- **Recall** - Odziv po klasama (TP / (TP + FN))
- **F1-Score** - Harmonijska sredina precision i recall
- **Confusion Matrix** - Matrica konfuzije
- **ROC Curve** - ROC kriva (podaci za crtanje)

### 5. Vizualizacija rezultata

```bash
# Vizualizuj jedan algoritam
python visualization/visualize_metrics.py metrics.json

# Uporedi dva algoritma (npr. Basic KNN vs KD-Tree KNN)
python visualization/visualize_metrics.py metrics_basic.json metrics_kdtree.json
```

Python skripta generiše:
- **Confusion Matrix** heatmap
- **Precision, Recall, F1-Score** bar charts
- **ROC krive** za sve klase
- **Poređenje** više algoritama na istom grafu

### 6. Primer output-a

```
=== KNN Basic Classifier Test ===
Dataset: iris.csv
k: 5
Auto-encode: Yes
Test ratio: 20%

Loading dataset...
Loaded with automatic categorical encoding
Total samples: 150
Dimensions: 4

Splitting dataset...
Training samples: 120
Test samples: 30

Training KNN...
Training time: 2 ms

Testing KNN...
Testing time: 45 ms
Average prediction time: 1.5 ms/sample

=== Classification Metrics ===

Accuracy: 96.67%

Per-class metrics:
     Class      Precision         Recall       F1-Score
-------------------------------------------------------
         0         1.0000         1.0000         1.0000
         1         0.9000         1.0000         0.9474
         2         1.0000         0.9000         0.9474

Confusion Matrix:
True\Pred          0          1          2
        0         10          0          0
        1          0         10          0
        2          0          1          9

Metrics saved to: metrics.json
```

### 7. Primeri dataseta

#### Iris (4D, 3 klase, numerički)
```bash
./test_knn_basic iris.csv 5 --distance euclidean
```

#### Titanic (mixed categorical + numeric)
```bash
./test_knn_basic titanic.csv 7 --auto-encode --distance manhattan
```

#### Wine (13D, numerički)
```bash
./test_knn_basic wine.csv 3 --distance minkowski --minkowski-p 3.0
```

### 8. Format JSON izlaza

```json
{
  "algorithm": "KNN_Basic",
  "accuracy": 0.9667,
  "precision": {"0": 1.0, "1": 0.9, "2": 1.0},
  "recall": {"0": 1.0, "1": 1.0, "2": 0.9},
  "f1_score": {"0": 1.0, "1": 0.9474, "2": 0.9474},
  "confusion_matrix": {
    "0": {"0": 10, "1": 0, "2": 0},
    "1": {"0": 0, "1": 10, "2": 0},
    "2": {"0": 0, "1": 1, "2": 9}
  },
  "roc_curve": {
    "0": [{"fpr": 0.0, "tpr": 1.0, "threshold": 0.5}],
    "1": [{"fpr": 0.0, "tpr": 1.0, "threshold": 0.5}],
    "2": [{"fpr": 0.05, "tpr": 0.9, "threshold": 0.5}]
  }
}
```

## Poređenje algoritama

Za poređenje Basic KNN vs KD-Tree KNN:

```bash
# Test basic KNN
./test_knn_basic dataset.csv 5 --output metrics_basic.json

# Test KD-Tree KNN (kasnije)
./test_knn_kdtree dataset.csv 5 --output metrics_kdtree.json

# Uporedi rezultate
python visualization/visualize_metrics.py metrics_basic.json metrics_kdtree.json
```

Python će generisati grafik sa **dve ROC krive** na istom grafu za direktno poređenje!

## Python zavisnosti

```bash
pip install matplotlib seaborn numpy
```

## Napomene

- **One-hot encoding** automatski detektuje kategoričke kolone
- **Test/Train split** se randomizuje (seed=42 za reproduktivnost)
- **ROC kriva** koristi One-vs-Rest pristup za multi-class
- **Metrike** se računaju odvojeno za svaku klasu

# Optimizacija k-NN Algoritma pomoću k-d Stabala

Projekat implementira i poredi različite optimizacije k-NN algoritma koristeći k-dimenzionalna stabla.

## Struktura Projekta

```
.
├── include/              # Header fajlovi
│   ├── kdtree/          # k-d stablo implementacije
│   ├── knn/             # k-NN implementacije
│   └── utils/           # Pomoćne funkcije (distance metrike, učitavanje podataka)
├── src/                 # Implementacioni fajlovi
│   ├── kdtree/          # k-d stablo
│   ├── knn/             # k-NN algoritmi
│   ├── optimizations/   # Optimizacije
│   └── utils/           # Pomoćne funkcije
├── benchmarks/          # Testiranje performansi
│   ├── include/         # Benchmark header-i
│   └── src/             # Benchmark implementacije
├── tests/               # Unit testovi
├── datasets/            # Skupovi podataka za testiranje
├── visualization/       # Alati za vizualizaciju
├── docs/                # Dokumentacija
└── results/             # Rezultati eksperimenata
    ├── benchmark_results/  # Rezultati testova performansi
    └── figures/         # Generisani grafici i plot-ovi
```

## Implementirane Funkcionalnosti

### 1. k-d Stablo Operacije
- Konstrukcija stabla (build)
- Umetanje čvora (insert)
- Brisanje čvora (delete)
- Pretraga najbližih susjeda (k-NN search)
- Prikaz strukture stabla

### 2. k-NN Algoritmi
- Klasični k-NN (brute-force)
- k-NN sa k-d stablom
- Različite distance metrike (Euclidean, Manhattan, Minkowski)
- Podrška za train/test split
- Predviđanje klasa pojedinačnih i grupnih uzoraka

### 3. Optimizacije
- Revised k-d tree (smanjenje nepotrebnih kalkulacija distanci)
- QuickNN (optimizacija memorijskog layouta)
- Različite varijante k-NN algoritma

### 4. Benchmarking i Testiranje
- Poređenje performansi različitih pristupa
- Mjerenje vremena izvršavanja
- Testiranje tačnosti klasifikacije
- Vizualizacija rezultata

## Kompajliranje Projekta

```bash
mkdir build
cd build
cmake ..
make
```

## Pokretanje Testova

### Windows (PowerShell)
```powershell
rm -r build  # Obriši stari build
mkdir build
cd build

# Forsiraj MinGW Makefiles generator
cmake -G "MinGW Makefiles" ..

# Build
mingw32-make

# Pokreni testove
cd tests
./test_kdtree.exe
./test_knn.exe
```

### Linux
```bash
rm -rf build  # Obriši stari build
mkdir build
cd build

cmake ..
make

# Pokreni testove
cd tests
./test_kdtree
./test_knn
```

Za pokretanje benchmark testova i detaljnije informacije o performansama, pogledajte [benchmarks/README.md](benchmarks/README.md).

## Skupovi Podataka

Podaci su organizovani u `datasets/` direktorijumu:
- `letter-recog/` - Letter Recognition dataset
- `covtype/` - Covertype dataset
- `wineqt/` - Wine Quality dataset

## Vizualizacija

U `visualization/` direktorijumu se nalaze Python skripte za vizualizaciju:
- `plot_benchmarks.py` - Vizualizacija benchmark rezultata
- `visualize_metrics.py` - Vizualizacija metrika performansi
- `visualize_kdtree.cpp` - C++ vizualizacija strukture k-d stabla

### Pokretanje Python skripti

```bash
# Instaliraj potrebne pakete
cd visualization
pip install -r requirements.txt

# Pokreni vizualizaciju metrika (jedan ili više JSON fajlova)
python visualize_metrics.py <metrics_file1.json> [metrics_file2.json ...]

# Primjeri:
python visualize_metrics.py metrics_basic.json
python visualize_metrics.py metrics_basic.json metrics_kdtree.json

# Pokreni vizualizaciju benchmark rezultata
python plot_benchmarks.py
```

# KNN Benchmark Suite

Benchmark sistem za upoređivanje performansi tri K-NN implementacije:
1. **KNNBasic** - Brute-force pristup (baseline)
2. **KNNKDTree** - Optimizacija koristeći k-d tree
3. **KNNNanoflann** - Externa biblioteka koja koristi napredne optimizacije: randomizaciju u konstrukciji stabla, multiple k-d trees i priority queues za pretragu, što omogućava značajno efikasnije performanse od standardnog k-d tree pristupa.

## Priprema i Kompilacija

### 1. Instalacija nanoflann biblioteke

Koristite pripremljene skripte:
- **Linux/Mac:** `./setup_nanoflann.sh`
- **Windows:** `.\setup_nanoflann.ps1`

### 2. Instalacija Python biblioteka (za vizualizaciju)

```bash
pip install -r requirements.txt
```

### 3. Kompilacija

**Linux/Mac:**
```bash
mkdir build
cd build
cmake ..
make
```

**Windows (MinGW):**
```powershell
mkdir build
cd build
cmake -G "MinGW Makefiles" ..
mingw32-make
```

## Pokretanje

Iz `build` foldera:
```bash
./knn_benchmark  # ili .\knn_benchmark.exe na Windows-u
```

Rezultati se čuvaju u `build/benchmarks/results/`:
- `benchmark_results.json` - JSON format
- `benchmark_comprehensive.csv` - CSV format

## Vizualizacija Rezultata

Iz **root** foldera projekta (`Implementacija`):
```bash
python benchmarks/visualize_results.py
```

**NAPOMENA:** Nemojte pokretati iz `build` foldera jer skripta očekuje relativne putanje.

### Generisani Grafici

Rezultati u `build/benchmarks/results/plots/`:

1. **curse_of_dimensionality.png** - Prikaz pada performansi sa porastom dimenzija (2D do 64D)
2. **scalability.png** - Skalabilnost algoritama sa brojem uzoraka (100 do 20000)
3. **k_parameter_impact.png** - Uticaj K parametra na vreme upita (k=1 do k=100)
4. **distance_calculations_real_datasets.png** - Prosječan broj kalkulacija distanci po query-ju za svaki algoritam na realnim datasetima

### Podaci u Vizualizaciji

- **Query time** - Prosječno vrijeme izvršavanja upita (ms)
- **Speedup** - Ubrzanje u odnosu na KNNBasic
- **Distance calculations** - Broj kalkulacija distanci (manje = efikasnije)
- **Accuracy, Precision, Recall, F1** - Metrike klasifikacije za realne datasete

## Mjerene Metrike

Za svaki test:
- **Query time** - Prosječno vrijeme po upitu (ms)
- **Build time** - Vrijeme konstrukcije strukture podataka (ms)
- **Speedup** - Ubrzanje u odnosu na KNNBasic
- **Distance calculations** - Broj kalkulacija distanci (tačan broj za KNNBasic/KNNKDTree, aproksimacija za KNNNanoflann)
- **Accuracy, Precision, Recall, F1** - Metrike klasifikacije (samo za realne datasete)

## Napomene

- Realni dataseti se ograničavaju na **10,000 uzoraka** za bržu analizu
- Benchmark koristi **fixed seed (42)** za reproducibilnost
- Warmup run se izvršava prije mjerenja
- LaTeX tabele se generišu automatski u `build/benchmarks/results/benchmark_table.tex`

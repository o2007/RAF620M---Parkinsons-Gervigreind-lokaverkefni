# Parkinsons_gervigreind_spa

Lokaverkefni í RAF620M – Inngangur að vélrænu námi og gervigreind.
Unnið af Agötu Mozwilo og Ólaf Þór Fortune

## Uppbygging verkefnis
```
Parkinsons_gervigreind_spa/
│
├── data/           # Gagnasöfn, ekki vistuð í Git
├── ppmi_outputs/   # Úttak úr keyrslum, ekki vistuð í Git
├── src/            # Python kóði
└── requirements.txt
```

## Helstu skrár
- `src/init.py`: Skilgreinir öll mikilvægustu föll
- `src/linear_regression.py`: línulegt grunnlíkan.
- `src/random_forest_model.py`: Random Forest líkan.
- `src/svm_model.py`: SVR líkan.
- `src/xgboost_model.py`: XGBoost líkan.
- `src/comparison.py`: samanburður á öllum líkönum.
- `src/plot.py`: myndir fyrir skýrslu og yfirlit.

## Keyrsla - Mikilvægt að keyra í réttri röð
```bash
python src/linear_regression.py
python src/random_forest_model.py
python src/svm_model.py
python src/xgboost_model.py
python src/comparison.py
python src/plot.py
```

## Höfundar
- Agata Mozwilo
- Ólafur Þór Fortune

## Námskeið
RAF620M – Inngangur að vélrænu námi og gervigreind, 2026

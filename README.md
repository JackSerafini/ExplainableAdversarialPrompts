# Explainable Artificial Intelligence - Project

# Installation

Create a virtual environment and activate it:
```terminal
python3.13 -m venv venv
source venv/bin/activate
```

Then, clone the original TokenSHAP repository and install the dependencies:
```terminal
git clone https://github.com/ronigold/TokenSHAP.git
cd TokenSHAP
pip install -r requirements.txt
```

Allora, dobbiamo
- [ ] Generare tot frasi da algoritmo di Lorenzo
- [ ] Per ogni frase, applicare TokenSHAP e analizzare quali sono i token "importanti"
    - [ ] Creare file unico per gestire SHAP
    - [ ] Dividere le frasi in token, non più parole/frasi
    - [x] Ottenere risposte con meno token rispetto a quelle complete che SHAP genera
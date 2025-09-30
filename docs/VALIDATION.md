# Couverture des modules spécialisés

Le tableau ci-dessous récapitule les tests (unitaires ou d'intégration)
assurant la couverture des modules avancés utilisés par la matrice de
validation. Chaque ligne liste le scénario de la matrice correspondant ainsi que
les tests automatisés qui vérifient le module.

| Module | Scénario matrice | Tests automatisés | Statut |
| --- | --- | --- | --- |
| Duty-cycle | `duty_cycle_enforcement_class_a` | `test_poisson_independence.py`, matrice de validation | ✅
| Multicanal dynamique | `dynamic_multichannel_random_assignment` | `test_multichannel_selection.py`, `test_mobility_multichannel_integration.py`, matrice de validation | ✅
| Classe B mobile | `class_b_mobility_multichannel` | `test_class_bc.py`, matrice de validation | ✅
| Classe C mobile | `class_c_mobility_multichannel` | `test_mobility_multichannel_integration.py`, matrice de validation | ✅
| EXPLoRa-AT | `explora_at_balanced_airtime` | `loraflexsim/launcher/tests/test_explora_at.py`, matrice de validation | ✅
| ADR-ML | `adr_ml_adaptive_strategy` | `loraflexsim/launcher/tests/test_adr_ml.py`, matrice de validation | ✅

La présence des scénarios est contrôlée automatiquement par
`test_validation_matrix_covers_specialised_modules`, qui échoue si l'un des
modules ci-dessus n'est plus représenté.【F:tests/integration/test_validation_matrix.py†L80-L113】

Références détaillées des scénarios :
- `loraflexsim/validation/__init__.py` pour la configuration exacte de chaque
  cas.【F:loraflexsim/validation/__init__.py†L1-L209】
- Les tests listés ci-dessus documentent la logique de validation spécifique à
  chaque module.【F:tests/test_poisson_independence.py†L1-L55】【F:tests/test_multichannel_selection.py†L1-L20】【F:tests/test_mobility_multichannel_integration.py†L1-L18】【F:tests/test_class_bc.py†L1-L33】【F:loraflexsim/launcher/tests/test_explora_at.py†L1-L88】【F:loraflexsim/launcher/tests/test_adr_ml.py†L1-L27】

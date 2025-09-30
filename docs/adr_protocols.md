# Protocoles ADR LoRa

Cette page résume plusieurs stratégies d'Adaptive Data Rate (ADR) pour LoRaWAN.
Le protocole `ADR_ML` propose une approche fondée sur le machine learning.

| Protocole | Principe général | Machine learning | Avantages | Limites |
|-----------|-----------------|-----------------|-----------|---------|
| ADR standard | Algorithme LoRaWAN basé sur l'historique SNR pour ajuster SF et puissance. | Non | Simple, conforme à la spécification, faible surcharge réseau. | Réagit lentement, peu adapté aux nœuds mobiles ou aux environnements très variables. |
| EXPLoRa-SF | Répartition équitable des spreading factors pour maximiser le débit global. | Non | Améliore l'équité entre nœuds, réduit les collisions. | Nécessite une coordination centrale, peut augmenter la latence pour certains nœuds. |
| EXPLoRa-AT | Extension d'EXPLoRa optimisant SF et temps d'accès pour équilibrer l'occupation du canal. | Non | Meilleure utilisation du temps d'antenne, réduit la congestion. | Complexité accrue, besoin de synchronisation précise. |
| ADR-Lite | Version simplifiée avec seuils SNR fixes pour un ajustement rapide. | Non | Faible calcul côté serveur, adaptation rapide. | Moins précise, risque d'augmentation de la consommation ou des collisions. |
| ADR-Max | Algorithme ADR standard exploitant le SNR maximal des 20 derniers uplinks pour réduire SF et puissance. | Non | Compatible LoRaWAN, optimise le débit. | Requiert un historique suffisant, réagit lentement aux variations rapides. |
| RADR | ADR réactif ajustant SF et puissance à chaque message selon la dernière qualité du lien. | Oui | Très réactif aux variations, adapté aux scénarios de mobilité. | Mesures bruyantes entraînant des oscillations, nécessite plus de signalisations. |
| ADR_ML | Modèle fondé sur le machine learning pour prédire SF et puissance à partir des caractéristiques du lien. | Oui | Peut s'adapter à des scénarios complexes, performances potentielles supérieures. | Besoin de données d'entraînement, coût computationnel, explicabilité limitée. |

L'algorithme **ADR-Lite** n'emploie aucun procédé de machine learning : il se contente de comparer le SNR mesuré à des seuils fixes pour ajuster rapidement le spreading factor.

## ADR-Max

Cette stratégie suit l'algorithme ADR standard décrit dans la spécification LoRaWAN.
Le serveur calcule le SNR maximal observé parmi les 20 derniers uplinks d'un nœud.
Lorsque ce SNR offre une marge suffisante, l'algorithme réduit d'abord le
spreading factor puis la puissance d'émission afin d'augmenter le débit tout en
restant compatible avec LoRaWAN.

Dans LoRaFlexSim, la mesure utilisée correspond désormais au **SNIR remonté par
la passerelle** ayant décodé l'uplink. Le serveur conserve un historique par
passerelle, ce qui reproduit fidèlement le comportement de FLoRa dans les
déploiements multi-gateways.

Une trace FLoRa multi-passerelles de 25 uplinks (fichier
``flora_multi_gateway_txconfig.json``) a été rejouée dans le test
``tests/integration/test_adr_standard_alignment.py``. Chaque décision TXCONFIG
(SF, puissance et passerelle utilisée) est comparée à la référence et les
fenêtres RX planifiées doivent coïncider à 1 µs près. Ce test valide que les
commandes ADR générées par ``adr_standard_1`` correspondent aux décisions
observées dans FLoRa pour ce scénario.

**Avantages**

- Optimise le débit et la consommation énergétique.
- Conserve une compatibilité totale avec la spécification LoRaWAN.

**Risques**

- Nécessite au moins 20 uplinks pour une estimation fiable.
- Réagit lentement aux fluctuations rapides du canal radio.

## EXPLoRa-AT

Cette variante d'EXPLoRa initialise tous les nœuds avec une marge
d'installation de 10 dB et le spreading factor SF12. Chaque nœud émet
ainsi à puissance maximale et l'algorithme ajuste ensuite SF et puissance
en fonction du SNR mesuré lors des premiers uplinks pour équilibrer
l'occupation du canal.

Le serveur trie d'abord les nœuds par RSSI décroissant puis répartit les
spreading factors de manière à ce que le temps d'antenne cumulé soit
identique pour chaque groupe. La durée d'un paquet pour un SF donné est
calculée par la fonction :func:`Channel.airtime(sf, L)`\u2009; elle implémente
la formule LoRa classique\u00a0:

\[
t_{\text{air}} = t_{\text{preamble}} + t_{\text{payload}},\qquad
t_{\text{preamble}} = (N_{\text{preamble}}+4.25)T_s,
\]
\[
T_s = \frac{2^{\text{SF}}}{\text{BW}},\qquad
n_{\text{payload}} = \max\left(\left\lceil\frac{8L-4\text{SF}+28+16}{4(\text{SF}-2d_e)}\right\rceil,0\right)(\text{CR}+4)+8,
\]
\[
t_{\text{payload}} = n_{\text{payload}}T_s
\]
avec \(L\) la taille de la charge utile (20\u00a0octets par défaut) et
\(d_e\) l'activation du "low data rate".

Les tailles de groupes sont alors dérivées pour que
\(N_{\text{SF}} t_{\text{air}}(\text{SF})\) soit constant\u00a0:

\[
N_{\text{SF}} = \frac{N}{t_{\text{air}}(\text{SF})\sum_{s=7}^{12} 1/t_{\text{air}}(s)}.
\]

Chaque nœud démarre en SF12 avec une puissance de 14\u00a0dBm. Après les
premières mesures de SNR, le serveur vérifie la marge
\(\text{margin} = \text{RSSI} - \text{bruit} - \text{SNR}_{\text{req}}(\text{SF}) - MARGIN_{\text{dB}}\).
La puissance est ajustée par pas de 3\u00a0dB pour maintenir une marge
positive; si celle\u2011ci reste négative, le SF est augmenté. Une marge
supérieure à 3\u00a0dB entraîne une réduction de puissance. Un
``LinkADRReq`` est envoyé à chaque changement.

### Exemple de configuration

```python
from loraflexsim.launcher import Simulator, explora_at

sim = Simulator(nodes=50, packets=100, interval=60.0)
explora_at.apply(sim)  # active EXPLoRa‑AT
sim.run()
```

Les nœuds les plus proches reçoivent ainsi un SF plus faible et une
puissance réduite, tandis que les nœuds éloignés conservent un SF plus
élevé et la pleine puissance, aboutissant à une occupation du canal
uniforme.

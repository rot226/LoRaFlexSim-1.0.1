import argparse
import csv
import random
import logging

# Configuration du logger pour afficher les informations
logging.basicConfig(level=logging.INFO, format="%(message)s")


def simulate(nodes, gateways, area, mode, interval, steps, channels=1):
    """Exécute une simulation LoRa simplifiée et retourne les métriques.

    Les transmissions peuvent se faire sur plusieurs canaux et plusieurs
    passerelles. Les nœuds sont répartis de façon uniforme sur les ``channels``
    et sur les ``gateways`` disponibles et les collisions ne surviennent
    qu'entre nœuds partageant à la fois le même canal **et** la même passerelle.
    """
    if nodes < 1:
        raise ValueError("nodes must be >= 1")
    if gateways < 1:
        raise ValueError("gateways must be >= 1")
    if channels < 1:
        raise ValueError("channels must be >= 1")

    # Initialisation des compteurs
    total_transmissions = 0
    collisions = 0
    delivered = 0
    energy_consumed = 0.0
    delays = []  # stockera le délai de chaque paquet livré

    # Génération des instants d'émission pour chaque nœud et attribution d'un canal
    send_times = {node: [] for node in range(nodes)}
    node_channels = {node: node % channels for node in range(nodes)}
    node_gateways = {node: node % max(1, gateways) for node in range(nodes)}
    for node in range(nodes):
        if mode.lower() == "periodic":
            t = 0
            while t < steps:
                send_times[node].append(t)
                t += interval
        else:  # mode "Random"
            # Émission aléatoire avec probabilité 1/interval à chaque pas de temps
            for t in range(steps):
                if random.random() < 1.0 / interval:
                    send_times[node].append(t)

    # Simulation pas à pas
    for t in range(steps):
        transmitting_nodes = [
            node for node, times in send_times.items() if t in times
        ]
        # Gérer les transmissions par passerelle puis par canal
        for gw in range(max(1, gateways)):
            gw_nodes = [
                n for n in transmitting_nodes if node_gateways[n] == gw
            ]
            for ch in range(channels):
                nodes_on_ch = [n for n in gw_nodes if node_channels[n] == ch]
                nb_tx = len(nodes_on_ch)
                if nb_tx > 0:
                    total_transmissions += nb_tx
                    if nb_tx == 1:
                        delivered += 1
                        energy_consumed += 1.0
                        delays.append(0)
                    else:
                        collisions += nb_tx
                        energy_consumed += nb_tx * 1.0

    # Calcul des métriques finales
    pdr = (
        (delivered / total_transmissions) * 100
        if total_transmissions > 0
        else 0
    )
    avg_delay = (sum(delays) / len(delays)) if delays else 0

    return delivered, collisions, pdr, energy_consumed, avg_delay


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulateur LoRa – Mode CLI")
    parser.add_argument(
        "--nodes", type=int, default=10, help="Nombre de nœuds"
    )
    parser.add_argument(
        "--gateways", type=int, default=1, help="Nombre de gateways"
    )
    parser.add_argument(
        "--area",
        type=int,
        default=1000,
        help="Taille de l'aire de simulation (côté du carré)",
    )
    parser.add_argument(
        "--channels", type=int, default=1, help="Nombre de canaux radio"
    )
    parser.add_argument(
        "--mode",
        choices=["Random", "Periodic"],
        default="Random",
        help="Mode de transmission",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Intervalle moyen ou fixe entre transmissions",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Nombre de pas de temps de la simulation",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Fichier CSV pour sauvegarder les résultats (optionnel)",
    )
    parser.add_argument(
        "--lorawan-demo",
        action="store_true",
        help="Exécute un exemple LoRaWAN",
    )
    args = parser.parse_args()

    logging.info(
        f"Simulation d'un réseau LoRa : {args.nodes} nœuds, {args.gateways} gateways, "
        f"aire={args.area}m, {args.channels} canaux, mode={args.mode}, "
        f"intervalle={args.interval}, steps={args.steps}"
    )
    if args.lorawan_demo:
        from launcher.node import Node
        from launcher.gateway import Gateway
        from launcher.server import NetworkServer

        gw = Gateway(0, 0, 0)
        ns = NetworkServer()
        ns.gateways = [gw]
        node = Node(0, 0, 0, 7, 20)
        frame = node.prepare_uplink(b"ping", confirmed=True)
        ns.send_downlink(node, b"ack")
        rx1, _ = node.schedule_receive_windows(0)
        gw.pop_downlink(node.id)  # illustration
        logging.info(
            f"Exemple LoRaWAN : trame uplink FCnt={frame.fcnt}, RX1={rx1}s"
        )
        exit()

    delivered, collisions, pdr, energy, avg_delay = simulate(
        args.nodes,
        args.gateways,
        args.area,
        args.mode,
        args.interval,
        args.steps,
        args.channels,
    )
    logging.info(
        f"Résultats : PDR={pdr:.2f}% , Paquets livrés={delivered}, Collisions={collisions}, "
        f"Énergie consommée={energy:.1f} unités, Délai moyen={avg_delay:.2f} unités de temps"
    )

    # Sauvegarde des résultats dans un CSV si demandé
    if args.output:
        with open(args.output, mode="w", newline="") as f:
            writer = csv.writer(f)
            # En-tête
            writer.writerow(
                [
                    "nodes",
                    "gateways",
                    "area",
                    "channels",
                    "mode",
                    "interval",
                    "steps",
                    "delivered",
                    "collisions",
                    "PDR(%)",
                    "energy",
                    "avg_delay",
                ]
            )
            # Données
            writer.writerow(
                [
                    args.nodes,
                    args.gateways,
                    args.area,
                    args.channels,
                    args.mode,
                    args.interval,
                    args.steps,
                    delivered,
                    collisions,
                    f"{pdr:.2f}",
                    f"{energy:.1f}",
                    f"{avg_delay:.2f}",
                ]
            )
        logging.info(f"Résultats enregistrés dans {args.output}")

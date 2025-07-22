import argparse
import csv
import random
import logging
import sys

PAYLOAD_SIZE = 20  # octets simulés par paquet

# Configuration du logger pour afficher les informations
logging.basicConfig(level=logging.INFO, format="%(message)s")


def simulate(nodes, gateways, mode, interval, steps, channels=1,
             *, fine_fading_std=0.0, noise_std=0.0):
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
    if interval <= 0:
        raise ValueError("interval must be > 0")
    if steps <= 0:
        raise ValueError("steps must be > 0")

    mode_lower = mode.lower()
    if mode_lower not in {"random", "periodic"}:
        raise ValueError("mode must be 'Random' or 'Periodic'")

    # Initialisation des compteurs
    total_transmissions = 0
    collisions = 0
    delivered = 0
    energy_consumed = 0.0
    # Liste des délais de livraison (0 pour chaque paquet car la transmission
    # réussie est immédiate dans ce modèle simplifié)
    delays = []

    # Génération des instants d'émission pour chaque nœud et attribution d'un canal
    send_times = {node: [] for node in range(nodes)}
    node_channels = {node: node % channels for node in range(nodes)}
    node_gateways = {node: node % max(1, gateways) for node in range(nodes)}
    for node in range(nodes):
        if mode_lower == "periodic":
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
    pending = {}

    for t in range(steps):
        # Ajouter les nouvelles transmissions prêtes à l'instant t
        for node, times in send_times.items():
            if t in times and node not in pending:
                pending[node] = t

        # Traiter les transmissions en attente par passerelle puis par canal
        for gw in range(max(1, gateways)):
            gw_nodes = [n for n in pending.keys() if node_gateways[n] == gw]
            for ch in range(channels):
                nodes_on_ch = [n for n in gw_nodes if node_channels[n] == ch]
                nb_tx = len(nodes_on_ch)
                if nb_tx > 0:
                    total_transmissions += nb_tx
                    energy_consumed += nb_tx * 1.0
                    if nb_tx == 1:
                        n = nodes_on_ch[0]
                        success = True
                        if fine_fading_std > 0.0 and random.gauss(0.0, fine_fading_std) < -3.0:
                            success = False
                        if noise_std > 0.0 and random.gauss(0.0, noise_std) > 3.0:
                            success = False
                        if success:
                            delivered += 1
                            delays.append(t - pending[n])
                            del pending[n]
                        else:
                            collisions += 1
                    else:
                        collisions += nb_tx - 1
                        winner = random.choice(nodes_on_ch)
                        for n in nodes_on_ch:
                            if n == winner:
                                delivered += 1
                                delays.append(t - pending[n])
                                del pending[n]
                            else:
                                # reste en attente pour le prochain tour
                                pass

    # Calcul des métriques finales
    pdr = (
        (delivered / total_transmissions) * 100
        if total_transmissions > 0
        else 0
    )
    avg_delay = (sum(delays) / len(delays)) if delays else 0
    throughput_bps = delivered * PAYLOAD_SIZE * 8 / steps if steps > 0 else 0.0

    return (
        delivered,
        collisions,
        pdr,
        energy_consumed,
        avg_delay,
        throughput_bps,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Simulateur LoRa – Mode CLI")
    parser.add_argument(
        "--nodes", type=int, default=10, help="Nombre de nœuds"
    )
    parser.add_argument(
        "--gateways", type=int, default=1, help="Nombre de gateways"
    )
    parser.add_argument(
        "--channels", type=int, default=1, help="Nombre de canaux radio"
    )
    parser.add_argument(
        "--mode",
        choices=["random", "periodic"],
        default="random",
        type=str.lower,
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
        "--runs",
        type=int,
        default=1,
        help="Nombre d'exécutions à réaliser",
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
    parser.add_argument(
        "--seed",
        type=int,
        help="Graine aléatoire pour reproduire les résultats",
    )
    parser.add_argument(
        "--fine-fading",
        type=float,
        default=0.0,
        help="Écart-type du fading fin (dB)",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="Écart-type du bruit thermique variable (dB)",
    )
    args = parser.parse_args(argv)

    if args.runs < 1:
        parser.error("--runs must be >= 1")

    logging.info(
        f"Simulation d'un réseau LoRa : {args.nodes} nœuds, {args.gateways} gateways, "
        f"{args.channels} canaux, mode={args.mode}, "
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
        sys.exit()

    results = []
    for i in range(args.runs):
        if args.seed is not None:
            random.seed(args.seed + i)

        delivered, collisions, pdr, energy, avg_delay, throughput = simulate(
            args.nodes,
            args.gateways,
            args.mode,
            args.interval,
            args.steps,
            args.channels,
            fine_fading_std=args.fine_fading,
            noise_std=args.noise_std,
        )
        results.append(
            (delivered, collisions, pdr, energy, avg_delay, throughput)
        )
        logging.info(
            f"Run {i + 1}/{args.runs} : PDR={pdr:.2f}% , Paquets livrés={delivered}, Collisions={collisions}, "
            f"Énergie consommée={energy:.1f} unités, Délai moyen={avg_delay:.2f} unités de temps, "
            f"Débit moyen={throughput:.2f} bps"
        )

    averages = [
        sum(r[i] for r in results) / len(results) for i in range(len(results[0]))
    ]
    logging.info(
        f"Moyenne : PDR={averages[2]:.2f}% , Paquets livrés={averages[0]:.2f}, Collisions={averages[1]:.2f}, "
        f"Énergie consommée={averages[3]:.1f} unités, Délai moyen={averages[4]:.2f} unités de temps, "
        f"Débit moyen={averages[5]:.2f} bps"
    )

    if args.output:
        with open(args.output, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "nodes",
                    "gateways",
                    "channels",
                    "mode",
                    "interval",
                    "steps",
                    "run",
                    "delivered",
                    "collisions",
                    "PDR(%)",
                    "energy",
                    "avg_delay",
                    "throughput_bps",
                ]
            )
            for run_idx, (d, c, p, e, ad, th) in enumerate(results, start=1):
                writer.writerow(
                    [
                        args.nodes,
                        args.gateways,
                        args.channels,
                        args.mode,
                        args.interval,
                        args.steps,
                        run_idx,
                        d,
                        c,
                        f"{p:.2f}",
                        f"{e:.1f}",
                        f"{ad:.2f}",
                        f"{th:.2f}",
                    ]
                )
        logging.info(f"Résultats enregistrés dans {args.output}")

    return results, tuple(averages)


if __name__ == "__main__":
    main()

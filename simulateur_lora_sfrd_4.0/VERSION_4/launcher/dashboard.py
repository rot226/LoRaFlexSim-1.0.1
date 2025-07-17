import os
import sys

import panel as pn
import plotly.graph_objects as go
import time
import threading
import pandas as pd

# Assurer la résolution correcte des imports quel que soit le répertoire
# depuis lequel ce fichier est exécuté. On ajoute le dossier parent
# (celui contenant le paquet ``launcher``) au ``sys.path`` s'il n'y est pas
# déjà. Ainsi, ``from launcher.simulator`` fonctionnera aussi avec la
# commande ``panel serve dashboard.py`` exécutée depuis ce dossier.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from launcher.simulator import Simulator  # noqa: E402
from launcher import adr_standard_1, adr_2, adr_3  # noqa: E402

# --- Initialisation Panel ---
pn.extension("plotly", raw_css=[
    ".coord-textarea textarea {font-size: 14pt;}",
])
# Définition du titre de la page via le document Bokeh directement
pn.state.curdoc.title = "Simulateur LoRa"

# --- Variables globales ---
sim = None
sim_callback = None
chrono_callback = None
start_time = None
elapsed_time = 0
max_real_time = None
paused = False
selected_adr_module = adr_standard_1
total_runs = 1
current_run = 0
runs_events: list[pd.DataFrame] = []
runs_metrics: list[dict] = []
auto_fast_forward = False

# --- Widgets de configuration ---
num_nodes_input = pn.widgets.IntInput(name="Nombre de nœuds", value=20, step=1, start=1)
num_gateways_input = pn.widgets.IntInput(name="Nombre de passerelles", value=1, step=1, start=1)
area_input = pn.widgets.FloatInput(name="Taille de l'aire (m)", value=1000.0, step=100.0, start=100.0)
mode_select = pn.widgets.RadioButtonGroup(
    name="Mode d'émission", options=["Aléatoire", "Périodique"], value="Aléatoire"
)
interval_input = pn.widgets.FloatInput(name="Intervalle moyen (s)", value=30.0, step=1.0, start=0.1)
packets_input = pn.widgets.IntInput(name="Nombre de paquets (0=infin)", value=0, step=1, start=0)
seed_input = pn.widgets.IntInput(
    name="Graine (0 = aléatoire)", value=0, step=1, start=0
)
num_runs_input = pn.widgets.IntInput(name="Nombre de runs", value=1, start=1)
adr_node_checkbox = pn.widgets.Checkbox(name="ADR nœud", value=False)
adr_server_checkbox = pn.widgets.Checkbox(name="ADR serveur", value=False)

# --- Boutons de sélection du profil ADR ---
adr1_button = pn.widgets.Button(name="adr_1", button_type="primary")
adr2_button = pn.widgets.Button(name="adr_2")
adr3_button = pn.widgets.Button(name="adr_3")
adr_active_badge = pn.pane.HTML("", width=80)

# --- Choix SF et puissance initiaux identiques ---
fixed_sf_checkbox = pn.widgets.Checkbox(name="Choisir SF unique", value=False)
sf_value_input = pn.widgets.IntSlider(name="SF initial", start=7, end=12, value=7, step=1, disabled=True)

fixed_power_checkbox = pn.widgets.Checkbox(name="Choisir puissance unique", value=False)
tx_power_input = pn.widgets.FloatSlider(name="Puissance Tx (dBm)", start=2, end=20, value=14, step=1, disabled=True)

# --- Multi-canaux ---
num_channels_input = pn.widgets.IntInput(name="Nb sous-canaux", value=1, step=1, start=1)
channel_dist_select = pn.widgets.RadioButtonGroup(
    name="Répartition canaux", options=["Round-robin", "Aléatoire"], value="Round-robin"
)

# --- Widget pour activer/désactiver la mobilité des nœuds ---
mobility_checkbox = pn.widgets.Checkbox(name="Activer la mobilité des nœuds", value=False)

# Widgets pour régler la vitesse minimale et maximale des nœuds mobiles
mobility_speed_min_input = pn.widgets.FloatInput(name="Vitesse min (m/s)", value=2.0, step=0.5, start=0.1)
mobility_speed_max_input = pn.widgets.FloatInput(name="Vitesse max (m/s)", value=10.0, step=0.5, start=0.1)

# --- Durée réelle de simulation et bouton d'accélération ---
real_time_duration_input = pn.widgets.FloatInput(name="Durée réelle max (s)", value=0.0, step=1.0, start=0.0)
fast_forward_button = pn.widgets.Button(name="Accélérer jusqu'à la fin", button_type="primary", disabled=True)

# --- Paramètres radio FLoRa ---
flora_mode_toggle = pn.widgets.Toggle(name="Mode FLoRa complet", button_type="default", value=False)
detection_threshold_input = pn.widgets.FloatInput(
    name="Seuil détection (dBm)", value=-110.0, step=1.0, start=-150.0
)
min_interference_input = pn.widgets.FloatInput(
    name="Min interference (s)", value=0.0, step=0.1, start=0.0
)
# --- Paramètres supplémentaires ---
battery_capacity_input = pn.widgets.FloatInput(
    name="Capacité batterie (J)", value=0.0, step=10.0, start=0.0
)
payload_size_input = pn.widgets.IntInput(
    name="Taille payload (o)", value=20, step=1, start=1
)
node_class_select = pn.widgets.RadioButtonGroup(
    name="Classe LoRaWAN", options=["A", "B", "C"], value="A"
)
# Lorsque le mode FLoRa est activé, cette valeur est fixée à 5 s

# --- Positions manuelles ---
manual_pos_toggle = pn.widgets.Checkbox(name="Positions manuelles")
position_textarea = pn.widgets.TextAreaInput(
    name="Coordonnées",
    height=100,
    visible=False,
    width=400,
    css_classes=["coord-textarea"],
)

# --- Boutons de contrôle ---
start_button = pn.widgets.Button(name="Lancer la simulation", button_type="success")
stop_button = pn.widgets.Button(name="Arrêter la simulation", button_type="warning", disabled=True)
pause_button = pn.widgets.Button(name="Pause", button_type="primary", disabled=True)

# --- Nouveau bouton d'export et message d'état ---
export_button = pn.widgets.Button(name="Exporter résultats (dossier courant)", button_type="primary", disabled=True)
export_message = pn.pane.HTML("Clique sur Exporter pour générer le fichier CSV après la simulation.")

# --- Indicateurs de métriques ---
pdr_indicator = pn.indicators.Number(name="PDR", value=0, format="{value:.1%}")
collisions_indicator = pn.indicators.Number(name="Collisions", value=0, format="{value:d}")
energy_indicator = pn.indicators.Number(name="Énergie Tx (J)", value=0.0, format="{value:.3f}")
delay_indicator = pn.indicators.Number(name="Délai moyen (s)", value=0.0, format="{value:.3f}")
throughput_indicator = pn.indicators.Number(name="Débit (bps)", value=0.0, format="{value:.2f}")

# Tableau récapitulatif du PDR par nœud (global et récent)
pdr_table = pn.pane.DataFrame(
    pd.DataFrame(columns=["Node", "PDR", "Recent PDR"]),
    height=200,
    width=220,
)

# --- Chronomètre ---
chrono_indicator = pn.indicators.Number(name="Durée simulation (s)", value=0, format="{value:.1f}")


# --- Pane pour la carte des nœuds/passerelles ---
# Agrandir la surface d'affichage de la carte pour une meilleure lisibilité
map_pane = pn.pane.Plotly(height=600, sizing_mode="stretch_width")

# --- Pane pour l'histogramme SF ---
sf_hist_pane = pn.pane.Plotly(height=250, sizing_mode="stretch_width")


# --- Mise à jour de la carte ---
def update_map():
    global sim
    if sim is None:
        return
    fig = go.Figure()
    x_nodes = [node.x for node in sim.nodes]
    y_nodes = [node.y for node in sim.nodes]
    node_ids = [str(node.id) for node in sim.nodes]
    fig.add_scatter(
        x=x_nodes,
        y=y_nodes,
        mode="markers+text",
        name="Nœuds",
        text=node_ids,
        textposition="middle center",
        marker=dict(symbol="circle", color="blue", size=32),
        textfont=dict(color="white", size=14),
    )
    x_gw = [gw.x for gw in sim.gateways]
    y_gw = [gw.y for gw in sim.gateways]
    gw_ids = [str(gw.id) for gw in sim.gateways]
    fig.add_scatter(
        x=x_gw,
        y=y_gw,
        mode="markers+text",
        name="Passerelles",
        text=gw_ids,
        textposition="middle center",
        marker=dict(symbol="star", color="red", size=28, line=dict(width=1, color="black")),
        textfont=dict(color="white", size=14),
    )
    area = area_input.value
    # Add a small extra space on the Y axis so edge nodes remain fully visible
    extra_y = area * 0.125
    display_area_y = area + extra_y
    fig.update_layout(
        title="Position des nœuds et passerelles",
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        xaxis_range=[0, area],
        yaxis_range=[-extra_y, display_area_y],
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    map_pane.object = fig


# --- Callback pour changer le label de l'intervalle selon le mode d'émission ---
def on_mode_change(event):
    if event.new == "Aléatoire":
        interval_input.name = "Intervalle moyen (s)"
    else:
        interval_input.name = "Période (s)"


mode_select.param.watch(on_mode_change, "value")


# --- Sélection du profil ADR ---
def _update_adr_badge(name: str) -> None:
    adr_active_badge.object = (
        f"<span style='background-color: #28a745; color:white; padding:2px 6px; border-radius:4px'>{name}</span>"
    )


def select_adr(module, name: str) -> None:
    global selected_adr_module
    selected_adr_module = module
    adr_node_checkbox.value = True
    adr_server_checkbox.value = True
    _update_adr_badge(name)
    for btn in (adr1_button, adr2_button, adr3_button):
        btn.button_type = "default"
    if name == "ADR 1":
        adr1_button.button_type = "primary"
    elif name == "ADR 2":
        adr2_button.button_type = "primary"
    else:
        adr3_button.button_type = "primary"
    if sim is not None:
        module.apply(sim)


_update_adr_badge("ADR 1")

# --- Callback chrono ---
def periodic_chrono_update():
    global chrono_indicator, start_time, elapsed_time, max_real_time
    if start_time is not None:
        elapsed_time = time.time() - start_time
        chrono_indicator.value = elapsed_time
        if max_real_time is not None and elapsed_time >= max_real_time:
            on_stop(None)


# --- Callback étape de simulation ---
def step_simulation():
    if sim is None:
        return
    cont = sim.step()
    metrics = sim.get_metrics()
    pdr_indicator.value = metrics["PDR"]
    collisions_indicator.value = metrics["collisions"]
    energy_indicator.value = metrics["energy_J"]
    delay_indicator.value = metrics["avg_delay_s"]
    throughput_indicator.value = metrics["throughput_bps"]
    table_df = pd.DataFrame(
        {
            "Node": list(metrics["pdr_by_node"].keys()),
            "PDR": list(metrics["pdr_by_node"].values()),
            "Recent PDR": [
                metrics["recent_pdr_by_node"][nid]
                for nid in metrics["pdr_by_node"].keys()
            ],
        }
    )
    pdr_table.object = table_df
    sf_dist = metrics["sf_distribution"]
    sf_fig = go.Figure(
        data=[go.Bar(x=[f"SF{sf}" for sf in sf_dist.keys()], y=list(sf_dist.values()))]
    )
    sf_fig.update_layout(
        title="Répartition des SF par nœud",
        xaxis_title="SF",
        yaxis_title="Nombre de nœuds",
        yaxis_range=[0, sim.num_nodes],
    )
    sf_hist_pane.object = sf_fig
    update_map()
    if not cont:
        on_stop(None)
        return


# --- Bouton "Lancer la simulation" ---
def setup_simulation(seed_offset: int = 0):
    """Crée et démarre un simulateur avec les paramètres du tableau de bord."""
    global sim, sim_callback, start_time, chrono_callback, elapsed_time, max_real_time, paused
    elapsed_time = 0

    if sim_callback:
        sim_callback.stop()
        sim_callback = None
    if chrono_callback:
        chrono_callback.stop()
        chrono_callback = None

    seed_val = int(seed_input.value)
    seed = seed_val + seed_offset if seed_val != 0 else None

    sim = Simulator(
        num_nodes=int(num_nodes_input.value),
        num_gateways=int(num_gateways_input.value),
        area_size=float(area_input.value),
        transmission_mode="Random" if mode_select.value == "Aléatoire" else "Periodic",
        packet_interval=float(interval_input.value),
        packets_to_send=int(packets_input.value),
        adr_node=adr_node_checkbox.value,
        adr_server=adr_server_checkbox.value,
        mobility=mobility_checkbox.value,
        mobility_speed=(float(mobility_speed_min_input.value), float(mobility_speed_max_input.value)),
        channels=[868e6 + i * 200e3 for i in range(num_channels_input.value)],
        channel_distribution="random" if channel_dist_select.value == "Aléatoire" else "round-robin",
        fixed_sf=int(sf_value_input.value) if fixed_sf_checkbox.value else None,
        fixed_tx_power=float(tx_power_input.value) if fixed_power_checkbox.value else None,
        battery_capacity_j=float(battery_capacity_input.value) if battery_capacity_input.value > 0 else None,
        payload_size_bytes=int(payload_size_input.value),
        node_class=node_class_select.value,
        detection_threshold_dBm=float(detection_threshold_input.value),
        min_interference_time=float(min_interference_input.value),
        seed=seed,
    )

    if manual_pos_toggle.value:
        for line in position_textarea.value.splitlines():
            parts = [p.strip() for p in line.split(',') if p.strip()]
            if not parts:
                continue
            kind = parts[0]
            kv = {}
            for p in parts[1:]:
                if '=' in p:
                    k, v = p.split('=', 1)
                    kv[k.strip()] = v.strip()
            try:
                idx = int(kv.get('id', ''))
                x = float(kv.get('x', ''))
                y = float(kv.get('y', ''))
            except ValueError:
                continue
            if kind.startswith('node'):
                for n in sim.nodes:
                    if n.id == idx:
                        n.x = x
                        n.y = y
                        break
            elif kind.startswith('gw') or kind.startswith('gateway'):
                for gw in sim.gateways:
                    if gw.id == idx:
                        gw.x = x
                        gw.y = y
                        break

    # Appliquer le profil ADR sélectionné
    if selected_adr_module:
        selected_adr_module.apply(sim)

    # La mobilité est désormais gérée directement par le simulateur
    start_time = time.time()
    max_real_time = real_time_duration_input.value if real_time_duration_input.value > 0 else None
    chrono_callback = pn.state.add_periodic_callback(periodic_chrono_update, period=100, timeout=None)

    update_map()
    pdr_indicator.value = 0
    collisions_indicator.value = 0
    energy_indicator.value = 0
    delay_indicator.value = 0
    chrono_indicator.value = 0
    sf_counts = {sf: sum(1 for node in sim.nodes if node.sf == sf) for sf in range(7, 13)}
    sf_fig = go.Figure(
        data=[go.Bar(x=[f"SF{sf}" for sf in sf_counts.keys()], y=list(sf_counts.values()))]
    )
    sf_fig.update_layout(
        title="Répartition des SF par nœud",
        xaxis_title="SF",
        yaxis_title="Nombre de nœuds",
        yaxis_range=[0, sim.num_nodes],
    )
    sf_hist_pane.object = sf_fig
    num_nodes_input.disabled = True
    num_gateways_input.disabled = True
    area_input.disabled = True
    mode_select.disabled = True
    interval_input.disabled = True
    packets_input.disabled = True
    adr_node_checkbox.disabled = True
    adr_server_checkbox.disabled = True
    fixed_sf_checkbox.disabled = True
    sf_value_input.disabled = True
    fixed_power_checkbox.disabled = True
    tx_power_input.disabled = True
    num_channels_input.disabled = True
    channel_dist_select.disabled = True
    mobility_checkbox.disabled = True
    mobility_speed_min_input.disabled = True
    mobility_speed_max_input.disabled = True
    flora_mode_toggle.disabled = True
    detection_threshold_input.disabled = True
    min_interference_input.disabled = True
    battery_capacity_input.disabled = True
    payload_size_input.disabled = True
    node_class_select.disabled = True
    seed_input.disabled = True
    num_runs_input.disabled = True
    real_time_duration_input.disabled = True
    start_button.disabled = True
    stop_button.disabled = False
    fast_forward_button.disabled = False
    pause_button.disabled = False
    pause_button.name = "Pause"
    pause_button.button_type = "primary"
    paused = False
    export_button.disabled = True
    export_message.object = "Clique sur Exporter pour générer le fichier CSV après la simulation."

    sim.running = True
    sim_callback = pn.state.add_periodic_callback(step_simulation, period=100, timeout=None)


# --- Bouton "Lancer la simulation" ---
def on_start(event):
    global total_runs, current_run, runs_events, runs_metrics
    total_runs = int(num_runs_input.value)
    current_run = 1
    runs_events.clear()
    runs_metrics.clear()
    setup_simulation(seed_offset=0)


# --- Bouton "Arrêter la simulation" ---
def on_stop(event):
    global sim, sim_callback, chrono_callback, start_time, max_real_time, paused
    global current_run, total_runs, runs_events, auto_fast_forward
    if sim is None or not sim.running:
        return

    sim.running = False
    if event is not None:
        auto_fast_forward = False
    if sim_callback:
        sim_callback.stop()
        sim_callback = None
    if chrono_callback:
        chrono_callback.stop()
        chrono_callback = None

    try:
        df = sim.get_events_dataframe()
        if df is not None:
            runs_events.append(df.assign(run=current_run))
    except Exception:
        pass
    try:
        runs_metrics.append(sim.get_metrics())
    except Exception:
        pass

    if current_run < total_runs:
        if runs_metrics:
            avg = {
                key: sum(m[key] for m in runs_metrics) / len(runs_metrics)
                for key in runs_metrics[0].keys()
            }
            pdr_indicator.value = avg.get("PDR", 0.0)
            collisions_indicator.value = avg.get("collisions", 0)
            energy_indicator.value = avg.get("energy_J", 0.0)
            delay_indicator.value = avg.get("avg_delay_s", 0.0)
            throughput_indicator.value = avg.get("throughput_bps", 0.0)
        current_run += 1
        seed_offset = current_run - 1
        setup_simulation(seed_offset=seed_offset)
        if auto_fast_forward:
            fast_forward()
        return

    num_nodes_input.disabled = False
    num_gateways_input.disabled = False
    area_input.disabled = False
    mode_select.disabled = False
    interval_input.disabled = False
    packets_input.disabled = False
    adr_node_checkbox.disabled = False
    adr_server_checkbox.disabled = False
    fixed_sf_checkbox.disabled = False
    sf_value_input.disabled = not fixed_sf_checkbox.value
    fixed_power_checkbox.disabled = False
    tx_power_input.disabled = not fixed_power_checkbox.value
    num_channels_input.disabled = False
    channel_dist_select.disabled = False
    mobility_checkbox.disabled = False
    mobility_speed_min_input.disabled = False
    mobility_speed_max_input.disabled = False
    flora_mode_toggle.disabled = False
    detection_threshold_input.disabled = False
    min_interference_input.disabled = False
    battery_capacity_input.disabled = False
    payload_size_input.disabled = False
    node_class_select.disabled = False
    seed_input.disabled = False
    num_runs_input.disabled = False
    real_time_duration_input.disabled = False
    start_button.disabled = False
    stop_button.disabled = True
    fast_forward_button.disabled = True
    pause_button.disabled = True
    pause_button.name = "Pause"
    pause_button.button_type = "primary"
    paused = False
    export_button.disabled = False

    start_time = None
    max_real_time = None
    auto_fast_forward = False
    if runs_metrics:
        avg = {
            key: sum(m[key] for m in runs_metrics) / len(runs_metrics)
            for key in runs_metrics[0].keys()
            if key in runs_metrics[0]
        }
        pdr_indicator.value = avg.get("PDR", 0.0)
        collisions_indicator.value = avg.get("collisions", 0)
        energy_indicator.value = avg.get("energy_J", 0.0)
        delay_indicator.value = avg.get("avg_delay_s", 0.0)
        throughput_indicator.value = avg.get("throughput_bps", 0.0)
        last = runs_metrics[-1]
        table_df = pd.DataFrame(
            {
                "Node": list(last["pdr_by_node"].keys()),
                "PDR": list(last["pdr_by_node"].values()),
                "Recent PDR": [
                    last["recent_pdr_by_node"][nid]
                    for nid in last["pdr_by_node"].keys()
                ],
            }
        )
        pdr_table.object = table_df
    export_message.object = "✅ Simulation terminée. Tu peux exporter les résultats."


# --- Export CSV local : Méthode universelle ---
def exporter_csv(event=None):
    global runs_events, runs_metrics
    if runs_events:
        try:
            df = pd.concat(runs_events, ignore_index=True)
            if df.empty:
                export_message.object = "⚠️ Aucune donnée à exporter !"
                return
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            chemin = os.path.join(os.getcwd(), f"resultats_simulation_{timestamp}.csv")
            df.to_csv(chemin, index=False, encoding="utf-8")
            metrics_path = os.path.join(os.getcwd(), f"metrics_{timestamp}.csv")
            if runs_metrics:
                pd.DataFrame(runs_metrics).to_csv(metrics_path, index=False, encoding="utf-8")
            export_message.object = (
                f"✅ Résultats exportés : <b>{chemin}</b><br>"
                f"Métriques : <b>{metrics_path}</b><br>(Ouvre-les avec Excel ou pandas)"
            )
            try:
                os.startfile(os.getcwd())
            except Exception:
                pass
        except Exception as e:
            export_message.object = f"❌ Erreur lors de l'export : {e}"
    else:
        export_message.object = "⚠️ Lance la simulation d'abord !"


export_button.on_click(exporter_csv)


# --- Bouton d'accélération ---
def fast_forward(event=None):
    global sim, sim_callback, chrono_callback, start_time, max_real_time, auto_fast_forward
    doc = pn.state.curdoc
    if sim and sim.running:
        auto_fast_forward = True
        if sim.packets_to_send == 0:
            export_message.object = (
                "⚠️ Définissez un nombre de paquets supérieur à 0 "
                "pour utiliser l'accélération."
            )
            return

        # Disable buttons during fast forward
        fast_forward_button.disabled = True
        stop_button.disabled = True

        # Stop periodic callbacks to avoid concurrent updates
        if sim_callback:
            sim_callback.stop()
            sim_callback = None
        if chrono_callback:
            chrono_callback.stop()
            chrono_callback = None

        # Pause chrono so time does not keep increasing during fast forward
        start_time = None
        max_real_time = None

        def run_and_update():
            sim.run()

            def update_ui():
                metrics = sim.get_metrics()
                pdr_indicator.value = metrics["PDR"]
                collisions_indicator.value = metrics["collisions"]
                energy_indicator.value = metrics["energy_J"]
                delay_indicator.value = metrics["avg_delay_s"]
                throughput_indicator.value = metrics["throughput_bps"]
                sf_dist = metrics["sf_distribution"]
                sf_fig = go.Figure(
                    data=[go.Bar(x=[f"SF{sf}" for sf in sf_dist.keys()], y=list(sf_dist.values()))]
                )
                sf_fig.update_layout(
                    title="Répartition des SF par nœud",
                    xaxis_title="SF",
                    yaxis_title="Nombre de nœuds",
                    yaxis_range=[0, sim.num_nodes],
                )
                sf_hist_pane.object = sf_fig
                update_map()
                on_stop(None)

            doc.add_next_tick_callback(update_ui)

        threading.Thread(target=run_and_update, daemon=True).start()


fast_forward_button.on_click(fast_forward)


# --- Bouton "Pause/Reprendre" ---
def on_pause(event=None):
    global sim_callback, chrono_callback, start_time, elapsed_time, paused
    if sim is None or not sim.running:
        return

    if not paused:
        if sim_callback:
            sim_callback.stop()
            sim_callback = None
        if chrono_callback:
            chrono_callback.stop()
            chrono_callback = None
        if start_time is not None:
            elapsed_time = time.time() - start_time
        pause_button.name = "Reprendre"
        pause_button.button_type = "success"
        paused = True
    else:
        start_time = time.time() - elapsed_time
        if sim_callback is None:
            sim_callback = pn.state.add_periodic_callback(step_simulation, period=100, timeout=None)
        if chrono_callback is None:
            chrono_callback = pn.state.add_periodic_callback(periodic_chrono_update, period=100, timeout=None)
        pause_button.name = "Pause"
        pause_button.button_type = "primary"
        paused = False


pause_button.on_click(on_pause)


# --- Case à cocher mobilité : pour mobilité à chaud, hors simulation ---
def on_mobility_toggle(event):
    global sim
    if sim and sim.running:
        sim.mobility_enabled = event.new
        if event.new:
            for node in sim.nodes:
                sim.mobility_model.assign(node)
                sim.schedule_mobility(node, sim.current_time + sim.mobility_model.step)


mobility_checkbox.param.watch(on_mobility_toggle, "value")


# --- Activation des champs SF et puissance ---
def on_fixed_sf_toggle(event):
    sf_value_input.disabled = not event.new


def on_fixed_power_toggle(event):
    tx_power_input.disabled = not event.new


fixed_sf_checkbox.param.watch(on_fixed_sf_toggle, "value")
fixed_power_checkbox.param.watch(on_fixed_power_toggle, "value")

# --- Affichage zone manuelle ---
def on_manual_toggle(event):
    position_textarea.visible = event.new

manual_pos_toggle.param.watch(on_manual_toggle, "value")

# --- Mode FLoRa complet ---
def on_flora_toggle(event):
    if event.new:
        detection_threshold_input.value = -110.0
        # En mode FLoRa, la durée minimale d'interférence est fixée à 5 s
        min_interference_input.value = 5.0
        detection_threshold_input.disabled = True
        min_interference_input.disabled = True
        flora_mode_toggle.button_type = "primary"
    else:
        detection_threshold_input.disabled = False
        min_interference_input.disabled = False
        flora_mode_toggle.button_type = "default"

flora_mode_toggle.param.watch(on_flora_toggle, "value")

# --- Boutons ADR ---
adr1_button.on_click(lambda event: select_adr(adr_standard_1, "ADR 1"))
adr2_button.on_click(lambda event: select_adr(adr_2, "ADR 2"))
adr3_button.on_click(lambda event: select_adr(adr_3, "ADR 3"))

# --- Associer les callbacks aux boutons ---
start_button.on_click(on_start)
stop_button.on_click(on_stop)

# --- Mise en page du dashboard ---
controls = pn.WidgetBox(
    num_nodes_input,
    num_gateways_input,
    area_input,
    mode_select,
    interval_input,
    packets_input,
    seed_input,
    num_runs_input,
    adr_node_checkbox,
    adr_server_checkbox,
    pn.Row(adr1_button, adr2_button, adr3_button, adr_active_badge),
    fixed_sf_checkbox,
    sf_value_input,
    fixed_power_checkbox,
    tx_power_input,
    num_channels_input,
    channel_dist_select,
    mobility_checkbox,
    mobility_speed_min_input,
    mobility_speed_max_input,
    flora_mode_toggle,
    detection_threshold_input,
    min_interference_input,
    battery_capacity_input,
    payload_size_input,
    node_class_select,
    real_time_duration_input,
    pn.Row(start_button, stop_button),
    pn.Row(fast_forward_button, pause_button),
    export_button,
    export_message,
)
controls.width = 350

metrics_col = pn.Column(
    chrono_indicator,
    pdr_indicator,
    collisions_indicator,
    energy_indicator,
    delay_indicator,
    throughput_indicator,
    pdr_table,
)
metrics_col.width = 220

center_col = pn.Column(
    map_pane,
    sf_hist_pane,
    pn.Column(manual_pos_toggle, position_textarea, width=400),
    sizing_mode="stretch_width",
)
center_col.width = 650

dashboard = pn.Row(
    controls,
    center_col,
    metrics_col,
    sizing_mode="stretch_width",
)
dashboard.servable(title="Simulateur LoRa")

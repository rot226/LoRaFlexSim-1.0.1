.PHONY: validate

validate:
	pytest -k "channel"
	pytest -k "omnet_phy or rx_chain or overlap_snir or flora_capture or startup_currents or pa_ramp"
	pytest -k "gateway or collision_capture or compare_flora"
	pytest -k "network_server or no_random_drop or run_simulate or class_bc"
	pytest -k "run_simulate or rest_api_gap or dashboard"
	pytest -k "lorawan or class_a or rx_windows or adr or flora_energy"
	pytest -k "mobility"
	pytest -k "rest_api or web_api"
	python scripts/run_validation.py

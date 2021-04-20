# P2P Energy Sharing simulation 

design and development:
- [Alex Gabriel](https://github.com/gabriel-alex)
- [Rima Oulhaj](https://gitlab.com/rimaoulhaj)

supervised by:
- Alex Gabriel
- Laurent Dupont 

## Description 
The project aims to create an open source P2P energy sharing simulation using optimization and multi-agent system. 

### Data used
Information concerning tariff can be found on [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/arretes-tarifaires-photovoltaiques-en-metropole/) or [photovoltaique.info](https://www.photovoltaique.info/fr/tarifs-dachat-et-autoconsommation/tarifs-dachat/arrete-tarifaire-en-vigueur/). In the case of saling surplus to inject it in the grid, the price is 0.10€/kWh.


## Set up the projet

This procedure to experiment the projet suppose you already have installed Python3 on your computer and you are quite familiar with virtual environment in Python.

```bash 
# create the virtual environment 
python3 -m venv .

# activate the virtual environment
source bin/activate # on Linux environment 
Scripts/activate # on windows

# install packages
# on windows remove pkg-resources==0.0.0 from requirements.txt
pip3 install -r requirements.txt

# deactivate the virtual environment
deactivate
```

Simulating 4 independant prosumers with PV
```bash
python3 main_agent_battery_off.py
```

Simulating 4 independant prosumers with PV and battery 
```bash
python3 main_agent_with_battery.py
```

Simulating 4 optimized independant prosumers with PV and bettery
```bash
python3 main_agent_with_battery_opti.py
```
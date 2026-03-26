"""
Green Energy Theme Portfolio Analyzer & Nifty Prediction Engine.

Analyses the historical Green Energy Theme portfolio to learn stock selection
patterns, then applies those learned factors to generate buy predictions
for the broader Nifty universe.
"""

import json
import math
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

DATA_PATH = Path("data") / "green_energy_portfolio.json"

# ── Stock sector/theme classification (known green-energy universe) ───────
STOCK_THEMES: dict[str, list[str]] = {
    "Triveni Turbine Ltd": ["Power Equipment", "Steam Turbines", "Clean Energy"],
    "Borosil Renewables Ltd": ["Solar Glass", "Solar Value Chain", "Manufacturing"],
    "Shivalik Bimetal Controls Ltd": ["Electrical Components", "Bimetal Strips", "Grid Infrastructure"],
    "TD Power Systems Ltd": ["Generators", "Power Systems", "Hydro/Wind"],
    "Fiem Industries Ltd": ["Auto LED/Electronics", "EV Ecosystem", "Energy Efficiency"],
    "Ganesha Ecosphere Ltd": ["Recycled Polyester", "Circular Economy", "Sustainability"],
    "Praj Industries Ltd": ["Bio-Energy Equipment", "Ethanol", "Clean Fuels"],
    "Pitti Engineering Ltd": ["Electrical Stampings", "Motors", "Grid Infrastructure"],
    "KPIT Technologies Ltd": ["Auto Software", "EV Software", "EV Ecosystem"],
    "Sanghvi Movers Ltd": ["Crane Rentals", "Wind Turbine Installation", "Infrastructure"],
    "Bharat Bijlee Ltd": ["Transformers", "Motors", "Grid Infrastructure"],
    "Apar Industries Ltd": ["Conductors", "Specialty Oils", "Grid Infrastructure"],
    "MTAR Technologies Ltd": ["Precision Engineering", "Nuclear/Space", "Clean Energy"],
    "Skipper Ltd": ["Transmission Towers", "PVC Pipes", "Grid Infrastructure"],
    "Tata Power Company Ltd": ["Power Utility", "Solar EPC", "Clean Energy"],
    "Inox Wind Ltd": ["Wind Turbines", "Wind Energy", "Clean Energy"],
    "Ion Exchange (India) Ltd": ["Water Treatment", "Environment", "Sustainability"],
    "Shakti Pumps (India) Ltd": ["Solar Pumps", "Water Pumps", "Clean Energy"],
    "HPL Electric & Power Ltd": ["Smart Meters", "Switchgear", "Grid Infrastructure"],
    "KSB Ltd": ["Industrial Pumps", "Valves", "Infrastructure"],
    "Kalpataru Projects International Ltd": ["Transmission EPC", "Grid Infrastructure", "Power T&D"],
    "Sterling and Wilson Renewable Energy Ltd": ["Solar EPC", "Clean Energy", "Utility Scale Solar"],
    "Hitachi Energy India Ltd": ["Power Electronics", "Grid Automation", "Grid Infrastructure"],
    "Genus Power Infrastructures Ltd": ["Smart Meters", "Metering", "Grid Infrastructure"],
    "Techno Electric & Engineering Company Ltd": ["Power T&D EPC", "Wind Farms", "Clean Energy"],
    "Swelect Energy Systems Ltd": ["Solar Modules", "Solar Value Chain", "Manufacturing"],
    "Pennar Industries Ltd": ["Steel Structures", "Solar Mounting", "Infrastructure"],
    "Websol Energy System Ltd": ["Solar Cells", "Solar Value Chain", "Manufacturing"],
    "Pondy Oxides and Chemicals Ltd": ["Lead Recycling", "Circular Economy", "Battery Ecosystem"],
    "Jash Engineering Ltd": ["Water Infrastructure", "Sluice Gates", "Environment"],
    "Power Mech Projects Ltd": ["Power Plant Services", "O&M", "Infrastructure"],
    "Transrail Lighting Ltd": ["Transmission Towers", "Conductors", "Grid Infrastructure"],
    "Syrma SGS Technology Ltd": ["EMS", "IoT Devices", "Electronics Manufacturing"],
    "Gravita India Ltd": ["Lead/Aluminium Recycling", "Circular Economy", "Sustainability"],
    "Indian Energy Exchange Ltd": ["Energy Exchange", "Power Trading", "Clean Energy"],
    "Praj Industries Ltd": ["Bio-Energy", "Ethanol Plants", "Clean Fuels"],
    "Waaree Energies Ltd": ["Solar Modules", "Solar Value Chain", "Manufacturing"],
    "Premier Energies Ltd": ["Solar Cells/Modules", "Solar Value Chain", "Manufacturing"],
    "ACME Solar Holdings Ltd": ["Solar IPP", "Utility Solar", "Clean Energy"],
    "Exide Industries Ltd": ["Batteries", "Energy Storage", "Battery Ecosystem"],
    "REC Limited": ["Power Finance", "Grid Lending", "Infrastructure Finance"],
    "Sona BLW Precision Forgings Ltd": ["Differential Gears", "EV Drivetrain", "EV Ecosystem"],
    "HBL Engineering Ltd": ["Specialty Batteries", "Railway Electronics", "Energy Storage"],
    "Neogen Chemicals Ltd": ["Lithium Compounds", "Specialty Chemicals", "Battery Ecosystem"],
    "Everest Kanto Cylinder Ltd": ["CNG/Hydrogen Cylinders", "Clean Fuels", "Gas Infrastructure"],
    "INOX India Ltd": ["Cryogenic Equipment", "LNG/Industrial Gas", "Gas Infrastructure"],
    "GNG Electronics Ltd": ["EMS", "Power Electronics", "Electronics Manufacturing"],
    "Jain Resource Recycling Ltd": ["E-waste Recycling", "Circular Economy", "Sustainability"],
    "CE Info Systems Ltd": ["Mapping/GIS", "EV Navigation", "Technology"],
    "Emmvee Photovoltaic Power Ltd": ["Solar Modules", "Solar Value Chain", "Manufacturing"],
    "Ather Energy Ltd": ["Electric 2-Wheelers", "EV OEM", "EV Ecosystem"],
    "Sansera Engineering Ltd": ["Precision Forgings", "Auto Components", "EV Ecosystem"],
    "Siemens Energy India Ltd": ["Power T&D", "Grid Equipment", "Grid Infrastructure"],
    "Avalon Technologies Ltd": ["EMS", "PCB Assembly", "Electronics Manufacturing"],
    "Transformers and Rectifiers (India) Ltd": ["Transformers", "Grid Infrastructure", "Power T&D"],
    "Oswal Pumps Ltd": ["Pumps", "Agriculture", "Water Infrastructure"],
    "Quality Power Electrical Equipments Ltd": ["Power Equipment", "Grid Infrastructure", "Transformers"],
    "Vedanta Ltd": ["Mining", "Zinc/Aluminium", "Commodities"],
    "Thermax Limited": ["Boilers", "Clean Tech", "Energy Efficiency"],
    "KEC International Ltd": ["Transmission EPC", "Grid Infrastructure", "Power T&D"],
    "Adani Green Energy Ltd": ["Solar/Wind IPP", "Utility Scale Renewable", "Clean Energy"],
    "Greaves Cotton Ltd": ["Small Engines", "EV 3-Wheeler", "EV Ecosystem"],
    "Crompton Greaves Consumer Electricals Ltd": ["Fans/Lighting", "Energy Efficiency", "Consumer"],
    "Nippon India ETF Nifty 1D Rate Liquid BeES": ["Liquid ETF", "Cash Management", "ETF"],
    "Zerodha Nifty 1D Rate Liquid ETF": ["Liquid ETF", "Cash Management", "ETF"],
}

# ── Nifty Broad-Based Indices Universe for Buy Predictions ────────────────
# Covers stocks across: Nifty 50, Next 50, 100, 200, 500, Total Market,
# 500 Multicap 50:25:25, 500 LargeMidSmall Equal-Cap, Midcap 150/50/Select/100,
# Smallcap 500/250/50/100, Microcap 250, LargeMidcap 250, MidSmallcap 400/50:50,
# India FPI 150 — filtered for green-energy, power, EV, grid, sustainability themes.
NIFTY_PREDICTION_UNIVERSE: dict[str, dict[str, Any]] = {
    # ───────────── NIFTY 50 / NEXT 50 (Large Cap) ─────────────
    "NTPC": {"name": "NTPC Ltd", "sector": "Power", "themes": ["Solar Expansion", "Clean Energy", "PSU"], "mcap_cr": 350000, "indices": ["Nifty 50"]},
    "TATAPOWER": {"name": "Tata Power Company Ltd", "sector": "Power", "themes": ["Solar EPC", "Clean Energy", "Power Utility"], "mcap_cr": 125000, "indices": ["Nifty 50"]},
    "ADANIGREEN": {"name": "Adani Green Energy Ltd", "sector": "Power", "themes": ["Solar IPP", "Wind IPP", "Clean Energy"], "mcap_cr": 180000, "indices": ["Nifty Next 50"]},
    "POWERGRID": {"name": "Power Grid Corporation", "sector": "Power", "themes": ["Grid Infrastructure", "Transmission", "PSU"], "mcap_cr": 280000, "indices": ["Nifty 50"]},
    "ADANIENSO": {"name": "Adani Energy Solutions", "sector": "Power", "themes": ["Power T&D", "Smart Metering", "Grid Infrastructure"], "mcap_cr": 95000, "indices": ["Nifty Next 50"]},
    "SIEMENS": {"name": "Siemens Ltd", "sector": "Capital Goods", "themes": ["Grid Automation", "Power Electronics", "Grid Infrastructure"], "mcap_cr": 180000, "indices": ["Nifty Next 50"]},
    "ABB": {"name": "ABB India Ltd", "sector": "Capital Goods", "themes": ["Power Electronics", "Automation", "Grid Infrastructure"], "mcap_cr": 130000, "indices": ["Nifty Next 50"]},
    "RECLTD": {"name": "REC Limited", "sector": "Finance", "themes": ["Power Finance", "Grid Lending", "PSU"], "mcap_cr": 120000, "indices": ["Nifty 50"]},
    "PFC": {"name": "Power Finance Corporation", "sector": "Finance", "themes": ["Power Finance", "Grid Lending", "PSU"], "mcap_cr": 150000, "indices": ["Nifty Next 50"]},
    "NHPC": {"name": "NHPC Ltd", "sector": "Power", "themes": ["Hydro Power", "Clean Energy", "PSU"], "mcap_cr": 90000, "indices": ["Nifty Next 50"]},
    "CUMMINSIND": {"name": "Cummins India Ltd", "sector": "Capital Goods", "themes": ["Power Generators", "Diesel/Gas", "Power Equipment"], "mcap_cr": 55000, "indices": ["Nifty Next 50"]},
    "TATAELXSI": {"name": "Tata Elxsi Ltd", "sector": "IT", "themes": ["Auto Software", "EV Software", "EV Ecosystem"], "mcap_cr": 28000, "indices": ["Nifty Next 50"]},
    "BHEL": {"name": "Bharat Heavy Electricals", "sector": "Capital Goods", "themes": ["Power Equipment", "Solar EPC", "PSU"], "mcap_cr": 85000, "indices": ["Nifty Next 50"]},
    "TATAMOTORS": {"name": "Tata Motors Ltd", "sector": "Automobiles", "themes": ["EV OEM", "EV Ecosystem", "Auto"], "mcap_cr": 250000, "indices": ["Nifty 50"]},
    "M&M": {"name": "Mahindra & Mahindra", "sector": "Automobiles", "themes": ["EV OEM", "Farm Equipment", "EV Ecosystem"], "mcap_cr": 380000, "indices": ["Nifty 50"]},
    "BAJAJ-AUTO": {"name": "Bajaj Auto Ltd", "sector": "Automobiles", "themes": ["EV 2-Wheeler", "CNG Vehicles", "EV Ecosystem"], "mcap_cr": 240000, "indices": ["Nifty 50"]},
    "RELIANCE": {"name": "Reliance Industries", "sector": "Conglomerate", "themes": ["Green Hydrogen", "New Energy", "Solar Manufacturing"], "mcap_cr": 1700000, "indices": ["Nifty 50"]},
    "ADANIPORTS": {"name": "Adani Ports & SEZ", "sector": "Infrastructure", "themes": ["Green Ports", "Infrastructure", "Logistics"], "mcap_cr": 280000, "indices": ["Nifty 50"]},
    "JSL": {"name": "Jindal Stainless Ltd", "sector": "Metals", "themes": ["Stainless Steel", "Infrastructure", "Manufacturing"], "mcap_cr": 52000, "indices": ["Nifty Next 50"]},

    # ───────────── NIFTY 100 / 200 (Large-Mid) ─────────────
    "SJVN": {"name": "SJVN Ltd", "sector": "Power", "themes": ["Hydro/Solar", "Clean Energy", "PSU"], "mcap_cr": 42000, "indices": ["Nifty 200"]},
    "IREDA": {"name": "IREDA", "sector": "Finance", "themes": ["Green Finance", "Clean Energy Lending", "PSU"], "mcap_cr": 48000, "indices": ["Nifty 200"]},
    "SUZLON": {"name": "Suzlon Energy Ltd", "sector": "Capital Goods", "themes": ["Wind Turbines", "Wind Energy", "Clean Energy"], "mcap_cr": 72000, "indices": ["Nifty 200"]},
    "KPITTECH": {"name": "KPIT Technologies Ltd", "sector": "IT", "themes": ["EV Software", "Auto Tech", "EV Ecosystem"], "mcap_cr": 38000, "indices": ["Nifty 200"]},
    "SONACOMS": {"name": "Sona BLW Precision Forgings", "sector": "Auto Ancillary", "themes": ["EV Drivetrain", "Differential Gears", "EV Ecosystem"], "mcap_cr": 32000, "indices": ["Nifty 200"]},
    "THERMAX": {"name": "Thermax Limited", "sector": "Capital Goods", "themes": ["Clean Tech", "Boilers", "Energy Efficiency"], "mcap_cr": 22000, "indices": ["Nifty 200"]},
    "EXIDEIND": {"name": "Exide Industries Ltd", "sector": "Auto Ancillary", "themes": ["Batteries", "Energy Storage", "Battery Ecosystem"], "mcap_cr": 32000, "indices": ["Nifty 200"]},
    "AMARARAJA": {"name": "Amara Raja Energy", "sector": "Auto Ancillary", "themes": ["Batteries", "Energy Storage", "Lithium-ion"], "mcap_cr": 20000, "indices": ["Nifty 200"]},
    "APARINDS": {"name": "Apar Industries Ltd", "sector": "Capital Goods", "themes": ["Conductors", "Grid Infrastructure", "Power T&D"], "mcap_cr": 25000, "indices": ["Nifty 200"]},
    "KALPATPOWR": {"name": "Kalpataru Projects Intl", "sector": "Infrastructure", "themes": ["Transmission EPC", "Grid Infrastructure", "Power T&D"], "mcap_cr": 18000, "indices": ["Nifty 200"]},
    "OLECTRA": {"name": "Olectra Greentech Ltd", "sector": "Automobiles", "themes": ["Electric Buses", "EV OEM", "EV Ecosystem"], "mcap_cr": 10000, "indices": ["Nifty 200"]},
    "JBMA": {"name": "JBM Auto Ltd", "sector": "Automobiles", "themes": ["Electric Buses", "EV OEM", "EV Ecosystem"], "mcap_cr": 12000, "indices": ["Nifty 200"]},
    "CGPOWER": {"name": "CG Power & Industrial", "sector": "Capital Goods", "themes": ["Motors", "Transformers", "Grid Infrastructure"], "mcap_cr": 95000, "indices": ["Nifty 200"]},
    "KAYNES": {"name": "Kaynes Technology India", "sector": "Capital Goods", "themes": ["EMS", "IoT", "Electronics Manufacturing"], "mcap_cr": 22000, "indices": ["Nifty 200"]},
    "DIXON": {"name": "Dixon Technologies", "sector": "Consumer Electronics", "themes": ["EMS", "Electronics Manufacturing", "Manufacturing"], "mcap_cr": 62000, "indices": ["Nifty 100"]},
    "TRENT": {"name": "Trent Ltd", "sector": "Retail", "themes": ["Sustainability", "Retail", "Consumer"], "mcap_cr": 200000, "indices": ["Nifty 100"]},
    "VOLTAS": {"name": "Voltas Ltd", "sector": "Capital Goods", "themes": ["Energy Efficiency", "HVAC", "Consumer"], "mcap_cr": 40000, "indices": ["Nifty 200"]},
    "SCHAEFFLER": {"name": "Schaeffler India", "sector": "Auto Ancillary", "themes": ["Bearings", "EV Drivetrain", "EV Ecosystem"], "mcap_cr": 28000, "indices": ["Nifty 200"]},

    # ───────────── NIFTY 500 / NIFTY TOTAL MARKET ─────────────
    "WAAREEENER": {"name": "Waaree Energies Ltd", "sector": "Capital Goods", "themes": ["Solar Modules", "Solar Value Chain", "Manufacturing"], "mcap_cr": 55000, "indices": ["Nifty 500"]},
    "PREMIERENER": {"name": "Premier Energies Ltd", "sector": "Capital Goods", "themes": ["Solar Cells", "Solar Value Chain", "Manufacturing"], "mcap_cr": 28000, "indices": ["Nifty 500"]},
    "INOXWIND": {"name": "Inox Wind Ltd", "sector": "Capital Goods", "themes": ["Wind Turbines", "Wind Energy", "Clean Energy"], "mcap_cr": 14000, "indices": ["Nifty 500"]},
    "TRITURBINE": {"name": "Triveni Turbine Ltd", "sector": "Capital Goods", "themes": ["Steam Turbines", "Power Equipment", "Clean Energy"], "mcap_cr": 18000, "indices": ["Nifty 500"]},
    "TDPOWERSYS": {"name": "TD Power Systems Ltd", "sector": "Capital Goods", "themes": ["Generators", "Power Systems", "Hydro/Wind"], "mcap_cr": 8000, "indices": ["Nifty 500"]},
    "PRAJIND": {"name": "Praj Industries Ltd", "sector": "Capital Goods", "themes": ["Bio-Energy", "Ethanol Plants", "Clean Fuels"], "mcap_cr": 8600, "indices": ["Nifty 500"]},
    "GRAVITA": {"name": "Gravita India Ltd", "sector": "Metals", "themes": ["Lead Recycling", "Circular Economy", "Sustainability"], "mcap_cr": 15000, "indices": ["Nifty 500"]},
    "IONEXCHANG": {"name": "Ion Exchange (India) Ltd", "sector": "Capital Goods", "themes": ["Water Treatment", "Environment", "Sustainability"], "mcap_cr": 5500, "indices": ["Nifty 500"]},
    "KSB": {"name": "KSB Ltd", "sector": "Capital Goods", "themes": ["Industrial Pumps", "Valves", "Infrastructure"], "mcap_cr": 9000, "indices": ["Nifty 500"]},
    "GENUSPOWER": {"name": "Genus Power Infrastructures", "sector": "Capital Goods", "themes": ["Smart Meters", "Metering", "Grid Infrastructure"], "mcap_cr": 9500, "indices": ["Nifty 500"]},
    "SKIPPER": {"name": "Skipper Ltd", "sector": "Capital Goods", "themes": ["Transmission Towers", "Grid Infrastructure", "PVC"], "mcap_cr": 9000, "indices": ["Nifty 500"]},
    "ATHERENER": {"name": "Ather Energy Ltd", "sector": "Automobiles", "themes": ["Electric 2-Wheelers", "EV OEM", "EV Ecosystem"], "mcap_cr": 12000, "indices": ["Nifty 500"]},
    "ACMESOLAR": {"name": "ACME Solar Holdings", "sector": "Power", "themes": ["Solar IPP", "Utility Solar", "Clean Energy"], "mcap_cr": 12000, "indices": ["Nifty 500"]},
    "TECHNOELEC": {"name": "Techno Electric & Engg", "sector": "Capital Goods", "themes": ["Power T&D EPC", "Wind Farms", "Clean Energy"], "mcap_cr": 7000, "indices": ["Nifty 500"]},
    "HBLENGR": {"name": "HBL Engineering Ltd", "sector": "Capital Goods", "themes": ["Specialty Batteries", "Railway Electronics", "Energy Storage"], "mcap_cr": 7500, "indices": ["Nifty 500"]},
    "TRANSRAIL": {"name": "Transrail Lighting Ltd", "sector": "Capital Goods", "themes": ["Transmission Towers", "Conductors", "Grid Infrastructure"], "mcap_cr": 5500, "indices": ["Nifty 500"]},
    "NEOGENECHEM": {"name": "Neogen Chemicals Ltd", "sector": "Chemicals", "themes": ["Lithium Compounds", "Specialty Chemicals", "Battery Ecosystem"], "mcap_cr": 4000, "indices": ["Nifty 500"]},
    "HAPPSTMNDS": {"name": "Happiest Minds Technologies", "sector": "IT", "themes": ["IoT", "Digital Infra", "Technology"], "mcap_cr": 8000, "indices": ["Nifty 500"]},
    "BOROSILREN": {"name": "Borosil Renewables Ltd", "sector": "Capital Goods", "themes": ["Solar Glass", "Solar Value Chain", "Manufacturing"], "mcap_cr": 4500, "indices": ["Nifty 500"]},
    "JSWENERGY": {"name": "JSW Energy Ltd", "sector": "Power", "themes": ["Solar/Wind", "Green Hydrogen", "Clean Energy"], "mcap_cr": 60000, "indices": ["Nifty 500"]},
    "TORNTPOWER": {"name": "Torrent Power Ltd", "sector": "Power", "themes": ["Solar EPC", "Distribution", "Clean Energy"], "mcap_cr": 55000, "indices": ["Nifty 500"]},
    "CESC": {"name": "CESC Ltd", "sector": "Power", "themes": ["Power Distribution", "Renewable IPP", "Power Utility"], "mcap_cr": 20000, "indices": ["Nifty 500"]},
    "NSLNISP": {"name": "NTPC Green Energy Ltd", "sector": "Power", "themes": ["Solar IPP", "Wind IPP", "Clean Energy"], "mcap_cr": 45000, "indices": ["Nifty 500"]},
    "JPPOWER": {"name": "Jaiprakash Power Ventures", "sector": "Power", "themes": ["Hydro Power", "Clean Energy", "Infrastructure"], "mcap_cr": 8000, "indices": ["Nifty 500"]},
    "ORIENTELEC": {"name": "Orient Electric Ltd", "sector": "Consumer Durables", "themes": ["Energy Efficiency", "Fans/Lighting", "Consumer"], "mcap_cr": 6500, "indices": ["Nifty 500"]},
    "ELGIEQUIP": {"name": "Elgi Equipments Ltd", "sector": "Capital Goods", "themes": ["Compressors", "Energy Efficiency", "Manufacturing"], "mcap_cr": 15000, "indices": ["Nifty 500"]},
    "KEI": {"name": "KEI Industries Ltd", "sector": "Capital Goods", "themes": ["Cables", "Grid Infrastructure", "Power T&D"], "mcap_cr": 28000, "indices": ["Nifty 500"]},
    "POLYCAB": {"name": "Polycab India Ltd", "sector": "Capital Goods", "themes": ["Cables & Wires", "Grid Infrastructure", "Power T&D"], "mcap_cr": 70000, "indices": ["Nifty 200"]},
    "HAVELLS": {"name": "Havells India Ltd", "sector": "Capital Goods", "themes": ["Cables", "Energy Efficiency", "Consumer"], "mcap_cr": 65000, "indices": ["Nifty 100"]},
    "BFRG": {"name": "BF Utilities Ltd", "sector": "Power", "themes": ["Wind Energy", "Clean Energy", "Utilities"], "mcap_cr": 3500, "indices": ["Nifty 500"]},
    "GE": {"name": "GE T&D India Ltd", "sector": "Capital Goods", "themes": ["Grid Equipment", "Substations", "Grid Infrastructure"], "mcap_cr": 15000, "indices": ["Nifty 500"]},
    "RATNAMANI": {"name": "Ratnamani Metals & Tubes", "sector": "Metals", "themes": ["Pipes", "Infrastructure", "Manufacturing"], "mcap_cr": 13000, "indices": ["Nifty 500"]},
    "GRAPHITE": {"name": "Graphite India Ltd", "sector": "Metals", "themes": ["Graphite Electrodes", "EV Battery Supply Chain", "Battery Ecosystem"], "mcap_cr": 7000, "indices": ["Nifty 500"]},
    "HIMATSEIDE": {"name": "Himatsingka Seide Ltd", "sector": "Textiles", "themes": ["Sustainable Textiles", "Sustainability", "Manufacturing"], "mcap_cr": 2000, "indices": ["Nifty 500"]},

    # ───────────── NIFTY MIDCAP 150 / 100 / 50 / SELECT ─────────────
    "MTARTECH": {"name": "MTAR Technologies Ltd", "sector": "Capital Goods", "themes": ["Precision Engineering", "Nuclear/Space", "Clean Energy"], "mcap_cr": 6000, "indices": ["Nifty Midcap 100"]},
    "PITTIENG": {"name": "Pitti Engineering Ltd", "sector": "Capital Goods", "themes": ["Electrical Stampings", "Motors", "Grid Infrastructure"], "mcap_cr": 4000, "indices": ["Nifty Midcap 150"]},
    "BHARBIJLEE": {"name": "Bharat Bijlee Ltd", "sector": "Capital Goods", "themes": ["Transformers", "Motors", "Grid Infrastructure"], "mcap_cr": 3500, "indices": ["Nifty Midcap 150"]},
    "FIEMIND": {"name": "Fiem Industries Ltd", "sector": "Auto Ancillary", "themes": ["Auto LED", "EV Ecosystem", "Energy Efficiency"], "mcap_cr": 5500, "indices": ["Nifty Midcap 150"]},
    "SANGHVIMOV": {"name": "Sanghvi Movers Ltd", "sector": "Capital Goods", "themes": ["Crane Rentals", "Wind Installation", "Infrastructure"], "mcap_cr": 3800, "indices": ["Nifty Midcap 150"]},
    "HPLELECTRIC": {"name": "HPL Electric & Power", "sector": "Capital Goods", "themes": ["Smart Meters", "Switchgear", "Grid Infrastructure"], "mcap_cr": 3200, "indices": ["Nifty Midcap 150"]},
    "GANESHECO": {"name": "Ganesha Ecosphere Ltd", "sector": "Textiles", "themes": ["Recycled Polyester", "Circular Economy", "Sustainability"], "mcap_cr": 2800, "indices": ["Nifty Midcap 150"]},
    "SHIVALIK": {"name": "Shivalik Bimetal Controls", "sector": "Capital Goods", "themes": ["Bimetal Strips", "Electrical Components", "Grid Infrastructure"], "mcap_cr": 3500, "indices": ["Nifty Midcap 150"]},
    "POWERMECH": {"name": "Power Mech Projects Ltd", "sector": "Infrastructure", "themes": ["Power Plant O&M", "Infrastructure", "Clean Energy"], "mcap_cr": 5000, "indices": ["Nifty Midcap 150"]},
    "SYRMA": {"name": "Syrma SGS Technology Ltd", "sector": "Capital Goods", "themes": ["EMS", "IoT Devices", "Electronics Manufacturing"], "mcap_cr": 6000, "indices": ["Nifty Midcap 150"]},
    "TRIL": {"name": "Transformers & Rectifiers India", "sector": "Capital Goods", "themes": ["Transformers", "Grid Infrastructure", "Power T&D"], "mcap_cr": 8000, "indices": ["Nifty Midcap 100"]},
    "PENIND": {"name": "Pennar Industries Ltd", "sector": "Infrastructure", "themes": ["Steel Structures", "Solar Mounting", "Infrastructure"], "mcap_cr": 3000, "indices": ["Nifty Midcap 150"]},
    "GREAVESCOT": {"name": "Greaves Cotton Ltd", "sector": "Automobiles", "themes": ["EV 3-Wheeler", "Small Engines", "EV Ecosystem"], "mcap_cr": 3500, "indices": ["Nifty Midcap 150"]},
    "INAEX": {"name": "Indian Energy Exchange", "sector": "Finance", "themes": ["Energy Exchange", "Power Trading", "Clean Energy"], "mcap_cr": 12000, "indices": ["Nifty Midcap 100"]},
    "TRIVENI": {"name": "Triveni Engineering", "sector": "Capital Goods", "themes": ["Ethanol", "Clean Fuels", "Bio-Energy"], "mcap_cr": 8500, "indices": ["Nifty Midcap 100"]},
    "SWSOLAR": {"name": "Sterling & Wilson Solar", "sector": "Capital Goods", "themes": ["Solar EPC", "Clean Energy", "Utility Scale Solar"], "mcap_cr": 5000, "indices": ["Nifty Midcap 150"]},
    "JASHENG": {"name": "Jash Engineering Ltd", "sector": "Capital Goods", "themes": ["Water Infrastructure", "Sluice Gates", "Environment"], "mcap_cr": 3000, "indices": ["Nifty Midcap 150"]},
    "AVALON": {"name": "Avalon Technologies Ltd", "sector": "Capital Goods", "themes": ["EMS", "PCB Assembly", "Electronics Manufacturing"], "mcap_cr": 3500, "indices": ["Nifty Midcap 150"]},
    "VOLTAMP": {"name": "Voltamp Transformers Ltd", "sector": "Capital Goods", "themes": ["Transformers", "Grid Infrastructure", "Power T&D"], "mcap_cr": 7000, "indices": ["Nifty Midcap 100"]},
    "HITECHCORP": {"name": "Hitachi Energy India Ltd", "sector": "Capital Goods", "themes": ["Power Electronics", "Grid Automation", "Grid Infrastructure"], "mcap_cr": 60000, "indices": ["Nifty Midcap Select"]},
    "GIPCL": {"name": "Gujarat Industries Power", "sector": "Power", "themes": ["Wind/Solar IPP", "Clean Energy", "PSU"], "mcap_cr": 4500, "indices": ["Nifty Midcap 150"]},
    "RVNL": {"name": "Rail Vikas Nigam Ltd", "sector": "Infrastructure", "themes": ["Railway Electrification", "Infrastructure", "PSU"], "mcap_cr": 45000, "indices": ["Nifty Midcap 50"]},
    "IRCON": {"name": "Ircon International Ltd", "sector": "Infrastructure", "themes": ["Railway Electrification", "Infrastructure", "PSU"], "mcap_cr": 15000, "indices": ["Nifty Midcap 100"]},

    # ───────────── NIFTY SMALLCAP 500 / 250 / 100 / 50 ─────────────
    "SWELECTES": {"name": "Swelect Energy Systems", "sector": "Capital Goods", "themes": ["Solar Modules", "Solar Value Chain", "Manufacturing"], "mcap_cr": 1200, "indices": ["Nifty Smallcap 250"]},
    "WEBSOL": {"name": "Websol Energy System Ltd", "sector": "Capital Goods", "themes": ["Solar Cells", "Solar Value Chain", "Manufacturing"], "mcap_cr": 1800, "indices": ["Nifty Smallcap 250"]},
    "EMMVEE": {"name": "Emmvee Photovoltaic Power", "sector": "Capital Goods", "themes": ["Solar Modules", "Solar Value Chain", "Manufacturing"], "mcap_cr": 800, "indices": ["Nifty Smallcap 500"]},
    "SHAKTIPU": {"name": "Shakti Pumps (India) Ltd", "sector": "Capital Goods", "themes": ["Solar Pumps", "Water Pumps", "Clean Energy"], "mcap_cr": 5000, "indices": ["Nifty Smallcap 100"]},
    "OSWALPU": {"name": "Oswal Pumps Ltd", "sector": "Capital Goods", "themes": ["Pumps", "Agriculture", "Water Infrastructure"], "mcap_cr": 600, "indices": ["Nifty Smallcap 500"]},
    "EKC": {"name": "Everest Kanto Cylinder", "sector": "Capital Goods", "themes": ["CNG/Hydrogen Cylinders", "Clean Fuels", "Gas Infrastructure"], "mcap_cr": 1800, "indices": ["Nifty Smallcap 250"]},
    "INOXINDIA": {"name": "INOX India Ltd", "sector": "Capital Goods", "themes": ["Cryogenic Equipment", "LNG/Industrial Gas", "Gas Infrastructure"], "mcap_cr": 8000, "indices": ["Nifty Smallcap 100"]},
    "GNG": {"name": "GNG Electronics Ltd", "sector": "Capital Goods", "themes": ["EMS", "Power Electronics", "Electronics Manufacturing"], "mcap_cr": 1500, "indices": ["Nifty Smallcap 500"]},
    "JAINRESRC": {"name": "Jain Resource Recycling", "sector": "Metals", "themes": ["E-waste Recycling", "Circular Economy", "Sustainability"], "mcap_cr": 600, "indices": ["Nifty Smallcap 500"]},
    "CEINFO": {"name": "CE Info Systems Ltd", "sector": "IT", "themes": ["Mapping/GIS", "EV Navigation", "Technology"], "mcap_cr": 8000, "indices": ["Nifty Smallcap 100"]},
    "SANSERA": {"name": "Sansera Engineering Ltd", "sector": "Auto Ancillary", "themes": ["Precision Forgings", "Auto Components", "EV Ecosystem"], "mcap_cr": 6500, "indices": ["Nifty Smallcap 100"]},
    "SIEMENSENR": {"name": "Siemens Energy India Ltd", "sector": "Capital Goods", "themes": ["Power T&D", "Grid Equipment", "Grid Infrastructure"], "mcap_cr": 2000, "indices": ["Nifty Smallcap 250"]},
    "QUALITYPOW": {"name": "Quality Power Electrical", "sector": "Capital Goods", "themes": ["Power Equipment", "Grid Infrastructure", "Transformers"], "mcap_cr": 3000, "indices": ["Nifty Smallcap 250"]},
    "KECL": {"name": "KEC International Ltd", "sector": "Infrastructure", "themes": ["Transmission EPC", "Grid Infrastructure", "Power T&D"], "mcap_cr": 18000, "indices": ["Nifty Smallcap 100"]},
    "PONDYOXIDE": {"name": "Pondy Oxides & Chemicals", "sector": "Chemicals", "themes": ["Lead Recycling", "Circular Economy", "Battery Ecosystem"], "mcap_cr": 1200, "indices": ["Nifty Smallcap 250"]},
    "RTNPOWER": {"name": "RattanIndia Power Ltd", "sector": "Power", "themes": ["Power Generation", "Renewable Pivot", "Power Utility"], "mcap_cr": 3500, "indices": ["Nifty Smallcap 250"]},
    "ORIENTGR": {"name": "Orient Green Power", "sector": "Power", "themes": ["Wind Energy", "Biomass Power", "Clean Energy"], "mcap_cr": 1500, "indices": ["Nifty Smallcap 500"]},
    "GREENPOWER": {"name": "Green Power Co Ltd", "sector": "Power", "themes": ["Micro Hydro", "Clean Energy", "Sustainability"], "mcap_cr": 600, "indices": ["Nifty Smallcap 500"]},
    "CROMPTON": {"name": "Crompton Greaves CES", "sector": "Consumer Durables", "themes": ["Fans/Lighting", "Energy Efficiency", "Consumer"], "mcap_cr": 18000, "indices": ["Nifty Smallcap 100"]},
    "RRKABEL": {"name": "R R Kabel Ltd", "sector": "Capital Goods", "themes": ["Cables & Wires", "Grid Infrastructure", "Power T&D"], "mcap_cr": 16000, "indices": ["Nifty Smallcap 100"]},
    "VOLTAMP": {"name": "Voltamp Transformers", "sector": "Capital Goods", "themes": ["Transformers", "Grid Infrastructure", "Power T&D"], "mcap_cr": 7000, "indices": ["Nifty Smallcap 100"]},
    "WABAG": {"name": "VA Tech Wabag Ltd", "sector": "Capital Goods", "themes": ["Water Treatment", "Desalination", "Environment"], "mcap_cr": 9000, "indices": ["Nifty Smallcap 100"]},
    "DCXSYS": {"name": "DCX Systems Ltd", "sector": "Capital Goods", "themes": ["EMS", "Cables/Connectors", "Electronics Manufacturing"], "mcap_cr": 3500, "indices": ["Nifty Smallcap 250"]},
    "ESAF": {"name": "ESAF Small Finance Bank", "sector": "Finance", "themes": ["Green Microfinance", "Financial Inclusion", "Sustainability"], "mcap_cr": 4000, "indices": ["Nifty Smallcap 250"]},

    # ───────────── NIFTY MICROCAP 250 ─────────────
    "POCL": {"name": "Pondy Oxides & Chemicals", "sector": "Chemicals", "themes": ["Lead Recycling", "Battery Recycling", "Circular Economy"], "mcap_cr": 1200, "indices": ["Nifty Microcap 250"]},
    "ELECCAST": {"name": "Electrosteel Castings", "sector": "Metals", "themes": ["DI Pipes", "Water Infrastructure", "Infrastructure"], "mcap_cr": 3000, "indices": ["Nifty Microcap 250"]},
    "SADBHAV": {"name": "Sadbhav Engineering", "sector": "Infrastructure", "themes": ["Road EPC", "Infrastructure", "Construction"], "mcap_cr": 800, "indices": ["Nifty Microcap 250"]},
    "UNICHEMLAB": {"name": "Uniphos Enviro Ltd", "sector": "Chemicals", "themes": ["Environment Monitoring", "Sustainability", "Technology"], "mcap_cr": 600, "indices": ["Nifty Microcap 250"]},
    "RICOAUTO": {"name": "Rico Auto Industries", "sector": "Auto Ancillary", "themes": ["Castings", "EV Components", "EV Ecosystem"], "mcap_cr": 1500, "indices": ["Nifty Microcap 250"]},
    "TATVA": {"name": "Tatva Chintan Pharma", "sector": "Chemicals", "themes": ["Phase Transfer Catalyst", "Green Chemistry", "Sustainability"], "mcap_cr": 4000, "indices": ["Nifty Microcap 250"]},
    "WINDLAS": {"name": "Windlas Biotech Ltd", "sector": "Pharma", "themes": ["Green Manufacturing", "Sustainability", "Healthcare"], "mcap_cr": 1500, "indices": ["Nifty Microcap 250"]},
    "APLLTD": {"name": "Alkem Laboratories", "sector": "Pharma", "themes": ["Green Pharma", "Sustainability", "Healthcare"], "mcap_cr": 55000, "indices": ["Nifty Microcap 250"]},

    # ───────────── NIFTY LARGEMIDCAP 250 ─────────────
    "TATACOMM": {"name": "Tata Communications", "sector": "Telecom", "themes": ["Digital Infra", "IoT", "Smart Grid"], "mcap_cr": 36000, "indices": ["Nifty LargeMidcap 250"]},
    "LTTS": {"name": "L&T Technology Services", "sector": "IT", "themes": ["Engineering R&D", "EV Design", "EV Ecosystem"], "mcap_cr": 38000, "indices": ["Nifty LargeMidcap 250"]},
    "BLUESTARLTD": {"name": "Blue Star Ltd", "sector": "Capital Goods", "themes": ["HVAC", "Energy Efficiency", "Cooling"], "mcap_cr": 28000, "indices": ["Nifty LargeMidcap 250"]},
    "LALPATHLAB": {"name": "Dr Lal PathLabs", "sector": "Healthcare", "themes": ["Green Labs", "Sustainability", "Healthcare"], "mcap_cr": 18000, "indices": ["Nifty LargeMidcap 250"]},

    # ───────────── NIFTY MIDSMALLCAP 400 ─────────────
    "SOLARIND": {"name": "Solar Industries India", "sector": "Chemicals", "themes": ["Explosives", "Defense/Mining", "Manufacturing"], "mcap_cr": 62000, "indices": ["Nifty MidSmallcap 400"]},
    "DATAPATTNS": {"name": "Data Patterns India", "sector": "Capital Goods", "themes": ["Defense Electronics", "EMS", "Electronics Manufacturing"], "mcap_cr": 10000, "indices": ["Nifty MidSmallcap 400"]},
    "CENTRALBK": {"name": "Central Bank of India", "sector": "Finance", "themes": ["Green Lending", "PSU Banking", "Financial Inclusion"], "mcap_cr": 6000, "indices": ["Nifty MidSmallcap 400"]},
    "ANURAS": {"name": "Anuras Ltd", "sector": "Capital Goods", "themes": ["Power Equipment", "Grid Components", "Manufacturing"], "mcap_cr": 2500, "indices": ["Nifty MidSmallcap 400"]},
    "GIPCL": {"name": "Gujarat Industries Power", "sector": "Power", "themes": ["Wind/Solar IPP", "Clean Energy", "PSU"], "mcap_cr": 4500, "indices": ["Nifty MidSmallcap 400"]},

    # ───────────── NIFTY 500 MULTICAP 50:25:25 ─────────────
    "ADANIPOWER": {"name": "Adani Power Ltd", "sector": "Power", "themes": ["Power Generation", "Solar Pivot", "Power Utility"], "mcap_cr": 85000, "indices": ["Nifty500 Multicap"]},
    "VEDL": {"name": "Vedanta Ltd", "sector": "Metals", "themes": ["Mining", "Zinc/Aluminium", "Commodities"], "mcap_cr": 45000, "indices": ["Nifty500 Multicap"]},
    "HINDALCO": {"name": "Hindalco Industries", "sector": "Metals", "themes": ["Aluminium", "Copper", "EV Battery Supply Chain"], "mcap_cr": 120000, "indices": ["Nifty500 Multicap"]},
    "TATASTEEL": {"name": "Tata Steel Ltd", "sector": "Metals", "themes": ["Green Steel", "Hydrogen Steelmaking", "Sustainability"], "mcap_cr": 180000, "indices": ["Nifty500 Multicap"]},
    "JSWSTEEL": {"name": "JSW Steel Ltd", "sector": "Metals", "themes": ["Green Steel", "Solar Steel", "Sustainability"], "mcap_cr": 190000, "indices": ["Nifty500 Multicap"]},

    # ───────────── NIFTY INDIA FPI 150 ─────────────
    "BHARTIARTL": {"name": "Bharti Airtel Ltd", "sector": "Telecom", "themes": ["Green Telecom", "Data Centers", "Infrastructure"], "mcap_cr": 900000, "indices": ["Nifty India FPI 150"]},
    "LTIM": {"name": "LTIMindtree Ltd", "sector": "IT", "themes": ["Digital Infra", "Smart Grid Software", "Technology"], "mcap_cr": 150000, "indices": ["Nifty India FPI 150"]},
    "BALKRISIND": {"name": "Balkrishna Industries", "sector": "Auto Ancillary", "themes": ["Off-Highway Tyres", "Agriculture", "Manufacturing"], "mcap_cr": 45000, "indices": ["Nifty India FPI 150"]},
    "GODREJCP": {"name": "Godrej Consumer Products", "sector": "FMCG", "themes": ["Sustainability", "Green Products", "Consumer"], "mcap_cr": 95000, "indices": ["Nifty India FPI 150"]},

    # ───────────── ADDITIONAL MIDCAP / SMALLCAP ENERGY & INFRA ─────────────
    "NLC": {"name": "NLC India Ltd", "sector": "Power", "themes": ["Lignite/Solar", "Clean Energy", "PSU"], "mcap_cr": 25000, "indices": ["Nifty Midcap 100"]},
    "HUDCO": {"name": "HUDCO Ltd", "sector": "Finance", "themes": ["Green Housing", "Infrastructure Finance", "PSU"], "mcap_cr": 35000, "indices": ["Nifty Midcap 100"]},
    "IPCALAB": {"name": "IPCA Laboratories", "sector": "Pharma", "themes": ["Green Manufacturing", "Sustainability", "Healthcare"], "mcap_cr": 28000, "indices": ["Nifty Midcap 100"]},
    "ENGINERSIN": {"name": "Engineers India Ltd", "sector": "Capital Goods", "themes": ["Oil-to-Green Consulting", "Energy Transition", "PSU"], "mcap_cr": 12000, "indices": ["Nifty Midcap 150"]},
    "BAJAJELEC": {"name": "Bajaj Electricals Ltd", "sector": "Consumer Durables", "themes": ["Lighting/LED", "Energy Efficiency", "Consumer"], "mcap_cr": 10000, "indices": ["Nifty Midcap 150"]},
    "JYOTI": {"name": "Jyoti CNC Automation", "sector": "Capital Goods", "themes": ["CNC Machines", "Precision Manufacturing", "Manufacturing"], "mcap_cr": 12000, "indices": ["Nifty Midcap 100"]},
    "TARSONS": {"name": "Tarsons Products Ltd", "sector": "Healthcare", "themes": ["Lab Plastics", "Green Labs", "Manufacturing"], "mcap_cr": 4000, "indices": ["Nifty Midcap 150"]},
    "MNFL": {"name": "Meghmani Finechem", "sector": "Chemicals", "themes": ["Chlor-Alkali", "Green Chemistry", "Manufacturing"], "mcap_cr": 3500, "indices": ["Nifty Midcap 150"]},
    "NUVOCO": {"name": "Nuvoco Vistas Corp", "sector": "Cement", "themes": ["Green Cement", "Sustainability", "Construction"], "mcap_cr": 10000, "indices": ["Nifty Midcap 150"]},
    "BIRLACABLE": {"name": "Birla Cable Ltd", "sector": "Capital Goods", "themes": ["Optical Fibre", "Cables", "Grid Infrastructure"], "mcap_cr": 2000, "indices": ["Nifty Smallcap 250"]},
    "MAZDOCK": {"name": "Mazagon Dock Shipbuilders", "sector": "Capital Goods", "themes": ["Defense", "Shipbuilding", "Manufacturing"], "mcap_cr": 60000, "indices": ["Nifty Midcap 50"]},
    "COCHINSHIP": {"name": "Cochin Shipyard Ltd", "sector": "Capital Goods", "themes": ["Green Shipping", "Shipbuilding", "PSU"], "mcap_cr": 24000, "indices": ["Nifty Midcap 100"]},
    "BEL": {"name": "Bharat Electronics", "sector": "Capital Goods", "themes": ["Defense Electronics", "Solar Inverters", "PSU"], "mcap_cr": 180000, "indices": ["Nifty 100"]},
    "HAL": {"name": "Hindustan Aeronautics", "sector": "Capital Goods", "themes": ["Aerospace", "Defense", "PSU"], "mcap_cr": 250000, "indices": ["Nifty 100"]},
    "TITAGARH": {"name": "Titagarh Rail Systems", "sector": "Capital Goods", "themes": ["Rail Coaches", "Metro", "Infrastructure"], "mcap_cr": 12000, "indices": ["Nifty Smallcap 100"]},
    "RITES": {"name": "RITES Ltd", "sector": "Infrastructure", "themes": ["Railway Consulting", "Infrastructure", "PSU"], "mcap_cr": 12000, "indices": ["Nifty Smallcap 100"]},
    "IDEAFORGE": {"name": "ideaForge Technology", "sector": "Capital Goods", "themes": ["Drones", "Defense Tech", "Technology"], "mcap_cr": 3000, "indices": ["Nifty Smallcap 250"]},
    "AMBER": {"name": "Amber Enterprises India", "sector": "Consumer Durables", "themes": ["AC Components", "Energy Efficiency", "Manufacturing"], "mcap_cr": 14000, "indices": ["Nifty Midcap 100"]},
    "CENTURYPLY": {"name": "Century Plyboards", "sector": "Building Materials", "themes": ["Sustainable Plywood", "Green Building", "Manufacturing"], "mcap_cr": 9000, "indices": ["Nifty Midcap 150"]},
    "CLEAN": {"name": "Clean Science & Technology", "sector": "Chemicals", "themes": ["Green Chemistry", "Catalysts", "Sustainability"], "mcap_cr": 15000, "indices": ["Nifty Midcap 100"]},
    "EPIGRAL": {"name": "Epigral Ltd", "sector": "Chemicals", "themes": ["Chlor-Alkali", "Hydrogen", "Green Chemistry"], "mcap_cr": 5000, "indices": ["Nifty Smallcap 100"]},
    "FLUOROCHEM": {"name": "Gujarat Fluorochemicals", "sector": "Chemicals", "themes": ["Fluoropolymers", "Battery Materials", "Battery Ecosystem"], "mcap_cr": 22000, "indices": ["Nifty Midcap 100"]},
    "AARTI": {"name": "Aarti Industries Ltd", "sector": "Chemicals", "themes": ["Specialty Chemicals", "Green Chemistry", "Manufacturing"], "mcap_cr": 18000, "indices": ["Nifty Midcap 100"]},
    "NIACL": {"name": "New India Assurance", "sector": "Insurance", "themes": ["Green Insurance", "PSU", "Financial Services"], "mcap_cr": 12000, "indices": ["Nifty Midcap 150"]},
    "AFFLE": {"name": "Affle India Ltd", "sector": "IT", "themes": ["MarTech", "Digital Platform", "Technology"], "mcap_cr": 16000, "indices": ["Nifty Midcap 100"]},
    "NAZARA": {"name": "Nazara Technologies", "sector": "IT", "themes": ["Gaming", "Digital Platform", "Technology"], "mcap_cr": 6000, "indices": ["Nifty Smallcap 100"]},
    "ROUTE": {"name": "Route Mobile Ltd", "sector": "IT", "themes": ["CPaaS", "Digital Infra", "Technology"], "mcap_cr": 6000, "indices": ["Nifty Smallcap 100"]},
}


def _load_portfolio_data() -> dict:
    """Load the green energy portfolio JSON dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Portfolio data not found at {DATA_PATH}")
    with open(DATA_PATH, "r") as f:
        return json.load(f)


def _compute_performance_metrics(index_data: list[dict]) -> dict:
    """Compute portfolio vs benchmark performance metrics."""
    first = index_data[0]
    last = index_data[-1]
    n_days = len(index_data)
    years = n_days / 252

    ge_start, ge_end = first["green_energy"], last["green_energy"]
    ns_start, ns_end = first["nifty_sc100"], last["nifty_sc100"]

    ge_return = (ge_end / ge_start - 1) * 100
    ns_return = (ns_end / ns_start - 1) * 100

    ge_cagr = (math.pow(ge_end / ge_start, 1 / years) - 1) * 100 if years > 0 else 0
    ns_cagr = (math.pow(ns_end / ns_start, 1 / years) - 1) * 100 if years > 0 else 0

    # Max drawdown
    ge_peak = ge_start
    ge_max_dd = 0
    for d in index_data:
        ge_peak = max(ge_peak, d["green_energy"])
        dd = (ge_peak - d["green_energy"]) / ge_peak * 100
        ge_max_dd = max(ge_max_dd, dd)

    ns_peak = ns_start
    ns_max_dd = 0
    for d in index_data:
        ns_peak = max(ns_peak, d["nifty_sc100"])
        dd = (ns_peak - d["nifty_sc100"]) / ns_peak * 100
        ns_max_dd = max(ns_max_dd, dd)

    # Daily returns for volatility/Sharpe
    ge_daily_returns = []
    ns_daily_returns = []
    for i in range(1, len(index_data)):
        ge_daily_returns.append(index_data[i]["green_energy"] / index_data[i - 1]["green_energy"] - 1)
        ns_daily_returns.append(index_data[i]["nifty_sc100"] / index_data[i - 1]["nifty_sc100"] - 1)

    ge_vol = _std(ge_daily_returns) * math.sqrt(252) * 100
    ns_vol = _std(ns_daily_returns) * math.sqrt(252) * 100
    rf_annual = 6.0  # approximate risk-free rate India

    ge_sharpe = (ge_cagr - rf_annual) / ge_vol if ge_vol > 0 else 0
    ns_sharpe = (ns_cagr - rf_annual) / ns_vol if ns_vol > 0 else 0

    # Rebalance count
    rebalances = sum(1 for d in index_data if d["rebalance"])

    return {
        "period": {"start": first["date"], "end": last["date"], "trading_days": n_days, "years": round(years, 2)},
        "green_energy": {
            "total_return_pct": round(ge_return, 2),
            "cagr_pct": round(ge_cagr, 2),
            "annualized_vol_pct": round(ge_vol, 2),
            "sharpe_ratio": round(ge_sharpe, 2),
            "max_drawdown_pct": round(ge_max_dd, 2),
            "final_value": round(ge_end, 2),
        },
        "benchmark": {
            "total_return_pct": round(ns_return, 2),
            "cagr_pct": round(ns_cagr, 2),
            "annualized_vol_pct": round(ns_vol, 2),
            "sharpe_ratio": round(ns_sharpe, 2),
            "max_drawdown_pct": round(ns_max_dd, 2),
            "final_value": round(ns_end, 2),
        },
        "alpha_pct": round(ge_cagr - ns_cagr, 2),
        "total_rebalances": rebalances,
    }


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def _analyze_constituents(constituents: list[dict]) -> dict:
    """Analyze stock selection patterns from constituent data."""
    stock_appearances = defaultdict(int)
    stock_total_weight = defaultdict(float)
    stock_max_weight = defaultdict(float)
    stock_date_ranges = defaultdict(list)

    for c in constituents:
        s = c["stock"]
        w = c["weight"]
        stock_appearances[s] += 1
        stock_total_weight[s] += w
        stock_max_weight[s] = max(stock_max_weight[s], w)
        stock_date_ranges[s].append(c["date_range"])

    # Get unique date ranges (periods)
    all_date_ranges = sorted(set(c["date_range"] for c in constituents))
    total_periods = len(all_date_ranges)

    # Build stock details
    stock_details = []
    for stock in sorted(stock_appearances.keys(), key=lambda s: -stock_total_weight[s]):
        themes = STOCK_THEMES.get(stock, ["Uncategorized"])
        avg_weight = stock_total_weight[stock] / stock_appearances[stock]
        persistence = stock_appearances[stock] / total_periods * 100

        # Check if in latest period
        latest_period = all_date_ranges[-1]
        in_latest_period = any(c["date_range"] == latest_period and c["stock"] == stock for c in constituents)
        latest_weight = next(
            (c["weight"] for c in constituents if c["date_range"] == latest_period and c["stock"] == stock), 0
        )

        stock_details.append({
            "stock": stock,
            "periods": stock_appearances[stock],
            "total_periods": total_periods,
            "persistence_pct": round(persistence, 1),
            "avg_weight_pct": round(avg_weight * 100, 2),
            "max_weight_pct": round(stock_max_weight[stock] * 100, 2),
            "cumulative_weight": round(stock_total_weight[stock], 3),
            "themes": themes,
            "currently_held": in_latest_period,
            "current_weight_pct": round(latest_weight * 100, 2) if in_latest_period else None,
        })

    # Theme frequency analysis
    theme_freq = defaultdict(int)
    theme_weight = defaultdict(float)
    for c in constituents:
        stock_themes = STOCK_THEMES.get(c["stock"], [])
        for theme in stock_themes:
            theme_freq[theme] += 1
            theme_weight[theme] += c["weight"]

    theme_analysis = [
        {"theme": theme, "appearances": count, "total_weight": round(theme_weight[theme], 3)}
        for theme, count in sorted(theme_freq.items(), key=lambda x: -x[1])
    ]

    return {
        "total_unique_stocks": len(stock_appearances),
        "total_periods": total_periods,
        "stocks": stock_details,
        "themes": theme_analysis,
    }


def _compute_period_performance(index_data: list[dict]) -> list[dict]:
    """Compute performance between each rebalance period."""
    rebalance_indices = [0]  # start
    for i, d in enumerate(index_data):
        if d["rebalance"]:
            rebalance_indices.append(i)
    rebalance_indices.append(len(index_data) - 1)  # end

    period_perf = []
    for j in range(1, len(rebalance_indices)):
        start_idx = rebalance_indices[j - 1]
        end_idx = rebalance_indices[j]
        if end_idx <= start_idx:
            continue

        ge_start = index_data[start_idx]["green_energy"]
        ge_end = index_data[end_idx]["green_energy"]
        ns_start = index_data[start_idx]["nifty_sc100"]
        ns_end = index_data[end_idx]["nifty_sc100"]

        ge_ret = (ge_end / ge_start - 1) * 100
        ns_ret = (ns_end / ns_start - 1) * 100

        period_perf.append({
            "start_date": index_data[start_idx]["date"],
            "end_date": index_data[end_idx]["date"],
            "days": end_idx - start_idx,
            "ge_return_pct": round(ge_ret, 2),
            "benchmark_return_pct": round(ns_ret, 2),
            "alpha_pct": round(ge_ret - ns_ret, 2),
        })

    return period_perf


def _generate_nifty_predictions(constituent_analysis: dict) -> list[dict]:
    """
    Generate buy predictions for Nifty universe stocks based on learned
    factor patterns from the Green Energy Theme portfolio.

    The scoring model captures:
    1. Theme alignment (30%): How many portfolio themes the stock matches
    2. Demonstrated alpha (25%): If the stock was in the portfolio, its persistence
    3. Sector momentum (20%): Green energy, grid infra, EV ecosystem rotation
    4. Portfolio conviction (15%): Weight allocation patterns in the portfolio
    5. Recency factor (10%): Whether the stock is in the current portfolio
    """
    # Build theme importance weights from portfolio data
    theme_scores = {}
    for t in constituent_analysis["themes"]:
        theme_scores[t["theme"]] = t["total_weight"]

    max_theme_weight = max(theme_scores.values()) if theme_scores else 1

    # Build stock persistence map (% of periods a stock was in the portfolio)
    stock_persistence = {}
    stock_avg_weight = {}
    stock_currently_held = {}
    for s in constituent_analysis["stocks"]:
        stock_persistence[s["stock"]] = s["persistence_pct"]
        stock_avg_weight[s["stock"]] = s["avg_weight_pct"]
        stock_currently_held[s["stock"]] = s["currently_held"]

    predictions = []
    for ticker, info in NIFTY_PREDICTION_UNIVERSE.items():
        name = info["name"]
        themes = info["themes"]
        sector = info["sector"]

        # 1. Theme alignment score (0-100)
        theme_alignment = 0
        matched_themes = []
        for theme in themes:
            if theme in theme_scores:
                theme_alignment += (theme_scores[theme] / max_theme_weight) * 100
                matched_themes.append(theme)
        theme_alignment = min(theme_alignment / max(len(themes), 1), 100)

        # 2. Portfolio proven score (0-100)
        portfolio_score = 0
        if name in stock_persistence:
            portfolio_score = stock_persistence[name]

        # 3. Sector momentum (0-100) — weighted by sector's representation
        sector_theme_map = {
            "Power": ["Clean Energy", "Power Utility", "Solar EPC"],
            "Capital Goods": ["Power Equipment", "Grid Infrastructure", "Manufacturing"],
            "Finance": ["Infrastructure Finance", "Green Finance"],
            "IT": ["EV Software", "Auto Tech", "Technology"],
            "Auto Ancillary": ["EV Ecosystem", "Battery Ecosystem"],
            "Automobiles": ["EV OEM", "EV Ecosystem"],
            "Chemicals": ["Battery Ecosystem", "Specialty Chemicals", "Green Chemistry"],
            "Infrastructure": ["Grid Infrastructure", "Power T&D"],
            "Metals": ["Circular Economy", "Sustainability"],
            "Textiles": ["Circular Economy", "Sustainability"],
            "Conglomerate": ["Clean Energy", "Manufacturing"],
            "Consumer Durables": ["Energy Efficiency", "Manufacturing"],
            "Consumer Electronics": ["Electronics Manufacturing", "Manufacturing"],
            "Telecom": ["Infrastructure", "Technology"],
            "FMCG": ["Sustainability"],
            "Cement": ["Sustainability", "Infrastructure"],
            "Building Materials": ["Sustainability", "Manufacturing"],
            "Healthcare": ["Sustainability", "Manufacturing"],
            "Pharma": ["Sustainability", "Green Chemistry"],
            "Insurance": ["Sustainability"],
            "Retail": ["Sustainability"],
        }
        sector_themes = sector_theme_map.get(sector, [])
        sector_score = 0
        for st in sector_themes:
            if st in theme_scores:
                sector_score += (theme_scores[st] / max_theme_weight) * 50
        sector_score = min(sector_score, 100)

        # 4. Conviction score from weight patterns (0-100)
        conviction = 0
        if name in stock_avg_weight:
            conviction = min(stock_avg_weight[name] / 12.5 * 100, 100)

        # 5. Recency factor (0-100)
        recency = 100 if stock_currently_held.get(name, False) else 0

        # ── Composite score ──
        composite = (
            theme_alignment * 0.30
            + portfolio_score * 0.25
            + sector_score * 0.20
            + conviction * 0.15
            + recency * 0.10
        )

        # Decision signal
        if composite >= 75:
            signal = "STRONG_BUY"
        elif composite >= 55:
            signal = "BUY"
        elif composite >= 35:
            signal = "HOLD"
        elif composite >= 20:
            signal = "SELL"
        else:
            signal = "STRONG_SELL"

        # Reasoning
        reasons = []
        if matched_themes:
            reasons.append(f"Aligned with portfolio themes: {', '.join(matched_themes[:3])}")
        if portfolio_score > 50:
            reasons.append(f"High portfolio persistence ({portfolio_score:.0f}% of periods)")
        elif portfolio_score > 0:
            reasons.append(f"Present in portfolio ({portfolio_score:.0f}% of periods)")
        if stock_currently_held.get(name, False):
            reasons.append("Currently held in latest portfolio")
        if sector_score > 50:
            reasons.append(f"Strong sector momentum in {sector}")
        if conviction > 60:
            reasons.append(f"High conviction: avg weight {stock_avg_weight.get(name, 0):.1f}%")

        predictions.append({
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "themes": themes,
            "indices": info.get("indices", []),
            "signal": signal,
            "composite_score": round(composite, 1),
            "factor_scores": {
                "theme_alignment": round(theme_alignment, 1),
                "portfolio_proven": round(portfolio_score, 1),
                "sector_momentum": round(sector_score, 1),
                "conviction": round(conviction, 1),
                "recency": round(recency, 1),
            },
            "matched_themes": matched_themes,
            "reasoning": " | ".join(reasons) if reasons else "Limited theme overlap",
        })

    # Sort by composite score descending
    predictions.sort(key=lambda x: -x["composite_score"])

    return predictions


def _build_index_chart_data(index_data: list[dict]) -> list[dict]:
    """Build chart-friendly data (sampled for frontend)."""
    # Sample every 5th data point for reasonable chart size
    step = max(1, len(index_data) // 250)
    chart_data = []
    for i in range(0, len(index_data), step):
        d = index_data[i]
        chart_data.append({
            "date": d["date"],
            "ge": round(d["green_energy"], 2),
            "benchmark": round(d["nifty_sc100"], 2),
            "rebalance": d["rebalance"],
        })
    # Always include last point
    if chart_data[-1]["date"] != index_data[-1]["date"]:
        d = index_data[-1]
        chart_data.append({
            "date": d["date"],
            "ge": round(d["green_energy"], 2),
            "benchmark": round(d["nifty_sc100"], 2),
            "rebalance": d["rebalance"],
        })
    return chart_data


def _learned_factors_summary(constituent_analysis: dict) -> dict:
    """Summarize the learned factors from portfolio analysis."""
    top_themes = constituent_analysis["themes"][:10]
    top_stocks = constituent_analysis["stocks"][:10]

    # Identify factor categories
    factor_categories = {
        "Clean Energy": {"themes": ["Clean Energy", "Solar Value Chain", "Wind Energy", "Solar EPC", "Utility Scale Solar", "Hydro/Wind"], "weight": 0},
        "Grid Infrastructure": {"themes": ["Grid Infrastructure", "Power T&D", "Grid Automation", "Transformers", "Transmission Towers", "Smart Meters"], "weight": 0},
        "EV Ecosystem": {"themes": ["EV Ecosystem", "EV Software", "EV OEM", "EV Drivetrain", "Battery Ecosystem", "Energy Storage"], "weight": 0},
        "Circular Economy": {"themes": ["Circular Economy", "Sustainability", "Recycled Polyester"], "weight": 0},
        "Power Equipment": {"themes": ["Power Equipment", "Steam Turbines", "Generators", "Motors", "Electrical Components"], "weight": 0},
        "Manufacturing": {"themes": ["Manufacturing", "Electronics Manufacturing", "Precision Engineering"], "weight": 0},
    }

    theme_weight_map = {t["theme"]: t["total_weight"] for t in constituent_analysis["themes"]}
    for cat, info in factor_categories.items():
        for theme in info["themes"]:
            info["weight"] += theme_weight_map.get(theme, 0)
        info["weight"] = round(info["weight"], 3)

    factors_ranked = sorted(factor_categories.items(), key=lambda x: -x[1]["weight"])

    return {
        "top_factor_categories": [
            {"name": name, "weight": data["weight"], "themes": data["themes"]}
            for name, data in factors_ranked
        ],
        "top_portfolio_themes": [{"theme": t["theme"], "weight": round(t["total_weight"], 3)} for t in top_themes],
        "most_persistent_stocks": [
            {"stock": s["stock"], "persistence_pct": s["persistence_pct"], "avg_weight_pct": s["avg_weight_pct"]}
            for s in top_stocks
        ],
        "selection_characteristics": [
            "Small-mid cap green energy and power equipment companies",
            "High persistence: top stocks held across 70-100% of rebalance periods",
            "Concentrated positions: 5-12.5% weights in high-conviction names",
            "Sector rotation between clean energy, grid infra, and EV ecosystem",
            "Circular economy / sustainability theme as diversifier",
            "Rebalance approximately every 4-8 weeks with tactical adjustments",
            "Solar value chain (glass, cells, modules, EPC) is the largest sub-theme",
            "Grid infrastructure plays sustained across all market regimes",
            "Liquid ETF position (2-5%) for tactical cash management",
            "Active stance: 36 rebalances over 5 years — momentum + fundamental driven",
        ],
    }


def run_portfolio_analysis() -> dict:
    """
    Main entry point: Run full portfolio analysis and generate predictions.
    Returns a comprehensive JSON-serializable result.
    """
    data = _load_portfolio_data()
    index_data = data["index_values"]
    constituents = data["constituents"]

    performance = _compute_performance_metrics(index_data)
    constituent_analysis = _analyze_constituents(constituents)
    period_performance = _compute_period_performance(index_data)
    chart_data = _build_index_chart_data(index_data)
    learned_factors = _learned_factors_summary(constituent_analysis)
    predictions = _generate_nifty_predictions(constituent_analysis)

    return {
        "performance": performance,
        "constituent_analysis": {
            "total_unique_stocks": constituent_analysis["total_unique_stocks"],
            "total_periods": constituent_analysis["total_periods"],
            "top_stocks": constituent_analysis["stocks"][:20],
            "themes": constituent_analysis["themes"][:15],
        },
        "period_performance": period_performance,
        "chart_data": chart_data,
        "learned_factors": learned_factors,
        "predictions": predictions,
    }

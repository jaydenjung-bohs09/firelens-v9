# FireLens V3 — NASA Space Apps 2026

Upgrades: file-coverage audit, compare shared detection dates, day/night comparisons, UTC detection-time distribution, Park Fire proximity case study, original grid sensitivity analysis. The original baseline should reproduce (2 km, 22.84%), (5 km, 22.55%), (10 km, 20.22%) with July 2024 archive files.

## Google Colab quickstart
Upload `FireLens_V3.zip` into your existing Colab session. Extract the files under `/content/firelens_v3`, install packages from `/content/firelens_v3/requirements.txt`, stop your old Streamlit server, and launch `/content/firelens_v3/app.py`. Then open port 8501 with `output.serve_kernel_port_as_iframe(8501, height=800)`. Upload original MODIS and VIIRS CSVs again in the app: Colab Streamlit refreshes do not persist uploaded files across a fresh app instance.

Note: an absence of FIRMS fire hotspot records is not proof of satellite non-observation. Shared-detection-date restriction is a sensitivity check (selection bias), not a validated correction for observation opportunity.

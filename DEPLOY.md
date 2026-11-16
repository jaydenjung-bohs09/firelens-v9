# Deploy MODVIIR to Streamlit Community Cloud

1. Extract `FireLens_V9_Global_Candidate.zip` locally. Open the extracted `FireLens_V9_Global_Candidate` folder.
2. Create a public GitHub repository and upload **the contents of this folder** (so `app.py`, `requirements.txt` and `.streamlit/` are at the repository root). Do not place a ZIP file alone in GitHub.
3. Open https://share.streamlit.io/ and select **Create app** from the public repository, main file path `app.py`.
4. Wait for dependencies to install and open the deployed link. Verify Explore History, Investigate Evidence (choose July 24), Model Replay (choose September 3, 2025), and Methods & Sources. Inspect plots at laptop width and mobile width.
5. Test the public URL in a private/incognito browser window before submitting it to NASA. The judges should not need a Google account or data uploads.
6. Add the app URL, GitHub source URL, and presentation/video link to the official NASA project page. Only claim public deployment after checking the URLs.

## Essential launch checks

```bash
python tests/test_integrity.py
python tests/smoke_without_streamlit.py
streamlit run app.py
```

The first two commands do not validate visual rendering, browser interactions, or remote deployment. Run the third command after installing `requirements.txt` and check in a browser. A separately bundled V8 app remains available at `legacy_v8/app.py`, untouched.

## Files with archival provenance

- MODIS `fire_archive_M-C61_850130.csv`; VIIRS S-NPP `fire_archive_SV-C2_850131.csv` — real California July 2024 observations.
- The model source inventory names the 2022–2025 inputs as MODIS `fire_archive_M-C61_850338.csv` and VIIRS S-NPP `fire_archive_SV-C2_850339.csv`. The full original multi-year archives are not bundled here.

## Troubleshooting

- If the app fails on import, inspect the Streamlit Cloud logs and ensure the project root has `requirements.txt`.
- If charts look blank, confirm `data/predict/test_predictions.npz` and other files were uploaded, rather than ignored as oversized assets. The total project is small enough for a normal repository but GitHub may reject individual files above its limit.
- If the numerical checks fail, do not show the model claims until the artifact mismatch is resolved.
- For a reliable fallback, submit the seven-slide PDF and repository link even if hosted UI availability is interrupted; do not substitute synthetic imagery as real NASA observations.

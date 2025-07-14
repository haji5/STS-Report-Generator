# Report Generator Documentation

## 1. Overview

This program generates a comprehensive PowerPoint presentation (`.pptx`) with data visualizations based on tourism data. It analyzes visitor numbers, trips, overnight stays, origins, and demographic segments for various destinations (DMOs), and supports year-over-year comparisons.

## 2. Getting Started

1. **Download the Program**  
   Go to the GitHub releases page:  
   👉 [https://github.com/haji5/STS-Report-Generator/releases/latest](https://github.com/haji5/STS-Report-Generator/releases/latest)  
   Download the latest `ReportGenerator.exe` file.

2. **Run the Program**  
   Simply double-click the downloaded `ReportGenerator.exe` to launch the application. No installation or setup is required.

> ⚠️ On first launch, Windows Defender or your antivirus may warn about running an unknown app. Click "More info" > "Run anyway" if you're confident in the source.

## 3. Data File Requirements

Your data files (`VTN.csv`, `DMO.csv`) can be located anywhere on your computer; you'll select them through the application's file browser.

### File Format Requirements

- This program **only works with `.csv` files**.
- If your data is in Excel format (`.xlsx`, `.xls`), save it as `.csv` first.

### Required Files

**`VTN.csv`** – Visitor Travel Network data  
Required columns:
- `date` – Format: `YYYY-MM-DD`
- `DMO` – Numeric destination code
- `Visitor` – Number of visitors
- `Trips` – Number of trips
- `Nights` – Number of overnight stays
- `PERIOD_TYPE` – Must include `Months` and/or `Quarters`  
  - Used to determine chart type and time resolution  
  - Quarterly data is used for Top 5 Origins; monthly data is used for trends
- `ORIGIN_PRCDCSD` – Numeric origin location code
- `PRIZM_CODE` – Numeric demographic segment code

**`DMO.csv`** – Destination Marketing Organization mapping file  
Required columns:
- `DEST_CODE` – Numeric code corresponding to `DMO` in `VTN.csv`
- `DEST_NAME` – Full name of the destination
- `DMO` – Display name for the destination used in the final report  
  - This can be the same as `DEST_NAME` or a custom label

## 4. Using the Program

1. **Launch the App**  
   Double-click `ReportGenerator.exe`.

2. **Load Data**  
   - Click **Browse VTN** to select `VTN.csv`  
   - Click **Browse DMO** to select `DMO.csv`

3. **Choose Report Settings**  
   - Select the **primary year**
   - Optionally select a **comparison year**
   - Choose the **destination province**
   - (Optional) Enter a custom **color palette** using hex codes (e.g., `#FF5733,#33C4FF,#9C27B0`)

4. **Set Output Folder**  
   - Use **Browse Output Folder** to choose where to save the report  
   - If no folder is selected, an `output` folder will be created next to the `.exe`

5. **Generate Report**  
   - Click **Generate Report**  
   - A progress bar will show status; generation may take a few minutes  
   - When finished, a success message will confirm where the report was saved

## 5. Output

The output folder will contain:
- 📊 PNG chart images
- 🖼️ A PowerPoint presentation: `Report_YYYY.pptx`

This presentation includes all generated visuals, ready for review or presentation.

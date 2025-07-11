# Report Generator Documentation

## 1. Overview

This program generates a comprehensive PowerPoint presentation (`.pptx`) with data visualizations based on tourism data. It analyzes visitor numbers, trips, overnight stays, origins, and demographic segments for various destinations (DMOs), and supports year-over-year comparisons.

## 2. Getting Started: Installation & Setup

To get started, download the program from the official release page.

1.  **Download the Program**:
    * Go to the GitHub releases page: [https://github.com/haji5/STS-Report-Generator/releases/latest](https://github.com/haji5/STS-Report-Generator/releases/latest)
    * Download the `ReportGenerator-vX.X.zip` file.

2.  **Unzip the Folder**:
    * Extract the contents of the ZIP file to a location on your computer (e.g., your Desktop or Documents folder).

3.  **Verify the Files**:
    * Inside the unzipped folder, you should find the following structure. It is crucial that `ReportGenerator.exe` and the `assets` folder remain in the same directory.

    ```
    /ReportGenerator/
    |-- ReportGenerator.exe      <-- The program itself
    |-- /assets/
    |   |-- Geography.xlsx       <-- Provided file for mapping origin locations
    |   |-- Profile.csv          <-- Provided file for mapping PRIZM segments
    |   |-- STS+Logo.png         <-- The application icon
    ```

## 3. Data File Requirements

Your data files (`VTN.csv`, `DMO.csv`) can be located anywhere on your computer; you will select them using the program's interface.

* **`VTN.csv`**: The Visitor Travel Network data file. This is the primary input file containing detailed visitor data. It must contain the following columns:
    * `date`: The date of the visit (format: `YYYY-MM-DD`).
    * `DMO`: A numeric code for the destination, which corresponds to a `DEST_CODE` in the `DMO.csv` file.
    * `Visitor`: The number of visitors.
    * `Trips`: The number of trips taken.
    * `Nights`: The number of overnight stays.
    * `PERIOD_TYPE`: The time period for the data row (e.g., `Months`, `Quarters`).
        * For monthly trend line charts, the program requires rows where this is set to `Months`.
        * For quarterly analysis (like the Top 5 Origins charts), the program will prioritize rows where this is set to `Quarters`. If no quarterly data is found, it will automatically aggregate the monthly data.
    * `ORIGIN_PRCDCSD`: A numeric code representing the visitor's origin location.
    * `PRIZM_CODE`: A numeric code representing the visitor's demographic (PRIZM) segment.

* **`DMO.csv`**: The Destination Marketing Organization mapping file. This file links the numeric DMO codes from `VTN.csv` to their names. It must contain the following columns:
    * `DEST_CODE`: The unique numeric code for the destination.
    * `DEST_NAME`: The full name of the destination.
    * `DMO`: The display name for the destination (can be the same as `DEST_NAME`).

## 4. How to Use the Program

1.  **Launch the Application**: Double-click the `ReportGenerator.exe` file from the folder you unzipped. The main application window will appear.

2.  **Select Input Files**:
    * Click **Browse VTN** to select your `VTN.csv` data file from your computer.
    * Click **Browse DMO** to select your `DMO.csv` mapping file from your computer.

3.  **Choose Report Options**:
    * **Select Year**: Once the files are loaded, this dropdown will populate with all available years from your `VTN.csv` file. Select the primary year for your report.
    * **Select Comparison Year (optional)**: Select a year to compare against the primary year. This dropdown will only show years *before* the selected primary year.
    * **Select Destination Province**: Choose the province where the destinations are located. This is used for in-province vs. out-of-province analysis.
    * **Custom Color Palette (optional)**: You can enter a custom color scheme for the charts. Use comma-separated hex codes (e.g., `#FF5733, #33C4FF, #9C27B0`). A validation message will confirm if the format is correct.

4.  **Select Output Folder (Recommended)**:
    * Click **Browse Output Folder** to choose where the final report and chart images will be saved.
    * If you don't select a folder, a new folder named `output` will be created inside the `ReportGenerator` directory.

5.  **Generate the Report**:
    * Click the **Generate Report** button.
    * The process will begin, and a progress bar will show the status of the generation. This may take several minutes depending on the size of your data.
    * Once complete, a confirmation message will appear, indicating where the report has been saved.

## 5. Output

The program will generate the following:

* **An output folder** (either the one you selected or a new `output` folder).
* **PNG Images**: Inside the output folder, you will find all the individual chart images generated for the report.
* **PowerPoint Report**: A file named `Report_YYYY.pptx` (where `YYYY` is the selected primary year) will be saved in the output folder. This presentation contains all the generated charts organized into slides.

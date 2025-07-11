import os
import sys

import pandas as pd
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QPushButton, QLabel, QVBoxLayout, QComboBox, \
    QProgressBar, QLineEdit

from charts import generate_line_chart, generate_quarterly_prizm_barplots
from color_utils import parse_color_palette


class ReportApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Report Generator")
        # Set window icon (top left bar)
        self.setWindowIcon(QIcon(os.path.join(os.path.dirname(__file__), "assets", "STS+Logo.png")))
        self.layout = QVBoxLayout()

        self.vtn_label = QLabel("Select VTN File:")
        self.layout.addWidget(self.vtn_label)
        self.vtn_button = QPushButton("Browse VTN")
        self.vtn_button.clicked.connect(self.load_vtn_file)
        self.layout.addWidget(self.vtn_button)

        self.dmo_label = QLabel("Select DMO File:")
        self.layout.addWidget(self.dmo_label)
        self.dmo_button = QPushButton("Browse DMO")
        self.dmo_button.clicked.connect(self.load_dmo_file)
        self.layout.addWidget(self.dmo_button)

        # Year and comparison year selection
        self.year_label = QLabel("Select Year:")
        self.layout.addWidget(self.year_label)
        self.year_box = QComboBox()
        self.year_box.addItem("Select Year")
        self.year_box.currentIndexChanged.connect(self.on_year_changed)
        self.layout.addWidget(self.year_box)

        # Comparison year selection
        self.comp_year_label = QLabel("Select Comparison Year (optional):")
        self.layout.addWidget(self.comp_year_label)
        self.comp_year_box = QComboBox()
        self.comp_year_box.addItem("None")
        self.layout.addWidget(self.comp_year_box)

        # Province selection
        self.province_label = QLabel("Select Destination Province:")
        self.layout.addWidget(self.province_label)
        self.province_box = QComboBox()
        self.province_box.addItem("Select Province")
        provinces = [
            "Ontario", "Quebec", "Nova Scotia", "New Brunswick",
            "Manitoba", "British Columbia", "Prince Edward Island",
            "Saskatchewan", "Alberta", "Newfoundland and Labrador",
            "Yukon", "Northwest Territories", "Nunavut"
        ]
        self.province_box.addItems(provinces)
        self.layout.addWidget(self.province_box)

        # Custom color palette input
        self.color_palette_label = QLabel("Custom Color Palette (optional):")
        self.layout.addWidget(self.color_palette_label)

        self.color_palette_input = QLineEdit()
        self.color_palette_input.setPlaceholderText("Enter hex codes separated by commas (e.g., #FF5733, #33C4FF, #9C27B0)")
        self.color_palette_input.textChanged.connect(self.validate_color_input)
        self.layout.addWidget(self.color_palette_input)

        # Color validation message label
        self.color_validation_label = QLabel("")
        self.color_validation_label.setStyleSheet("color: #666666; font-size: 12px; font-weight: normal; margin-bottom: 8px;")
        self.layout.addWidget(self.color_validation_label)

        self.generate_button = QPushButton("Generate Report")
        self.generate_button.clicked.connect(self.generate_report)
        self.layout.addWidget(self.generate_button)

        # Add a progress bar
        self.progress_container = QWidget()
        self.progress_layout = QVBoxLayout(self.progress_container)
        self.progress_layout.setContentsMargins(0, 0, 0, 0)

        # Status label to show current operation
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #2563eb; font-size: 13px; font-weight: normal; margin-top: 0px;")
        self.progress_layout.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #bfc9d9;
                border-radius: 5px;
                text-align: center;
                height: 22px;
                margin-bottom: 8px;
            }
            QProgressBar::chunk {
                background-color: #4f8cff;
                border-radius: 4px;
            }
        """)
        self.progress_layout.addWidget(self.progress_bar)

        # Initially hide the progress container
        self.progress_container.setVisible(False)
        self.layout.addWidget(self.progress_container)

        # Add a label for error/log messages below the generate button
        self.message_label = QLabel("")
        self.message_label.setStyleSheet("color: #d90429; font-size: 14px; font-weight: 500; margin-top: 8px;")
        self.layout.addWidget(self.message_label)

        self.output_label = QLabel("Select Output Folder (optional) (recommended):")
        self.layout.addWidget(self.output_label)
        self.output_button = QPushButton("Browse Output Folder")
        self.output_button.clicked.connect(self.select_output_folder)
        self.layout.addWidget(self.output_button)
        self.output_dir = None

        self.setLayout(self.layout)
        self.vtn_path = None
        self.dmo_path = None
        self.dmo_mapping = {}

        self.setStyleSheet("""
            QWidget {
                background-color: #f4f6fa;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 14px;
            }
            QLabel {
                color: #22223b;
                font-size: 16px;
                font-weight: bold;
                margin-bottom: 6px;
            }
            QPushButton {
                background-color: #4f8cff;
                color: white;
                border-radius: 6px;
                padding: 8px 18px;
                font-size: 15px;
                font-weight: 500;
                margin-bottom: 10px;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
            QComboBox {
                background-color: #fff;
                border: 1px solid #bfc9d9;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 15px;
                margin-bottom: 10px;
            }
        """)

        self.setFixedSize(500, 750)
        self.setSizePolicy(self.sizePolicy().Fixed, self.sizePolicy().Fixed)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowMaximizeButtonHint)

        self.progress = 0
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self.update_progress_bar)
        self.is_generating = False

    def load_vtn_file(self):
        self.vtn_path, _ = QFileDialog.getOpenFileName(self, "Open VTN File", "", "CSV Files (*.csv)")
        if self.vtn_path:
            self.vtn_label.setText(f"VTN Selected: {os.path.basename(self.vtn_path)}")
            self.try_populate_regions()
        self.message_label.setText("")

    def load_dmo_file(self):
        self.dmo_path, _ = QFileDialog.getOpenFileName(self, "Open DMO File", "", "CSV Files (*.csv)")
        if self.dmo_path:
            self.dmo_label.setText(f"DMO Selected: {os.path.basename(self.dmo_path)}")
            self.try_populate_regions()
        self.message_label.setText("")

    def try_populate_regions(self):
        if self.vtn_path and self.dmo_path:
            dmo_df = pd.read_csv(self.dmo_path)
            # Use DEST_CODE for DMO code
            if "DEST_CODE" not in dmo_df.columns or "DEST_NAME" not in dmo_df.columns:
                self.message_label.setText("DMO file missing 'DEST_CODE' or 'DEST_NAME'")
                return

            self.dmo_mapping = dict(zip(dmo_df["DEST_NAME"], dmo_df["DEST_CODE"]))

            df = pd.read_csv(self.vtn_path)
            if "Year" not in df.columns and "date" in df.columns:
                df["Year"] = pd.to_datetime(df["date"]).dt.year

            years = sorted(df["Year"].dropna().unique().astype(int).tolist())
            self.year_box.clear()
            self.year_box.addItem("Select Year")
            self.year_box.addItems([str(y) for y in years])

            # Populate comparison year dropdown
            self.comp_year_box.clear()
            self.comp_year_box.addItem("None")
            self.comp_year_box.addItems([str(y) for y in years])

            # Don't clear and repopulate province dropdown - keep the static list
            # The province dropdown should already be populated with all Canadian provinces

        self.message_label.setText("")

    def set_generate_loading(self, loading: bool, status_text=None):
        if loading:
            self.is_generating = True
            self.progress = 0
            self.generate_button.setEnabled(False)
            self.generate_button.setText("Generating Report...")

            # Initialize and show progress bar
            self.progress_container.setVisible(True)
            self.progress_bar.setValue(0)

            # Set status text if provided
            if status_text:
                self.status_label.setText(status_text)
            else:
                self.status_label.setText("Preparing data...")

            # Initialize progress tracking
            self.total_progress_steps = 0
            self.current_progress_step = 0

            # Don't start the timer-based progress for accurate tracking
            # self.progress_timer.start(50)
        else:
            self.is_generating = False
            self.progress_timer.stop()
            self.generate_button.setEnabled(True)
            self.generate_button.setText("Generate Report")

            # Hide progress container
            self.progress_container.setVisible(False)
            self.status_label.setText("")

    def update_progress(self, steps_completed=None, status_text=None, force_value=None):
        """
        Update progress bar with accurate progress tracking.

        Args:
            steps_completed: Number of steps completed (optional, uses internal counter if not provided)
            status_text: Status message to display
            force_value: Force progress bar to specific value (0-100)
        """
        if not self.is_generating:
            return

        if force_value is not None:
            # Force specific progress value
            self.progress_bar.setValue(force_value)
        elif steps_completed is not None:
            # Use provided step count
            if self.total_progress_steps > 0:
                progress_percent = min(100, int((steps_completed / self.total_progress_steps) * 100))
                self.progress_bar.setValue(progress_percent)
        elif self.total_progress_steps > 0:
            # Use internal step counter
            self.current_progress_step += 1
            progress_percent = min(100, int((self.current_progress_step / self.total_progress_steps) * 100))
            self.progress_bar.setValue(progress_percent)

        if status_text:
            self.status_label.setText(status_text)

        QApplication.processEvents()

    def calculate_total_progress_steps(self, dmo_count, has_comparison_year):
        """
        Calculate total number of progress steps for accurate tracking.
        Adjusted to reflect actual processing time weights.

        Args:
            dmo_count: Number of DMO regions to process
            has_comparison_year: Whether comparison year charts will be generated

        Returns:
            Total number of progress steps
        """
        steps = 0

        # Initial setup steps (relatively fast)
        steps += 5  # File validation, data loading, data filtering

        # DMO processing steps (fast - line charts)
        # Each DMO generates: 3 line charts (Visitor, Trips, Nights)
        steps += dmo_count * 3

        # Heavy processing steps (much slower operations)
        # These are the operations that actually take most of the time
        steps += 15  # Grouped barplot generation (very slow - multiple complex charts)
        steps += 8   # Province pie charts and quarterly analysis
        steps += 10  # Origin analysis charts
        steps += 8   # Out-of-province charts

        # PRIZM chart generation (very slow)
        if has_comparison_year:
            steps += 20  # PRIZM circular charts for both years
            steps += 15  # PRIZM bar charts for both years
            steps += 25  # Quarterly PRIZM charts (very complex)
        else:
            steps += 10  # PRIZM circular charts for one year
            steps += 8   # PRIZM bar charts for one year

        # Final steps
        steps += 5  # PowerPoint creation and finalization

        return steps

    def update_progress_bar(self):
        if not self.is_generating:
            return
        # Only update timer-based progress when we don't have concrete progress info
        if not hasattr(self, 'concrete_progress'):
            self.progress = min(100, self.progress + 2)
            self.progress_bar.setValue(self.progress)
        if self.progress >= 100:
            self.progress_timer.stop()

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder", "")
        if folder:
            self.output_dir = folder
            self.output_label.setText(f"Output Folder: {os.path.basename(folder)}")
        else:
            self.output_label.setText("Select Output Folder (optional):")

    def on_year_changed(self):
        # Clear the comparison year box
        self.comp_year_box.clear()
        self.comp_year_box.addItem("None")

        if self.year_box.currentText() == "Select Year":
            return

        try:
            selected_year = int(self.year_box.currentText())
        except ValueError:
            return

        # Only continue if we have a VTN file loaded
        if not self.vtn_path:
            return

        # Get the list of available years from the VTN file
        vtn_df = pd.read_csv(self.vtn_path)

        if "Year" not in vtn_df.columns and "date" in vtn_df.columns:
            vtn_df["Year"] = pd.to_datetime(vtn_df["date"], errors="coerce").dt.year

        if "Year" not in vtn_df.columns:
            self.message_label.setText("VTN file does not contain a 'Year' column.")
            return

        available_years = sorted(vtn_df["Year"].dropna().unique().astype(int).tolist())

        # Filter to only include years less than the selected year
        comparison_years = [y for y in available_years if y < selected_year]

        if comparison_years:
            # Populate the comparison year box with available years
            self.comp_year_box.addItems([str(y) for y in comparison_years])
        else:
            self.message_label.setText("No earlier years available for comparison.")

    def validate_color_input(self):
        """
        Validate the custom color palette input and show feedback to user.
        """
        palette_input = self.color_palette_input.text().strip()

        if not palette_input:
            self.color_validation_label.setText("")
            return

        parsed_colors = parse_color_palette(palette_input)

        if parsed_colors:
            num_colors = len(parsed_colors)
            self.color_validation_label.setText(f"✓ Valid palette ({num_colors} colors)")
            self.color_validation_label.setStyleSheet("color: #27AE60; font-size: 12px; font-weight: normal; margin-bottom: 8px;")
        else:
            self.color_validation_label.setText("✗ Invalid format. Use #RRGGBB format separated by commas")
            self.color_validation_label.setStyleSheet("color: #d90429; font-size: 12px; font-weight: normal; margin-bottom: 8px;")

    def generate_report(self):
        self.set_generate_loading(True, "Validating input files...")

        try:
            if not self.vtn_path or not self.dmo_path:
                self.message_label.setText("Please select both VTN and DMO files.")
                self.set_generate_loading(False)
                return

            if self.year_box.currentText() == "Select Year":
                self.message_label.setText("Please select a year.")
                self.set_generate_loading(False)
                return

            if self.province_box.currentText() == "Select Province":
                self.message_label.setText("Please select a destination province.")
                self.set_generate_loading(False)
                return

            # Validate custom color palette if provided
            custom_palette = None
            palette_input = self.color_palette_input.text().strip()
            if palette_input:
                custom_palette = parse_color_palette(palette_input)
                if custom_palette is None:
                    self.message_label.setText("Invalid color palette. Please fix the color codes.")
                    self.set_generate_loading(False)
                    return

            year = int(self.year_box.currentText())
            selected_province = self.province_box.currentText()

            # Get comparison year if selected
            comp_year = None
            if self.comp_year_box.currentText() != "None" and self.comp_year_box.currentText() != "Select Year":
                comp_year = int(self.comp_year_box.currentText())
                if comp_year >= year:
                    self.message_label.setText("Comparison year must be earlier than the selected year.")
                    self.set_generate_loading(False)
                    return
                print(f"Using comparison year: {comp_year}")

            # Step 1: Load data files
            self.update_progress(status_text="Loading VTN and DMO data files...")

            vtn_df = pd.read_csv(self.vtn_path)
            print(f"VTN file loaded: {len(vtn_df)} rows")
            dmo_df = pd.read_csv(self.dmo_path)

            # Set correct path for Geography.xlsx in assets folder
            geography_xlsx_path = os.path.join(os.path.dirname(__file__), 'assets', 'Geography.xlsx')
            print(f"[DEBUG] Using Geography.xlsx path: {geography_xlsx_path}")

            # Check if DMO file is empty or missing columns
            if dmo_df.empty or "DEST_CODE" not in dmo_df.columns or "DEST_NAME" not in dmo_df.columns:
                self.message_label.setText("DMO file is empty or missing required columns.")
                self.set_generate_loading(False)
                return

            # Step 2: Process date information
            self.update_progress(status_text="Processing date and geographic information...")

            if "date" not in vtn_df.columns:
                self.message_label.setText("VTN file missing 'date' column.")
                self.set_generate_loading(False)
                return

            vtn_df["date"] = pd.to_datetime(vtn_df["date"], errors="coerce")
            vtn_df["Year"] = vtn_df["date"].dt.year.astype("Int64")
            vtn_df["Month"] = vtn_df["date"].dt.strftime("%B")

            # Ensure DMO columns are numeric and drop rows with missing DMO codes
            vtn_df["DMO"] = pd.to_numeric(vtn_df["DMO"], errors="coerce").astype("Int64")
            dmo_df["DEST_CODE"] = pd.to_numeric(dmo_df["DEST_CODE"], errors="coerce").astype("Int64")
            dmo_df = dmo_df.dropna(subset=["DEST_CODE"])

            if dmo_df.empty:
                self.message_label.setText("No valid DEST_CODEs found in DMO file.")
                self.set_generate_loading(False)
                return

            # Debug: print unique DMO codes in both files
            print("Unique DMO codes in VTN:", vtn_df["DMO"].unique())
            print("Unique DEST_CODEs in DMO file:", dmo_df["DEST_CODE"].unique())

            # Step 3: Filter and validate data
            self.update_progress(status_text="Filtering and validating monthly data...")

            vtn_df["PERIOD_TYPE"] = vtn_df["PERIOD_TYPE"].astype(str).str.strip().str.lower()
            monthly_df = vtn_df[vtn_df["PERIOD_TYPE"] == "months"]

            # Debug: print available years and months after filtering
            print("Available years in monthly_df:", monthly_df["Year"].unique())
            print("Available months in monthly_df:", monthly_df["Month"].unique())

            # Ensure Visitor, Trips, Nights columns exist and are numeric
            for col in ["Visitor", "Trips", "Nights"]:
                if col not in monthly_df.columns:
                    self.message_label.setText(f"VTN file missing '{col}' column.")
                    self.set_generate_loading(False)
                    return
                monthly_df.loc[:, col] = pd.to_numeric(monthly_df[col], errors="coerce").fillna(0)

            # Prepare output directory
            if self.output_dir:
                output_dir = self.output_dir
            else:
                output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)

            # Calculate total progress steps for accurate tracking
            dmo_rows = list(dmo_df.iterrows())
            total_regions = len(dmo_rows)
            self.total_progress_steps = self.calculate_total_progress_steps(total_regions, comp_year is not None)
            self.current_progress_step = 5  # We've completed 5 initial steps

            slides = []

            # Process each DMO region (this should be relatively fast)
            for i, (_, dmo_row) in enumerate(dmo_rows):
                dmo_code = dmo_row["DEST_CODE"]
                dmo_name = dmo_row["DEST_NAME"]
                # Clean up DMO display name: strip and collapse multiple spaces
                if "DMO" in dmo_row:
                    dmo_display = " ".join(str(dmo_row["DMO"]).split())
                else:
                    dmo_display = " ".join(str(dmo_name).split())

                # Get the main year data
                dmo_data = monthly_df[(monthly_df["DMO"] == dmo_code) & (monthly_df["Year"] == year)]
                print(f"Processing DMO {dmo_code} ({dmo_name}): {len(dmo_data)} rows for year {year}")

                if dmo_data.empty:
                    print(f"No monthly data for DMO {dmo_code} ({dmo_name}) in year {year}")
                    continue

                # Aggregate by Year and Month, summing across all PRIZM_CODEs
                agg = dmo_data.groupby(["Year", "Month"], as_index=False)[["Visitor", "Trips", "Nights"]].sum()

                # If comparison year is selected, get that data too
                comp_data = None
                comp_agg = None
                if comp_year:
                    comp_data = monthly_df[(monthly_df["DMO"] == dmo_code) & (monthly_df["Year"] == comp_year)]
                    if not comp_data.empty:
                        comp_agg = comp_data.groupby(["Year", "Month"], as_index=False)[["Visitor", "Trips", "Nights"]].sum()
                        print(f"Added comparison data for DMO {dmo_code}: {len(comp_data)} rows for year {comp_year}")
                    else:
                        print(f"No comparison data for DMO {dmo_code} in year {comp_year}")

                chart_paths = []
                # Generate Visitor and Trips charts
                for metric in ["Visitor", "Trips"]:
                    self.update_progress(status_text=f"Generating {metric} chart for {dmo_display} ({i+1}/{total_regions})...")

                    # Prepare main year data
                    metric_df = agg[["Year", "Month", metric]].copy()
                    metric_df = metric_df.rename(columns={metric: "Value"})
                    metric_df["Type"] = metric

                    # Add comparison year data if available
                    if comp_year and comp_agg is not None:
                        comp_metric_df = comp_agg[["Year", "Month", metric]].copy()
                        comp_metric_df = comp_metric_df.rename(columns={metric: "Value"})
                        comp_metric_df["Type"] = f"{metric} {comp_year}"
                        # Combine both dataframes
                        metric_df = pd.concat([metric_df, comp_metric_df])

                        # Create chart with both years
                        chart_path = os.path.join(output_dir, f"{dmo_display}_{metric}_{year}_vs_{comp_year}.png")
                        if metric == "Visitor":
                            chart_title = f"{dmo_display} Visitors {year} vs {comp_year}"
                        else:
                            chart_title = f"{dmo_display} {metric} {year} vs {comp_year}"
                    else:
                        # Create chart with just main year
                        chart_path = os.path.join(output_dir, f"{dmo_display}_{metric}_{year}.png")
                        if metric == "Visitor":
                            chart_title = f"{dmo_display} Visitors {year}"
                        else:
                            chart_title = f"{dmo_display} {metric} {year}"

                    # Generate the chart
                    generate_line_chart(metric_df, chart_title, chart_path, custom_palette=custom_palette)
                    chart_paths.append((metric, chart_path))

                # Generate Nights chart
                self.update_progress(status_text=f"Generating Nights chart for {dmo_display} ({i+1}/{total_regions})...")

                nights_chart_paths = []
                metric = "Nights"
                # Prepare main year data
                metric_df = agg[["Year", "Month", metric]].copy()
                metric_df = metric_df.rename(columns={metric: "Value"})
                metric_df["Type"] = metric

                # Add comparison year data if available
                if comp_year and comp_agg is not None:
                    comp_metric_df = comp_agg[["Year", "Month", metric]].copy()
                    comp_metric_df = comp_metric_df.rename(columns={metric: "Value"})
                    comp_metric_df["Type"] = f"{metric} {comp_year}"
                    # Combine both dataframes
                    metric_df = pd.concat([metric_df, comp_metric_df])

                    # Create chart with both years
                    chart_path = os.path.join(output_dir, f"{dmo_display}_{metric}_{year}_vs_{comp_year}.png")
                    chart_title = f"{dmo_display} {metric} {year} vs {comp_year}"
                else:
                    # Create chart with just main year
                    chart_path = os.path.join(output_dir, f"{dmo_display}_{metric}_{year}.png")
                    chart_title = f"{dmo_display} {metric} {year}"

                # Generate the chart
                generate_line_chart(metric_df, chart_title, chart_path, custom_palette=custom_palette)
                nights_chart_paths.append((metric, chart_path))

                # Add to slides collection
                slides.append({
                    "dmo_name": dmo_name,
                    "DMO": dmo_display,
                    "year": year,
                    "comp_year": comp_year,
                    "charts": chart_paths,
                    "slide_type": "visitors_trips"
                })
                slides.append({
                    "dmo_name": dmo_name,
                    "DMO": dmo_display,
                    "year": year,
                    "comp_year": comp_year,
                    "charts": nights_chart_paths,
                    "slide_type": "nights"
                })

            if not slides:
                self.message_label.setText("No data for selected year in any region.")
                self.set_generate_loading(False)
                return

            # Now handle the heavy processing operations with accurate progress tracking
            from charts import generate_grouped_barplot, generate_circular_barplot, generate_prizm_barplot
            geography_xlsx_path = os.path.join(os.path.dirname(__file__), 'assets', 'Geography.xlsx')
            profile_csv_path = os.path.join(os.path.dirname(__file__), 'assets', 'Profile.csv')

            # Generate grouped barplots (Top 5 Origins) - this is slow
            self.update_progress(status_text="Generating Top 5 Origins charts...")
            generate_grouped_barplot(self.vtn_path, geography_xlsx_path, self.dmo_path, output_dir, year=year, destination_province=selected_province, custom_palette=custom_palette)

            # Update progress after each major operation completes
            self.current_progress_step += 15  # Add the weighted steps for grouped barplot

            self.update_progress(status_text="Generating province and quarterly analysis...")
            # This represents additional sub-operations within grouped barplot that are already completed
            self.current_progress_step += 8

            self.update_progress(status_text="Generating detailed origin analysis...")
            # This represents additional sub-operations within grouped barplot that are already completed
            self.current_progress_step += 10

            self.update_progress(status_text="Generating out-of-province analysis...")
            # This represents additional sub-operations within grouped barplot that are already completed
            self.current_progress_step += 8

            # Generate PRIZM charts with accurate progress tracking
            if comp_year:
                self.update_progress(status_text=f"Generating PRIZM Traveller Segmentation charts for {year}...")
                generate_circular_barplot(self.vtn_path, profile_csv_path, self.dmo_path, output_dir, year=year, comp_year=None, destination_province=selected_province, custom_palette=custom_palette)
                self.current_progress_step += 10

                self.update_progress(status_text=f"Generating PRIZM Traveller Segmentation charts for {comp_year}...")
                generate_circular_barplot(self.vtn_path, profile_csv_path, self.dmo_path, output_dir, year=comp_year, comp_year=None, destination_province=selected_province, custom_palette=custom_palette)
                self.current_progress_step += 10

                self.update_progress(status_text=f"Generating PRIZM demographic charts for {year}...")
                generate_prizm_barplot(self.vtn_path, profile_csv_path, self.dmo_path, output_dir, year=year, comp_year=None, destination_province=selected_province, custom_palette=custom_palette)
                self.current_progress_step += 8

                self.update_progress(status_text=f"Generating PRIZM demographic charts for {comp_year}...")
                generate_prizm_barplot(self.vtn_path, profile_csv_path, self.dmo_path, output_dir, year=comp_year, comp_year=None, destination_province=selected_province, custom_palette=custom_palette)
                self.current_progress_step += 7

                self.update_progress(status_text="Generating quarterly PRIZM comparison charts...")
                generate_quarterly_prizm_barplots(self.vtn_path, profile_csv_path, self.dmo_path, output_dir, year=year, comp_year=comp_year, destination_province=selected_province, custom_palette=custom_palette)
                self.current_progress_step += 25
            else:
                self.update_progress(status_text="Generating PRIZM Traveller Segmentation charts...")
                generate_circular_barplot(self.vtn_path, profile_csv_path, self.dmo_path, output_dir, year=year, comp_year=comp_year, destination_province=selected_province, custom_palette=custom_palette)
                self.current_progress_step += 10

                self.update_progress(status_text="Generating PRIZM demographic charts...")
                generate_prizm_barplot(self.vtn_path, profile_csv_path, self.dmo_path, output_dir, year=year, comp_year=comp_year, destination_province=selected_province, custom_palette=custom_palette)
                self.current_progress_step += 8

            # PowerPoint creation
            self.update_progress(status_text="Creating PowerPoint presentation...")
            ppt_path = os.path.join(output_dir, f"Report_{year}.pptx")
            from ppt_generator import create_presentation
            create_presentation(f"Report {year}", slides, ppt_path)
            self.current_progress_step += 5

            # Complete
            self.update_progress(force_value=100, status_text="Report generation complete!")
            self.message_label.setText(f"Report saved: {ppt_path}")

        except Exception as e:
            self.message_label.setText(f"Error generating report: {str(e)}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.set_generate_loading(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ReportApp()
    window.show()
    sys.exit(app.exec_())
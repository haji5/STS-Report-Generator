import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import the color palette helper functions from color_utils module
from color_utils import get_color_palette

# Custom month ordering
MONTH_ORDER = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

# Quarter ordering
QUARTER_ORDER = ["Q1", "Q2", "Q3", "Q4"]


def generate_line_chart(df, region_name, output_path, custom_palette=None):
    # Expects df with columns: Year, Month, Value, Type
    df = df.copy()
    df["Month"] = pd.Categorical(df["Month"], categories=MONTH_ORDER, ordered=True)

    # Check if we have comparison data (multiple Type values or Years)
    has_comparison = len(df["Type"].unique()) > 1 or len(df["Year"].unique()) > 1

    # Sort data by Year and Month
    df = df.sort_values(["Year", "Month"])

    # Create figure and axis
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(13, 5.5), dpi=150)

    # Get color palette for different lines
    unique_types = df["Type"].unique()
    colors_list = get_color_palette(custom_palette, len(unique_types))

    # Create a mapping from type to color
    colors = {}
    for i, type_val in enumerate(unique_types):
        colors[type_val] = colors_list[i % len(colors_list)]

    # If we have comparison data, create a separate color for those lines
    if has_comparison:
        for type_val in df["Type"].unique():
            if "Visitor" in type_val and type_val != "Visitor":
                colors[type_val] = "#f5a623"  # Orange for comparison year Visitor
            elif "Trips" in type_val and type_val != "Trips":
                colors[type_val] = "#f5a623"  # Orange for comparison year Trips
            elif "Nights" in type_val and type_val != "Nights":
                colors[type_val] = "#a34a28"  # Dark orange for comparison year Nights

    # First, create all line plots without labels to determine their positions
    line_objects = {}

    # Calculate data range normalization to flatten curves and avoid x-axis conflicts
    all_values = df["Value"].values
    data_min = np.min(all_values)
    data_max = np.max(all_values)
    data_range = data_max - data_min

    # Compress the data range by 25% to flatten curves while maintaining relative relationships
    compression_factor = 0.75  # Use 75% of original range
    range_center = data_min + (data_range / 2)
    compressed_range = data_range * compression_factor

    # Calculate minimum baseline to keep curves well above x-axis
    baseline_offset = data_range * 0.15  # 15% of original range as baseline

    def normalize_value(value):
        # Normalize to compressed range and add baseline offset
        normalized = range_center - (compressed_range / 2) + ((value - data_min) / data_range) * compressed_range
        return normalized + baseline_offset

    for type_val in df["Type"].unique():
        series_df = df[df["Type"] == type_val]
        line_color = colors.get(type_val, "#005691")  # Default to blue if type not found

        # Apply normalization to flatten the curve
        normalized_values = [normalize_value(val) for val in series_df["Value"]]

        # Create the line plot with normalized values
        line, = ax.plot(
            series_df["Month"],
            normalized_values,
            marker='o',
            markersize=8,
            markerfacecolor="#FFFFFF",
            markeredgewidth=2.5,
            markeredgecolor=line_color,
            linewidth=2.8,
            color=line_color,
            label=type_val,  # Add label for the legend
            zorder=3
        )

        # Store line object and its original data for later label positioning
        line_objects[type_val] = {
            'line': line,
            'data': series_df[["Month", "Value"]].set_index("Month")["Value"].to_dict(),
            'normalized_data': dict(zip(series_df["Month"], normalized_values)),
            'color': line_color  # Store color for labels
        }

    # Create a dictionary to store all values for each month across all lines
    month_to_values = {}
    for month in MONTH_ORDER:
        month_to_values[month] = []
        for type_val, line_info in line_objects.items():
            if month in line_info['data']:
                month_to_values[month].append((type_val, line_info['data'][month]))

    # Calculate y-axis limits for proper annotation placement
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    # Now add the value labels with appropriate positioning using annotations
    for type_val, line_info in line_objects.items():
        series_df = df[df["Type"] == type_val]
        line_color = line_info['color']  # Get the color for this line

        # Add value labels for each point in this series
        for month, value in line_info['data'].items():
            if value == 0:
                continue

            # Determine if this line is the higher or lower at this month position
            is_higher_line = True
            if month in month_to_values and len(month_to_values[month]) > 1:
                # Sort the values for this month from highest to lowest
                sorted_values = sorted(month_to_values[month], key=lambda x: x[1], reverse=True)
                # Check if current type is the highest
                is_higher_line = sorted_values[0][0] == type_val

            # Format value with commas and millions if appropriate
            if abs(value) >= 1_000_000:
                label = f"{value / 1_000_000:,.1f}M"
            elif abs(value) >= 1_000:
                label = f"{value:,.0f}"
            else:
                label = f"{value}"

            # Determine vertical offset and position based on whether this is higher line
            # Use the normalized data for positioning annotations
            normalized_value = line_info['normalized_data'][month]

            if is_higher_line:
                # For higher values, place above with larger offset
                xy = (month, normalized_value)  # Point on the normalized line
                xytext = (0, 22)  # Offset for higher values
                va = 'bottom'
            else:
                # For lower values, place below with larger offset
                xy = (month, normalized_value)  # Point on the normalized line
                xytext = (0, -22)  # Offset for lower values
                va = 'top'

            # Create annotation with improved positioning and white box
            # Use the same color as the line for the annotation text
            annotation = ax.annotate(
                label,
                xy=xy,  # Point being annotated
                xytext=xytext,  # Offset from point
                textcoords='offset points',
                ha='center', va=va,
                fontsize=11,
                fontweight='bold',
                color=line_color,  # Match text color to line color
                # Increase padding around text and make box more visible
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    fc="white",
                    ec="#CCCCCC",  # Light gray border
                    alpha=1.0,  # Fully opaque
                    linewidth=0.5  # Thin border
                ),
                zorder=10,  # Ensure labels are above everything else
                annotation_clip=True
            )

    # Add legend if we have comparison data
    if has_comparison:
        ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=11)

    # Title and subtitle
    ax.set_title(region_name, fontsize=18, fontweight='bold', pad=18, color="#222222")

    # Remove axis labels
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Ticks and grid
    ax.tick_params(axis='x', labelrotation=45, labelsize=12, colors="#222222")
    # Hide y-axis tick labels by setting empty strings
    ax.set_yticklabels([])
    ax.yaxis.grid(True, which='major', linestyle='--', linewidth=1, alpha=0.6, zorder=1)
    ax.xaxis.grid(False)

    # Remove top/right spines, thicken left/bottom
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')

    # Add subtle background color
    ax.set_facecolor('#F8FAFC')
    fig.patch.set_facecolor('#F8FAFC')

    # Ensure y-axis has enough room for labels - increased bottom margin significantly
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * 0.15, y_max + y_range * 0.18)  # Increased bottom margin from 0.1 to 0.15

    # Tight layout
    plt.tight_layout(pad=2)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_charts_by_dmo(df, dmo_name_col="DEST_NAME", dmo_code_col="DMO", year=None, output_dir="output", custom_palette=None):
    """
    For a given DataFrame filtered by year, generate a chart for each DMO/city.
    Returns a list of output file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    chart_paths = []
    group_col = dmo_name_col if dmo_name_col in df.columns else dmo_code_col

    for dmo_value, group in df.groupby(group_col):
        region_name = str(dmo_value)
        chart_path = os.path.join(output_dir, f"bar_chart_{region_name}_{year}.png")
        generate_line_chart(group, region_name, chart_path, custom_palette=custom_palette)
        chart_paths.append(chart_path)
    return chart_paths


def generate_grouped_barplot(vtn_csv_path, geography_xlsx_path, dmo_csv_path, output_dir, year=None, destination_province=None, custom_palette=None):
    """
    For each DMO in VTN.csv, generate two grouped barplots:
    1. Top 5 origins (by Visitors) - All origins
    2. Top 5 out-of-province origins (by Visitors)

    Both plots show Nights, Trips, and Visitors for each origin by quarter.
    Origin names are mapped using Geography.xlsx.
    Plots are saved to output_dir as:
      - <DMO>_top5_origins.png (all origins)
      - <DMO>_top5_outofprov_origins.png (out-of-province origins)

    Args:
        vtn_csv_path: Path to VTN CSV file
        geography_xlsx_path: Path to Geography Excel file
        dmo_csv_path: Path to DMO CSV file
        output_dir: Directory to save output files
        year: Year to filter data for
        destination_province: User-selected province where destinations are located
        custom_palette: Custom color palette to use for charts

    If quarterly data (PERIOD_TYPE == "Quarters") is not available, monthly data will be aggregated into quarters.
    """
    print("[DEBUG] Entering generate_grouped_barplot")
    # Always use Geography.xlsx from assets folder
    assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
    geography_xlsx_path = os.path.join(assets_dir, 'Geography.xlsx')
    print(f"[DEBUG] Using Geography.xlsx path: {geography_xlsx_path}")

    # Read VTN.csv
    vtn_df = pd.read_csv(vtn_csv_path)
    print(f"[DEBUG] Loaded VTN file with {len(vtn_df)} rows")

    # Parse 'date' column to datetime and extract year/month
    if "date" in vtn_df.columns:
        vtn_df["date"] = pd.to_datetime(vtn_df["date"], errors="coerce")
        vtn_df["Year"] = vtn_df["date"].dt.year.astype("Int64")
        vtn_df["Month"] = vtn_df["date"].dt.strftime("%B")
        # Add quarter information based on month
        vtn_df["MonthNum"] = vtn_df["date"].dt.month
        vtn_df["Quarter"] = vtn_df["MonthNum"].apply(lambda m:
            "Q1" if m <= 3 else
            "Q2" if m <= 6 else
            "Q3" if m <= 9 else "Q4"
        )
        print(f"[DEBUG] Parsed date column, years available: {vtn_df['Year'].unique()}")

    # Check if we have quarterly data (PERIOD_TYPE == "Quarters")
    has_quarterly_data = False
    if "PERIOD_TYPE" in vtn_df.columns:
        available_period_types = vtn_df["PERIOD_TYPE"].astype(str).str.strip().unique()
        print(f"[DEBUG] Available PERIOD_TYPE values in VTN file: {available_period_types}")

        # Try to use quarterly data if available
        quarterly_df = vtn_df[vtn_df["PERIOD_TYPE"].astype(str).str.strip().str.lower() == "quarters"].copy()
        if not quarterly_df.empty:
            has_quarterly_data = True
            vtn_df = quarterly_df
            print(f"[DEBUG] Found {len(vtn_df)} rows with PERIOD_TYPE='Quarters'")

            # Ensure Quarter column exists for quarterly data
            if "PERIOD" in vtn_df.columns:
                vtn_df["Quarter"] = vtn_df["PERIOD"].astype(str).apply(lambda x: f"Q{x}" if x.isdigit() else x)

    # If we don't have quarterly data, use monthly data aggregated to quarters
    if not has_quarterly_data:
        print("[DEBUG] No quarterly data found, using monthly data aggregated to quarters")
        # Filter to just monthly data
        if "PERIOD_TYPE" in vtn_df.columns:
            monthly_df = vtn_df[vtn_df["PERIOD_TYPE"].astype(str).str.strip().str.lower() == "months"].copy()
            if not monthly_df.empty:
                vtn_df = monthly_df
                print(f"[DEBUG] Using {len(vtn_df)} rows of monthly data")

        # Check if we have the Quarter column now
        if "Quarter" not in vtn_df.columns:
            print("[WARNING] Unable to determine quarter information from data")
            return

    # Filter for selected year
    if year is not None:
        if 'Year' in vtn_df.columns:
            vtn_df = vtn_df[vtn_df['Year'] == year]
            print(f"[DEBUG] After filtering for year {year}, {len(vtn_df)} rows remain")

    # Check if we have any data at all
    if vtn_df.empty:
        print("[ERROR] No valid data found in VTN file after filtering")
        return

    # Ensure Visitor, Trips, Nights columns exist and are numeric
    for col in ["Visitor", "Trips", "Nights"]:
        if col in vtn_df.columns:
            vtn_df.loc[:, col] = pd.to_numeric(vtn_df[col], errors="coerce").fillna(0)
        else:
            print(f"[WARNING] Column '{col}' not found in VTN file")
            return

    # Read DMO.csv to map DMO code to name (use the one passed in)
    dmo_df = pd.read_csv(dmo_csv_path)
    # Use DEST_CODE as key, DMO as value (DEST_CODE is the number in VTN, DMO is the location name)
    dmo_df['DEST_CODE'] = pd.to_numeric(dmo_df['DEST_CODE'], errors='coerce').astype('Int64')
    vtn_df.loc[:, 'DMO'] = pd.to_numeric(vtn_df['DMO'], errors='coerce').astype('Int64')

    # Read Geography.xlsx to create origin mappings
    origin_map = {}
    origin_province_map = {}

    try:
        geography_df = pd.read_excel(geography_xlsx_path, sheet_name='Origins')
        print(f"[DEBUG] Loaded Geography.xlsx with {len(geography_df)} rows")

        # Create origin code to name mapping
        for _, row in geography_df.iterrows():
            if 'ORIGIN_PRCDCSD' in row and 'ORIGIN_CSDNAMEE' in row:
                code = row['ORIGIN_PRCDCSD']
                name = row['ORIGIN_CSDNAMEE']
                province = row.get('ORIGIN_PRNAMEE', 'Unknown')

                origin_map[code] = name
                origin_province_map[code] = province

        print(f"[DEBUG] Created origin mappings for {len(origin_map)} locations")

    except Exception as e:
        print(f"[ERROR] Failed to load Geography.xlsx: {e}")
        return

    for _, dmo_row in dmo_df.iterrows():
        dmo_code = dmo_row['DEST_CODE']
        dmo_display = " ".join(str(dmo_row['DMO']).split()) if 'DMO' in dmo_row else str(dmo_row['DEST_NAME'])
        dmo_df_sub = vtn_df[vtn_df["DMO"] == dmo_code]
        if dmo_df_sub.empty:
            continue

        # Use the user-selected destination province instead of auto-detection
        if destination_province and destination_province != "All Provinces":
            dest_province = destination_province
            print(f"[DEBUG] Using user-selected destination province for {dmo_display}: {dest_province}")
        else:
            # Fallback to Ontario if no province selected or "All Provinces" selected
            dest_province = "Ontario"
            print(f"[DEBUG] No specific province selected, defaulting to Ontario for {dmo_display}")

        # Group by origin, sum metrics to identify top 5 origins
        grouped = dmo_df_sub.groupby("ORIGIN_PRCDCSD")[['Visitor', 'Trips', 'Nights']].sum()

        # Create safe filename for this DMO
        safe_name = dmo_display.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')

        # Generate province pie chart
        pie_title = f"Visitors to {dmo_display} by Province of Origin"
        pie_chart = generate_province_pie_chart(dmo_df_sub, origin_province_map, pie_title, custom_palette=custom_palette)
        pie_chart_path = os.path.join(output_dir, f"{safe_name}_province_pie.png")
        pie_chart.savefig(pie_chart_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(pie_chart)
        print(f"[DEBUG] Generated province pie chart at {pie_chart_path}")

        # Generate nights per visitor by quarter chart
        nights_title = f"Nights per Visitor by Quarter - {dmo_display}"
        nights_path = os.path.join(output_dir, f"{safe_name}_nights_per_visitor_quarterly.png")
        generate_nights_per_visitor_by_quarter(dmo_df_sub, origin_province_map, dest_province, nights_title, nights_path, custom_palette=custom_palette)

        # PLOT 1: ALL ORIGINS
        # Get top 5 origins by Visitors
        top5_origins = grouped.sort_values(by="Visitor", ascending=False).head(5).index.tolist()

        # Filter data for just these top 5 origins and include quarter information
        top5_data = dmo_df_sub[dmo_df_sub["ORIGIN_PRCDCSD"].isin(top5_origins)]

        if not top5_data.empty and "Quarter" in top5_data.columns:
            # Map origin codes to names
            origin_names = [origin_map.get(code, str(code)) for code in top5_origins]

            # Create a new figure with increased width for better visibility
            plt.style.use("seaborn-v0_8-whitegrid")
            fig, ax = plt.subplots(figsize=(16, 8), dpi=150)

            # Setup plot parameters
            n_origins = len(origin_names)
            n_quarters = len(QUARTER_ORDER)
            n_metrics = 3  # Nights, Trips, Visitors

            # Calculate width for individual bars and spacing - reduced gaps between cities
            city_spacing = 0.8  # Reduced spacing between cities (was 0.4)
            group_width = 2.4  # Increased width for each city group to make bars larger
            quarter_spacing = 0.05  # Minimal spacing between quarters within a city
            effective_quarter_width = (group_width * (1 - quarter_spacing)) / n_quarters
            bar_width = effective_quarter_width / n_metrics

            # Color mapping for metrics using custom palette
            colors_list = get_color_palette(custom_palette, 3)
            colors = {
                "Nights": colors_list[0],    # First color
                "Trips": colors_list[1],     # Second color
                "Visitors": colors_list[2]   # Third color
            }

            # Create x positions for bars grouped by city, then quarter, then metric
            x_positions = []

            # Setup positions for x-ticks (center of each origin group)
            x_ticks = []

            # Keep track of quarter positions for labels
            quarter_positions = [[] for _ in range(n_quarters)]

            # Keep track of all bars created for adding value labels
            all_bars = []

            # Loop through origins and create grouped bars
            for i, origin_idx in enumerate(range(n_origins)):
                # Center position for this origin group - reduced spacing
                origin_center = i * (group_width + city_spacing)
                x_ticks.append(origin_center)

                # For each quarter within this origin
                for q_idx, quarter in enumerate(QUARTER_ORDER):
                    # Filter data for this origin and quarter
                    quarter_data = top5_data[
                        (top5_data["ORIGIN_PRCDCSD"] == top5_origins[origin_idx]) &
                        (top5_data["Quarter"] == quarter)
                    ]

                    # Calculate the starting position for this quarter's group of bars with minimal spacing
                    quarter_center = origin_center - (group_width/2) + (q_idx * effective_quarter_width) + (q_idx * quarter_spacing * group_width / n_quarters) + (effective_quarter_width / 2)

                    # Store quarter center position for labels
                    quarter_positions[q_idx].append(quarter_center)

                    # Add each metric bar for this quarter
                    for m_idx, metric in enumerate(["Nights", "Trips", "Visitor"]):
                        metric_pos = quarter_center - (effective_quarter_width / 2) + (m_idx + 0.5) * bar_width

                        # Get the value (default to 0 if no data for this quarter)
                        value = 0
                        if not quarter_data.empty and metric in quarter_data.columns:
                            value = quarter_data[metric].sum()

                        # Create the bar with increased width
                        metric_display = "Visitors" if metric == "Visitor" else metric
                        bar = ax.bar(
                            metric_pos, value, bar_width * 0.95,  # Increased bar width utilization
                            color=colors[metric_display],
                            label=f"{quarter} {metric_display}" if origin_idx == 0 and q_idx == 0 else "_nolegend_"
                        )
                        all_bars.append((bar, value, metric_display))

            # Add value labels to bars - now showing all values regardless of size
            for bar_obj, value, metric_name in all_bars:
                if value > 0:
                    height = bar_obj[0].get_height()
                    if height >= 1_000_000:
                        label = f"{height/1_000_000:.1f}M"
                    elif height >= 1_000:
                        label = f"{height/1_000:.0f}K"
                    else:
                        label = f"{height:.0f}"

                    ax.text(
                        bar_obj[0].get_x() + bar_obj[0].get_width() / 2,
                        height * 1.02,
                        label,
                        ha='center', va='bottom',
                        fontsize=9, rotation=90,
                        fontweight='bold'
                    )

            # Remove existing x-ticks (city names will be moved to top)
            ax.set_xticks([])

            # Add city names at the top of each group
            for i, (origin_center, origin_name) in enumerate(zip(x_ticks, origin_names)):
                ax.text(origin_center, ax.get_ylim()[1] * 1.02, origin_name,
                        ha='center', va='bottom',
                        fontsize=14, fontweight='bold',  # Increased font size
                        color='#222222')

            # Add quarter labels under each group of quarterly bars with larger font
            for q_idx, quarter in enumerate(QUARTER_ORDER):
                for i, origin_idx in enumerate(range(n_origins)):
                    quarter_center = quarter_positions[q_idx][origin_idx]
                    ax.text(quarter_center, ax.get_ylim()[0] * 0.02, quarter,
                            ha='center', va='top',
                            fontsize=14, fontweight='bold', color='#444444')  # Increased font size and made bold

            # Add a legend for metrics only (not quarters)
            metric_patches = [
                plt.Rectangle((0, 0), 1, 1, color=colors["Nights"], label="Nights"),
                plt.Rectangle((0, 0), 1, 1, color=colors["Trips"], label="Trips"),
                plt.Rectangle((0, 0), 1, 1, color=colors["Visitors"], label="Visitors")
            ]
            ax.legend(handles=metric_patches, loc='upper right', fontsize=12)

            # Set title and labels with increased padding at the top
            ax.set_title(f"Top 5 Cities Visiting {dmo_display} by Quarter - All Cities",
                        fontsize=18, fontweight="bold", pad=30)  # Increased title font size

            # Remove existing y-tick labels
            ax.set_yticklabels([])

            # Add grid lines for better readability
            ax.yaxis.grid(True, linestyle='--', alpha=0.7)

            # Adjust layout with more space at the top for the title
            plt.tight_layout(rect=[0, 0.08, 1, 0.90])  # Adjusted margins for better fit

            # Save the figure
            safe_name = dmo_display.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')
            out_path = os.path.join(output_dir, f"{safe_name}_top5_origins.png")
            plt.savefig(out_path, bbox_inches='tight')
            plt.close(fig)

            # PLOT 2: OUT-OF-PROVINCE ORIGINS
            # Filter to only include out-of-province origins
            out_of_prov_origins = []
            for origin_code in grouped.index:
                origin_province = origin_province_map.get(origin_code)
                if origin_province and origin_province != dest_province:
                    out_of_prov_origins.append(origin_code)

            # If we have out-of-province origins, create the second plot
            if out_of_prov_origins:
                # Filter and get top 5 out-of-province origins
                out_of_prov_df = grouped.loc[out_of_prov_origins]
                top5_outprov_origins = out_of_prov_df.sort_values(by="Visitor", ascending=False).head(5).index.tolist()

                # Filter data for just these top 5 out-of-province origins and include quarter information
                top5_outprov_data = dmo_df_sub[dmo_df_sub["ORIGIN_PRCDCSD"].isin(top5_outprov_origins)]

                if not top5_outprov_data.empty and "Quarter" in top5_outprov_data.columns:
                    # Map origin codes to names
                    outprov_names = [origin_map.get(code, str(code)) for code in top5_outprov_origins]

                    # Create figure with same dimensions as the first plot
                    fig, ax = plt.subplots(figsize=(16, 8), dpi=150)

                    # Setup plot parameters (same as first plot)
                    n_origins = len(outprov_names)

                    # Reset tracking variables
                    x_ticks = []
                    quarter_positions = [[] for _ in range(n_quarters)]
                    all_bars = []

                    # Loop through out-of-province origins and create grouped bars (SAME LOGIC AS FIRST PLOT)
                    for i, origin_idx in enumerate(range(n_origins)):
                        # Center position for this origin group - same logic as first plot
                        origin_center = i * (group_width + city_spacing)
                        x_ticks.append(origin_center)

                        # For each quarter within this origin
                        for q_idx, quarter in enumerate(QUARTER_ORDER):
                            # Filter data for this origin and quarter
                            quarter_data = top5_outprov_data[
                                (top5_outprov_data["ORIGIN_PRCDCSD"] == top5_outprov_origins[origin_idx]) &
                                (top5_outprov_data["Quarter"] == quarter)
                            ]

                            # Calculate the starting position for this quarter's group of bars - SAME LOGIC AS FIRST PLOT
                            quarter_center = origin_center - (group_width/2) + (q_idx * effective_quarter_width) + (q_idx * quarter_spacing * group_width / n_quarters) + (effective_quarter_width / 2)

                            # Store quarter center position for labels
                            quarter_positions[q_idx].append(quarter_center)

                            # Add each metric bar for this quarter
                            for m_idx, metric in enumerate(["Nights", "Trips", "Visitor"]):
                                metric_pos = quarter_center - (effective_quarter_width / 2) + (m_idx + 0.5) * bar_width

                                # Get the value (default to 0 if no data for this quarter)
                                value = 0
                                if not quarter_data.empty and metric in quarter_data.columns:
                                    value = quarter_data[metric].sum()

                                # Create the bar
                                metric_display = "Visitors" if metric == "Visitor" else metric
                                bar = ax.bar(
                                    metric_pos, value, bar_width * 0.95,
                                    color=colors[metric_display],
                                    label=f"{quarter} {metric_display}" if origin_idx == 0 and q_idx == 0 else "_nolegend_"
                                )
                                all_bars.append((bar, value, metric_display))

                    # Add value labels to bars - same as first plot
                    for bar_obj, value, metric_name in all_bars:
                        if value > 0:
                            height = bar_obj[0].get_height()
                            if height >= 1_000_000:
                                label = f"{height/1_000_000:.1f}M"
                            elif height >= 1_000:
                                label = f"{height/1_000:.0f}K"
                            else:
                                label = f"{height:.0f}"

                            ax.text(
                                bar_obj[0].get_x() + bar_obj[0].get_width() / 2,
                                height * 1.02,
                                label,
                                ha='center', va='bottom',
                                fontsize=9, rotation=90,
                                fontweight='bold'
                            )

                    # Remove existing x-ticks (city names will be moved to top)
                    ax.set_xticks([])

                    # Add city names at the top of each group
                    for i, (origin_center, origin_name) in enumerate(zip(x_ticks, outprov_names)):
                        ax.text(origin_center, ax.get_ylim()[1] * 1.02, origin_name,
                                ha='center', va='bottom',
                                fontsize=14, fontweight='bold',  # Increased font size
                                color='#222222')

                    # Add quarter labels under each group of quarterly bars with larger font
                    for q_idx, quarter in enumerate(QUARTER_ORDER):
                        for i, origin_idx in enumerate(range(n_origins)):
                            quarter_center = quarter_positions[q_idx][origin_idx]
                            ax.text(quarter_center, ax.get_ylim()[0] * 0.02, quarter,
                                    ha='center', va='top',
                                    fontsize=14, fontweight='bold', color='#444444')  # Increased font size and made bold

                    # Add a legend for metrics only (not quarters)
                    metric_patches = [
                        plt.Rectangle((0, 0), 1, 1, color=colors["Nights"], label="Nights"),
                        plt.Rectangle((0, 0), 1, 1, color=colors["Trips"], label="Trips"),
                        plt.Rectangle((0, 0), 1, 1, color=colors["Visitors"], label="Visitors")
                    ]
                    ax.legend(handles=metric_patches, loc='upper right', fontsize=12)

                    # Set title and labels with increased padding at the top
                    ax.set_title(f"Top 5 Cities Visiting {dmo_display} by Quarter - Out-of-Province Only",
                                fontsize=18, fontweight="bold", pad=30)  # Increased title font size

                    # Remove existing y-tick labels
                    ax.set_yticklabels([])

                    # Add grid lines for better readability
                    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

                    # Adjust layout with more space at the top for the title
                    plt.tight_layout(rect=[0, 0.08, 1, 0.90])  # Adjusted margins for better fit

                    # Save the figure
                    safe_name = dmo_display.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')
                    outprov_path = os.path.join(output_dir, f"{safe_name}_top5_outofprov_origins.png")
                    plt.savefig(outprov_path, bbox_inches='tight')
                    plt.close(fig)
                else:
                    print(f"[DEBUG] No quarterly data found for out-of-province origins for {dmo_display}")
            else:
                print(f"[DEBUG] No out-of-province origins found for {dmo_display}")
        else:
            print(f"[DEBUG] No quarterly data found for {dmo_display}")

    print("[DEBUG] Exiting generate_grouped_barplot")


def generate_province_pie_chart(dmo_df_sub, origin_province_map, title=None, custom_palette=None):
    """
    Generate a pie chart showing visitor percentages by province.

    Args:
        dmo_df_sub: DataFrame with visitor data for a specific DMO
        origin_province_map: Dictionary mapping origin codes to provinces
        title: Optional chart title
        custom_palette: Custom color palette to use for pie slices

    Returns:
        Figure object for the pie chart
    """
    # Calculate province totals
    province_totals = {}
    for origin_code in dmo_df_sub["ORIGIN_PRCDCSD"].unique():
        province = origin_province_map.get(origin_code, "Unknown")
        visitors = dmo_df_sub[dmo_df_sub["ORIGIN_PRCDCSD"] == origin_code]["Visitor"].sum()
        province_totals[province] = province_totals.get(province, 0) + visitors

    # Filter provinces with less than 1% and group into "Other"
    total_visitors = sum(province_totals.values())
    pie_data = []
    other_total = 0

    for province, visitors in province_totals.items():
        percentage = (visitors / total_visitors) * 100
        if percentage >= 1.0:  # Only show provinces with >= 1%
            pie_data.append((province, percentage))
        else:
            other_total += percentage

    if other_total > 0:
        pie_data.append(("Other", other_total))

    # Sort by percentage (descending)
    pie_data.sort(key=lambda x: x[1], reverse=True)

    # Create the pie chart
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200, facecolor='white')

    labels = [item[0] for item in pie_data]
    values = [item[1] for item in pie_data]

    # Use get_color_palette function for consistent color management
    colors = get_color_palette(custom_palette, len(values))

    # Slight explosion for the largest slice
    explode = [0.03] + [0] * (len(values) - 1)

    # Create pie chart without labels (labels=None removes segment labels)
    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,  # Remove direct labels on segments
        colors=colors,
        autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',
        startangle=90,
        explode=explode,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )

    # Enhance the percentage labels
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(14)

    # Create legend with province names and percentages
    legend_labels = [f"{label} ({value:.1f}%)" for label, value in pie_data]
    ax.legend(wedges, legend_labels, title="Provinces", loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1), fontsize=12, title_fontsize=14)

    # Equal aspect ratio ensures circle
    ax.axis('equal')

    # Set title
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20, color='#333333')

    # Add subtle border
    fig.patch.set_linewidth(1)
    fig.patch.set_edgecolor('#dddddd')

    return fig

def generate_nights_per_visitor_by_quarter(dmo_df_sub, origin_province_map, dest_province, title, output_path, custom_palette=None):
    """
    Generate a bar chart showing nights per visitor by quarter for in-province vs out-of-province visitors.

    Args:
        dmo_df_sub: DataFrame with visitor data for a specific DMO
        origin_province_map: Dictionary mapping origin codes to provinces
        dest_province: The province where the DMO is located
        title: Chart title
        output_path: Path to save the chart
        custom_palette: Custom color palette to use for bars
    """
    # Calculate nights per visitor by quarter and province type
    quarterly_data = []

    for quarter in QUARTER_ORDER:
        quarter_data = dmo_df_sub[dmo_df_sub["Quarter"] == quarter]

        if quarter_data.empty:
            quarterly_data.append({
                "Quarter": quarter,
                "In-Province": 0,
                "Out-of-Province": 0
            })
            continue

        # Separate in-province and out-of-province data
        in_province_data = pd.DataFrame()
        out_province_data = pd.DataFrame()

        for origin_code in quarter_data["ORIGIN_PRCDCSD"].unique():
            province = origin_province_map.get(origin_code, "Unknown")
            origin_data = quarter_data[quarter_data["ORIGIN_PRCDCSD"] == origin_code]

            # Determine if this is in-province or out-of-province using the actual destination province
            if province == dest_province:
                in_province_data = pd.concat([in_province_data, origin_data], ignore_index=True)
            else:
                out_province_data = pd.concat([out_province_data, origin_data], ignore_index=True)

        # Calculate nights per visitor for each group
        in_province_nights_per_visitor = 0
        out_province_nights_per_visitor = 0

        if not in_province_data.empty:
            total_nights = in_province_data["Nights"].sum()
            total_visitors = in_province_data["Visitor"].sum()
            in_province_nights_per_visitor = total_nights / total_visitors if total_visitors > 0 else 0

        if not out_province_data.empty:
            total_nights = out_province_data["Nights"].sum()
            total_visitors = out_province_data["Visitor"].sum()
            out_province_nights_per_visitor = total_nights / total_visitors if total_visitors > 0 else 0

        quarterly_data.append({
            "Quarter": quarter,
            "In-Province": in_province_nights_per_visitor,
            "Out-of-Province": out_province_nights_per_visitor
        })

    # Create the chart
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Prepare data for plotting
    quarters = [item["Quarter"] for item in quarterly_data]
    in_province_values = [item["In-Province"] for item in quarterly_data]
    out_province_values = [item["Out-of-Province"] for item in quarterly_data]

    # Set up bar positions
    x = np.arange(len(quarters))
    width = 0.35

    # Get colors from custom palette or use defaults
    colors_list = get_color_palette(custom_palette, 2)

    # Create bars with proper labels using the actual province name
    bars1 = ax.bar(x - width/2, in_province_values, width,
                   label=dest_province, color=colors_list[0], alpha=0.8)
    bars2 = ax.bar(x + width/2, out_province_values, width,
                   label='Out-of-Province', color=colors_list[1], alpha=0.8)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                       f'{height:.1f}', ha='center', va='bottom', fontweight='bold')

    # Configure chart appearance
    ax.set_xlabel('Quarter', fontsize=12, fontweight='bold')
    ax.set_ylabel('Nights per Visitor', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=60)  # Increased pad for more legend space
    ax.set_xticks(x)
    ax.set_xticklabels(quarters)

    # Add legend well above the plot area to avoid any conflicts
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=True, framealpha=0.9)

    # Style the chart
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_facecolor('#F8FAFC')
    fig.patch.set_facecolor('#F8FAFC')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save the chart
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"[DEBUG] Generated nights per visitor chart: {output_path}")

def generate_circular_barplot(vtn_csv_path, profile_csv_path, dmo_csv_path, output_dir, year=None, comp_year=None, destination_province=None, custom_palette=None):
    """
    Generate pie charts showing visitor distribution by PRIZM & Traveller Segmentation Program.
    Creates pie charts for each DMO destination, with one chart for the main year and another for the comparison year if provided.

    Args:
        vtn_csv_path: Path to VTN CSV file
        profile_csv_path: Path to Profile CSV file with PRIZM_CODE mappings
        dmo_csv_path: Path to DMO CSV file
        output_dir: Directory to save output files
        year: Main year to analyze
        comp_year: Comparison year (optional)
        destination_province: User-selected destination province
        custom_palette: Custom color palette to use for segments

    """
    print("[DEBUG] Entering generate_circular_barplot")

    try:
        # Read Profile.csv first to get PRIZM_CODE mappings (small file)
        # Try different encodings to handle special characters
        profile_df = None
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings_to_try:
            try:
                profile_df = pd.read_csv(profile_csv_path, encoding=encoding)
                print(f"[DEBUG] Successfully loaded Profile.csv with {encoding} encoding")
                break
            except UnicodeDecodeError:
                print(f"[DEBUG] Failed to read Profile.csv with {encoding} encoding, trying next...")
                continue

        if profile_df is None:
            print("[ERROR] Failed to read Profile.csv with any encoding")
            return

        prizm_map = dict(zip(profile_df["PRIZM_CODE"], profile_df["EQ"]))  # Use EQ column for traveller segmentation
        print(f"[DEBUG] Loaded Profile.csv with {len(profile_df)} PRIZM codes")

        # Read DMO.csv (small file)
        dmo_df = pd.read_csv(dmo_csv_path)
        dmo_df['DEST_CODE'] = pd.to_numeric(dmo_df['DEST_CODE'], errors='coerce').astype('Int64')
        print(f"[DEBUG] Loaded DMO.csv with {len(dmo_df)} destinations")

        # Read VTN.csv efficiently - only load required columns and filter early
        print("[DEBUG] Reading VTN file efficiently...")
        required_columns = ['date', 'Year', 'DMO', 'PRIZM_CODE', 'Visitor']

        # Read with chunking to handle large files
        chunk_size = 100000  # Process 100k rows at a time
        vtn_chunks = []

        for chunk in pd.read_csv(vtn_csv_path, chunksize=chunk_size):
            print(f"[DEBUG] Processing chunk with {len(chunk)} rows")

            # Parse date and extract year if needed
            if "date" in chunk.columns and "Year" not in chunk.columns:
                chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
                chunk["Year"] = chunk["date"].dt.year.astype("Int64")

            # Convert data types early
            chunk['DMO'] = pd.to_numeric(chunk['DMO'], errors='coerce').astype('Int64')
            chunk['PRIZM_CODE'] = pd.to_numeric(chunk['PRIZM_CODE'], errors='coerce').astype('Int64')
            chunk['Visitor'] = pd.to_numeric(chunk['Visitor'], errors='coerce').fillna(0)

            # Filter for required years early to reduce memory usage
            years_to_keep = [year]
            if comp_year:
                years_to_keep.append(comp_year)

            chunk_filtered = chunk[chunk['Year'].isin(years_to_keep)]

            # Only keep rows with valid PRIZM_CODE and non-zero visitors
            chunk_filtered = chunk_filtered[
                (chunk_filtered['PRIZM_CODE'].notna()) &
                (chunk_filtered['Visitor'] > 0) &
                (chunk_filtered['DMO'].notna())
            ]

            if not chunk_filtered.empty:
                # Only keep required columns to save memory
                cols_to_keep = [col for col in required_columns if col in chunk_filtered.columns]
                vtn_chunks.append(chunk_filtered[cols_to_keep])

            # Clear chunk from memory
            del chunk
            del chunk_filtered

        # Combine all chunks
        if not vtn_chunks:
            print("[DEBUG] No valid data found after filtering")
            return

        print(f"[DEBUG] Combining {len(vtn_chunks)} chunks...")
        vtn_df = pd.concat(vtn_chunks, ignore_index=True)
        del vtn_chunks  # Free memory

        print(f"[DEBUG] Final filtered VTN data: {len(vtn_df)} rows")

        # Process each DMO destination separately
        for _, dmo_row in dmo_df.iterrows():
            dmo_code = dmo_row['DEST_CODE']
            dmo_display = " ".join(str(dmo_row['DMO']).split()) if 'DMO' in dmo_row else str(dmo_row['DEST_NAME'])

            print(f"[DEBUG] Processing DMO: {dmo_display} (Code: {dmo_code})")

            dmo_vtn_data = vtn_df[vtn_df['DMO'] == dmo_code]

            if dmo_vtn_data.empty:
                print(f"[DEBUG] No data found for DMO {dmo_display}")
                continue

            safe_dmo_name = dmo_display.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')

            years_to_process = [year]
            if comp_year:
                years_to_process.append(comp_year)

            for current_year in years_to_process:
                if current_year is None:
                    continue

                print(f"[DEBUG] Processing year {current_year} for DMO {dmo_display}")

                year_data = dmo_vtn_data[dmo_vtn_data['Year'] == current_year]

                if year_data.empty:
                    print(f"[DEBUG] No data found for year {current_year} and DMO {dmo_display}")
                    continue

                print(f"[DEBUG] Aggregating {len(year_data)} rows by PRIZM_CODE for {dmo_display}...")
                prizm_totals = year_data.groupby('PRIZM_CODE', as_index=False)['Visitor'].sum()

                prizm_totals['Segment'] = prizm_totals['PRIZM_CODE'].map(prizm_map)

                prizm_totals = prizm_totals[prizm_totals['Segment'].notna()]

                if prizm_totals.empty:
                    print(f"[DEBUG] No valid segments found for year {current_year} and DMO {dmo_display}")
                    continue

                segment_totals = prizm_totals.groupby('Segment', as_index=False)['Visitor'].sum()
                segment_totals = segment_totals.sort_values('Visitor', ascending=False)

                segment_totals = segment_totals.head(5)

                print(f"[DEBUG] Found {len(segment_totals)} segments for year {current_year} and DMO {dmo_display}")

                # Create pie chart instead of circular barplot
                fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

                segments = segment_totals['Segment'].tolist()
                values = segment_totals['Visitor'].tolist()

                if not segments:
                    print(f"[DEBUG] No segments to plot for year {current_year} and DMO {dmo_display}")
                    plt.close(fig)
                    continue

                # Get colors from custom palette or use defaults
                colors = get_color_palette(custom_palette, len(segments))

                # Create pie chart with slight explosion for the largest slice
                explode = [0.05] + [0] * (len(values) - 1)

                # Create the pie chart
                wedges, texts, autotexts = ax.pie(
                    values,
                    labels=segments,
                    colors=colors,
                    autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',
                    startangle=90,
                    explode=explode,
                    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
                    textprops={'fontsize': 16, 'fontweight': 'bold'}  # Increased font size from 11 to 16
                )

                # Enhance the percentage labels
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(14)  # Increased from 12 to 14

                # Removed the boxed labels section that was using ax.annotate()

                # Equal aspect ratio ensures circle
                ax.axis('equal')

                # Set title
                ax.set_title(f"{dmo_display} - Visitors by Traveller Segmentation Program - {current_year}",
                            fontsize=16, fontweight='bold', pad=20, color='#2C3E50')

                # Set clean background
                ax.set_facecolor('white')
                fig.patch.set_facecolor('white')

                # Save the plot with DMO-specific filename
                output_filename = f"{safe_dmo_name}_prizm_pie_chart_{current_year}.png"
                output_path = os.path.join(output_dir, output_filename)
                plt.tight_layout()
                plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig)

                print(f"[DEBUG] Generated pie chart for {dmo_display} {current_year}: {output_path}")

    except Exception as e:
        print(f"[ERROR] Error in generate_circular_barplot: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    print("[DEBUG] Exiting generate_circular_barplot")

def generate_prizm_barplot(vtn_csv_path, profile_csv_path, dmo_csv_path, output_dir, year=None, comp_year=None, destination_province=None, custom_palette=None):
    """
    Generate ordinary barplots showing visitor distribution by PRIZM segments.
    Creates barplots for each DMO destination, with one chart for the main year and another for the comparison year if provided.

    Args:
        vtn_csv_path: Path to VTN CSV file
        profile_csv_path: Path to Profile CSV file with PRIZM_CODE mappings
        dmo_csv_path: Path to DMO CSV file
        output_dir: Directory to save output files
        year: Main year to analyze
        comp_year: Comparison year (optional)
        destination_province: User-selected destination province
        custom_palette: Custom color palette to use for segments
    """
    print("[DEBUG] Entering generate_prizm_barplot")

    try:
        # Read Profile.csv first to get PRIZM_CODE mappings (small file)
        # Try different encodings to handle special characters
        profile_df = None
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings_to_try:
            try:
                profile_df = pd.read_csv(profile_csv_path, encoding=encoding)
                print(f"[DEBUG] Successfully loaded Profile.csv with {encoding} encoding")
                break
            except UnicodeDecodeError:
                print(f"[DEBUG] Failed to read Profile.csv with {encoding} encoding, trying next...")
                continue

        if profile_df is None:
            print("[ERROR] Failed to read Profile.csv with any encoding")
            return

        prizm_map = dict(zip(profile_df["PRIZM_CODE"], profile_df["PRIZM"]))  # Use PRIZM column for segments
        print(f"[DEBUG] Loaded Profile.csv with {len(profile_df)} PRIZM codes")

        # Read DMO.csv (small file)
        dmo_df = pd.read_csv(dmo_csv_path)
        dmo_df['DEST_CODE'] = pd.to_numeric(dmo_df['DEST_CODE'], errors='coerce').astype('Int64')
        print(f"[DEBUG] Loaded DMO.csv with {len(dmo_df)} destinations")

        # Read VTN.csv efficiently - only load required columns and filter early
        print("[DEBUG] Reading VTN file efficiently...")
        required_columns = ['date', 'Year', 'DMO', 'PRIZM_CODE', 'Visitor']

        # Read with chunking to handle large files
        chunk_size = 100000  # Process 100k rows at a time
        vtn_chunks = []

        for chunk in pd.read_csv(vtn_csv_path, chunksize=chunk_size):
            print(f"[DEBUG] Processing chunk with {len(chunk)} rows")

            # Parse date and extract year if needed
            if "date" in chunk.columns and "Year" not in chunk.columns:
                chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
                chunk["Year"] = chunk["date"].dt.year.astype("Int64")

            # Convert data types early
            chunk['DMO'] = pd.to_numeric(chunk['DMO'], errors='coerce').astype('Int64')
            chunk['PRIZM_CODE'] = pd.to_numeric(chunk['PRIZM_CODE'], errors='coerce').astype('Int64')
            chunk['Visitor'] = pd.to_numeric(chunk['Visitor'], errors='coerce').fillna(0)

            # Filter for required years early to reduce memory usage
            years_to_keep = [year]
            if comp_year:
                years_to_keep.append(comp_year)

            chunk_filtered = chunk[chunk['Year'].isin(years_to_keep)]

            # Only keep rows with valid PRIZM_CODE and non-zero visitors
            chunk_filtered = chunk_filtered[
                (chunk_filtered['PRIZM_CODE'].notna()) &
                (chunk_filtered['Visitor'] > 0) &
                (chunk_filtered['DMO'].notna())
            ]

            if not chunk_filtered.empty:
                # Only keep required columns to save memory
                cols_to_keep = [col for col in required_columns if col in chunk_filtered.columns]
                vtn_chunks.append(chunk_filtered[cols_to_keep])

            # Clear chunk from memory
            del chunk
            del chunk_filtered

        # Combine all chunks
        if not vtn_chunks:
            print("[DEBUG] No valid data found after filtering")
            return

        print(f"[DEBUG] Combining {len(vtn_chunks)} chunks...")
        vtn_df = pd.concat(vtn_chunks, ignore_index=True)
        del vtn_chunks  # Free memory

        print(f"[DEBUG] Final filtered VTN data: {len(vtn_df)} rows")

        # Process each DMO destination separately
        for _, dmo_row in dmo_df.iterrows():
            dmo_code = dmo_row['DEST_CODE']
            dmo_display = " ".join(str(dmo_row['DMO']).split()) if 'DMO' in dmo_row else str(dmo_row['DEST_NAME'])

            print(f"[DEBUG] Processing DMO: {dmo_display} (Code: {dmo_code})")

            # Filter VTN data for this specific DMO
            dmo_vtn_data = vtn_df[vtn_df['DMO'] == dmo_code]

            if dmo_vtn_data.empty:
                print(f"[DEBUG] No data found for DMO {dmo_display}")
                continue

            # Create safe filename for this DMO
            safe_dmo_name = dmo_display.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')

            # Filter for the specified years
            years_to_process = [year]
            if comp_year:
                years_to_process.append(comp_year)

            for current_year in years_to_process:
                if current_year is None:
                    continue

                print(f"[DEBUG] Processing year {current_year} for DMO {dmo_display}")

                # Filter data for current year and current DMO
                year_data = dmo_vtn_data[dmo_vtn_data['Year'] == current_year]

                if year_data.empty:
                    print(f"[DEBUG] No data found for year {current_year} and DMO {dmo_display}")
                    continue

                # Aggregate visitors by PRIZM_CODE for this DMO efficiently
                print(f"[DEBUG] Aggregating {len(year_data)} rows by PRIZM_CODE for {dmo_display}...")
                prizm_totals = year_data.groupby('PRIZM_CODE', as_index=False)['Visitor'].sum()

                # Map PRIZM_CODE to PRIZM segments
                prizm_totals['Segment'] = prizm_totals['PRIZM_CODE'].map(prizm_map)

                # Remove rows where segment mapping failed
                prizm_totals = prizm_totals[prizm_totals['Segment'].notna()]

                if prizm_totals.empty:
                    print(f"[DEBUG] No valid PRIZM segments found for year {current_year} and DMO {dmo_display}")
                    continue

                # Group by PRIZM segment and sum visitors
                segment_totals = prizm_totals.groupby('Segment', as_index=False)['Visitor'].sum()
                segment_totals = segment_totals.sort_values('Visitor', ascending=False)

                # Keep only top 5 segments (same as circular barplots for consistency)
                segment_totals = segment_totals.head(5)

                print(f"[DEBUG] Found {len(segment_totals)} PRIZM segments for year {current_year} and DMO {dmo_display}")

                # Create ordinary barplot
                fig, ax = plt.subplots(figsize=(10, 8))

                # Prepare data for plot
                segments = segment_totals['Segment'].tolist()
                values = segment_totals['Visitor'].tolist()

                if not segments:
                    print(f"[DEBUG] No segments to plot for year {current_year} and DMO {dmo_display}")
                    plt.close(fig)
                    continue

                # Create bars
                bars = ax.bar(range(len(segments)), values, alpha=0.8)

                # Get colors from custom palette or use defaults
                colors = get_color_palette(custom_palette, len(segments))

                # Apply colors to bars
                for bar, color in zip(bars, colors[:len(bars)]):
                    bar.set_facecolor(color)
                    bar.set_edgecolor('white')
                    bar.set_linewidth(0.8)

                # Add value labels on bars
                for i, (bar, value) in enumerate(zip(bars, values)):
                    # Format large numbers
                    if value >= 1_000_000:
                        label = f"{value/1_000_000:.1f}M"
                    elif value >= 1_000:
                        label = f"{value/1_000:.0f}K"
                    else:
                        label = f"{value:.0f}"

                    # Position label at the top of the bar
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                           label, ha='center', va='bottom', fontsize=13, fontweight='bold')

                # Customize the chart
                ax.set_title(f"{dmo_display} - Visitors by PRIZM Segments - {current_year}",
                            fontsize=16, fontweight='bold', pad=20, color='#2C3E50')

                ax.set_xlabel('PRIZM Segments', fontsize=12, fontweight='bold')
                ax.set_ylabel('Number of Visitors', fontsize=12, fontweight='bold')

                # Set x-axis labels with rotation for better readability
                ax.set_xticks(range(len(segments)))
                ax.set_xticklabels(segments, rotation=45, ha='right', fontsize=10)

                # Format y-axis
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K' if x >= 1000 else f'{x:.0f}'))

                # Add grid for better readability
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_axisbelow(True)

                # Set clean background
                ax.set_facecolor('white')
                fig.patch.set_facecolor('white')

                # Adjust layout to prevent label cutoff
                plt.tight_layout()

                # Save the plot with DMO-specific filename
                output_filename = f"{safe_dmo_name}_prizm_barplot_{current_year}.png"
                output_path = os.path.join(output_dir, output_filename)
                plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig)

                print(f"[DEBUG] Generated PRIZM barplot for {dmo_display} {current_year}: {output_path}")

    except Exception as e:
        print(f"[ERROR] Error in generate_prizm_barplot: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    print("[DEBUG] Exiting generate_prizm_barplot")


def generate_prizm_comparison_barplot(vtn_csv_path, profile_csv_path, dmo_csv_path, output_dir, year=None, comp_year=None, destination_province=None, custom_palette=None):
    """
    Generate side-by-side comparison barplots showing visitor distribution by PRIZM segments for two years.
    Creates comparison barplots for each DMO destination when both years are provided.

    Args:
        vtn_csv_path: Path to VTN CSV file
        profile_csv_path: Path to Profile CSV file with PRIZM_CODE mappings
        dmo_csv_path: Path to DMO CSV file
        output_dir: Directory to save output files
        year: Main year to analyze
        comp_year: Comparison year
        destination_province: User-selected destination province
        custom_palette: Custom color palette to use for segments
    """
    print("[DEBUG] Entering generate_prizm_comparison_barplot")

    if not comp_year:
        print("[DEBUG] No comparison year provided, skipping comparison barplot")
        return

    try:
        # Read Profile.csv first to get PRIZM_CODE mappings (small file)
        profile_df = None
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings_to_try:
            try:
                profile_df = pd.read_csv(profile_csv_path, encoding=encoding)
                print(f"[DEBUG] Successfully loaded Profile.csv with {encoding} encoding")
                break
            except UnicodeDecodeError:
                print(f"[DEBUG] Failed to read Profile.csv with {encoding} encoding, trying next...")
                continue

        if profile_df is None:
            print("[ERROR] Failed to read Profile.csv with any encoding")
            return

        prizm_map = dict(zip(profile_df["PRIZM_CODE"], profile_df["PRIZM"]))  # Use PRIZM column for segments
        print(f"[DEBUG] Loaded Profile.csv with {len(profile_df)} PRIZM codes")

        # Read DMO.csv (small file)
        dmo_df = pd.read_csv(dmo_csv_path)
        dmo_df['DEST_CODE'] = pd.to_numeric(dmo_df['DEST_CODE'], errors='coerce').astype('Int64')
        print(f"[DEBUG] Loaded DMO.csv with {len(dmo_df)} destinations")

        # Read VTN.csv efficiently
        print("[DEBUG] Reading VTN file efficiently...")
        required_columns = ['date', 'Year', 'DMO', 'PRIZM_CODE', 'Visitor']

        # Read with chunking to handle large files
        chunk_size = 100000
        vtn_chunks = []

        for chunk in pd.read_csv(vtn_csv_path, chunksize=chunk_size):
            print(f"[DEBUG] Processing chunk with {len(chunk)} rows")

            # Parse date and extract year if needed
            if "date" in chunk.columns and "Year" not in chunk.columns:
                chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
                chunk["Year"] = chunk["date"].dt.year.astype("Int64")

            # Convert data types early
            chunk['DMO'] = pd.to_numeric(chunk['DMO'], errors='coerce').astype('Int64')
            chunk['PRIZM_CODE'] = pd.to_numeric(chunk['PRIZM_CODE'], errors='coerce').astype('Int64')
            chunk['Visitor'] = pd.to_numeric(chunk['Visitor'], errors='coerce').fillna(0)

            # Filter for required years
            years_to_keep = [year, comp_year]
            chunk_filtered = chunk[chunk['Year'].isin(years_to_keep)]

            # Only keep rows with valid PRIZM_CODE and non-zero visitors
            chunk_filtered = chunk_filtered[
                (chunk_filtered['PRIZM_CODE'].notna()) &
                (chunk_filtered['Visitor'] > 0) &
                (chunk_filtered['DMO'].notna())
            ]

            if not chunk_filtered.empty:
                cols_to_keep = [col for col in required_columns if col in chunk_filtered.columns]
                vtn_chunks.append(chunk_filtered[cols_to_keep])

            del chunk
            del chunk_filtered

        # Combine all chunks
        if not vtn_chunks:
            print("[DEBUG] No valid data found after filtering")
            return

        print(f"[DEBUG] Combining {len(vtn_chunks)} chunks...")
        vtn_df = pd.concat(vtn_chunks, ignore_index=True)
        del vtn_chunks

        print(f"[DEBUG] Final filtered VTN data: {len(vtn_df)} rows")

        # Process each DMO destination separately
        for _, dmo_row in dmo_df.iterrows():
            dmo_code = dmo_row['DEST_CODE']
            dmo_display = " ".join(str(dmo_row['DMO']).split()) if 'DMO' in dmo_row else str(dmo_row['DEST_NAME'])

            print(f"[DEBUG] Processing DMO: {dmo_display} (Code: {dmo_code})")

            # Filter VTN data for this specific DMO
            dmo_vtn_data = vtn_df[vtn_df['DMO'] == dmo_code]

            if dmo_vtn_data.empty:
                print(f"[DEBUG] No data found for DMO {dmo_display}")
                continue

            # Create safe filename for this DMO
            safe_dmo_name = dmo_display.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')

            # Create side-by-side comparison barplots for the two years
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle(f"{dmo_display} - PRIZM Segments Comparison: {comp_year} vs {year}",
                        fontsize=18, fontweight='bold', y=0.95)

            # Process both years
            for idx, (current_year, ax) in enumerate([(comp_year, ax1), (year, ax2)]):
                # Filter data for current year and current DMO
                year_data = dmo_vtn_data[dmo_vtn_data['Year'] == current_year]

                if year_data.empty:
                    ax.text(0.5, 0.5, f"No data\nfor {current_year}",
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=14, style='italic')
                    ax.set_title(f"{current_year}", fontsize=16, fontweight='bold')
                    continue

                # Aggregate visitors by PRIZM_CODE
                prizm_totals = year_data.groupby('PRIZM_CODE', as_index=False)['Visitor'].sum()

                # Map PRIZM_CODE to PRIZM segments
                prizm_totals['Segment'] = prizm_totals['PRIZM_CODE'].map(prizm_map)
                prizm_totals = prizm_totals[prizm_totals['Segment'].notna()]

                if prizm_totals.empty:
                    ax.text(0.5, 0.5, f"No valid segments\nfor {current_year}",
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=14, style='italic')
                    ax.set_title(f"{current_year}", fontsize=16, fontweight='bold')
                    continue

                # Group by PRIZM segment and sum visitors
                segment_totals = prizm_totals.groupby('Segment', as_index=False)['Visitor'].sum()
                segment_totals = segment_totals.sort_values('Visitor', ascending=False)

                # Keep only top 5 segments
                segment_totals = segment_totals.head(5)

                # Prepare data for barplot
                segments = segment_totals['Segment'].tolist()
                values = segment_totals['Visitor'].tolist()

                if not segments:
                    ax.text(0.5, 0.5, f"No segments to plot for {current_year}",
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=14, style='italic')
                    continue

                # Get colors from custom palette or use defaults, with variations for each year
                if custom_palette:
                    base_colors = get_color_palette(custom_palette, len(segments))
                    if idx == 0:  # Comparison year - use slightly darker variants
                        colors = [f"{color}CC" if len(color) == 7 else color for color in base_colors]  # Add transparency
                    else:  # Main year - use original colors
                        colors = base_colors
                else:
                    # Default color scheme - different colors for each year
                    colors = ['#2E86C1', '#F39C12', '#E74C3C', '#27AE60', '#8E44AD']
                    if idx == 0:  # Comparison year - use cooler colors
                        colors = ['#3498DB', '#E67E22', '#E74C3C', '#2ECC71', '#9B59B6']
                    else:  # Main year - use warmer colors
                        colors = ['#2980B9', '#D35400', '#C0392B', '#27AE60', '#8E44AD']

                # Create bars
                bars = ax.bar(range(len(segments)), values, alpha=0.8)

                # Apply colors to bars
                for bar, color in zip(bars, colors[:len(bars)]):
                    bar.set_facecolor(color)
                    bar.set_edgecolor('white')
                    bar.set_linewidth(0.5)

                # Add value labels on bars
                for i, (bar, value) in enumerate(zip(bars, values)):
                    if value > 0:
                        # Format large numbers
                        if value >= 1_000_000:
                            label = f"{value/1_000_000:.1f}M"
                        elif value >= 1_000:
                            label = f"{value/1_000:.0f}K"
                        else:
                            label = f"{value:.0f}"

                        # Position label at the top of the bar
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                               label, ha='center', va='bottom', fontsize=10, fontweight='bold')

                # Customize the chart
                ax.set_title(f"{current_year}", fontsize=16, fontweight='bold')
                ax.set_xticks(range(len(segments)))
                ax.set_xticklabels(segments, rotation=45, ha='right', fontsize=10)
                ax.set_ylabel('Visitors', fontsize=12, fontweight='bold')

                # Format y-axis
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K' if x >= 1000 else f'{x:.0f}'))

                # Add grid for better readability
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_axisbelow(True)

                # Set clean background
                ax.set_facecolor('white')

            # Adjust layout and save
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)  # Make room for the main title

            # Save the combined plot
            output_filename = f"{safe_dmo_name}_prizm_comparison_{comp_year}_{year}.png"
            output_path = os.path.join(output_dir, output_filename)
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            print(f"[DEBUG] Generated PRIZM comparison barplots for { dmo_display}: {output_path}")

    except Exception as e:
        print(f"[ERROR] Error in generate_prizm_comparison_barplot: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    print("[DEBUG] Exiting generate_prizm_comparison_barplot")

def generate_quarterly_prizm_barplots(vtn_csv_path, profile_csv_path, dmo_csv_path, output_dir, year=None, comp_year=None, destination_province=None, custom_palette=None):
    """
    Generate quarterly comparison PRIZM barplots with 8 horizontal charts.
    Creates 2x4 grid: top row shows comparison year quarters, bottom row shows main year quarters.
    Each chart shows top 5 PRIZM segments as horizontal bars.

    Args:
        vtn_csv_path: Path to VTN CSV file
        profile_csv_path: Path to Profile CSV file with PRIZM_CODE mappings
        dmo_csv_path: Path to DMO CSV file
        output_dir: Directory to save output files
        year: Main year to analyze
        comp_year: Comparison year (required for this function)
        destination_province: User-selected destination province
        custom_palette: Custom color palette to use for segments
    """
    print("[DEBUG] Entering generate_quarterly_prizm_barplots")

    if not comp_year:
        print("[DEBUG] No comparison year provided, skipping quarterly PRIZM barplots")
        return

    try:
        # Read Profile.csv first to get PRIZM_CODE mappings (small file)
        profile_df = None
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings_to_try:
            try:
                profile_df = pd.read_csv(profile_csv_path, encoding=encoding)
                print(f"[DEBUG] Successfully loaded Profile.csv with {encoding} encoding")
                break
            except UnicodeDecodeError:
                print(f"[DEBUG] Failed to read Profile.csv with {encoding} encoding, trying next...")
                continue

        if profile_df is None:
            print("[ERROR] Failed to read Profile.csv with any encoding")
            return

        prizm_map = dict(zip(profile_df["PRIZM_CODE"], profile_df["PRIZM"]))  # Use PRIZM column for segments
        print(f"[DEBUG] Loaded Profile.csv with {len(profile_df)} PRIZM codes")

        # Read DMO.csv (small file)
        dmo_df = pd.read_csv(dmo_csv_path)
        dmo_df['DEST_CODE'] = pd.to_numeric(dmo_df['DEST_CODE'], errors='coerce').astype('Int64')
        print(f"[DEBUG] Loaded DMO.csv with {len(dmo_df)} destinations")

        # Read VTN.csv efficiently
        print("[DEBUG] Reading VTN file efficiently...")
        required_columns = ['date', 'Year', 'DMO', 'PRIZM_CODE', 'Visitor']

        # Read with chunking to handle large files
        chunk_size = 100000
        vtn_chunks = []

        for chunk in pd.read_csv(vtn_csv_path, chunksize=chunk_size):
            print(f"[DEBUG] Processing chunk with {len(chunk)} rows")

            # Parse date and extract year and quarter if needed
            if "date" in chunk.columns:
                chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
                if "Year" not in chunk.columns:
                    chunk["Year"] = chunk["date"].dt.year.astype("Int64")

                # Extract quarter information
                chunk["Month"] = chunk["date"].dt.month
                chunk["Quarter"] = chunk["Month"].apply(lambda m:
                    "Q1" if m <= 3 else
                    "Q2" if m <= 6 else
                    "Q3" if m <= 9 else "Q4"
                )

            # Convert data types early
            chunk['DMO'] = pd.to_numeric(chunk['DMO'], errors='coerce').astype('Int64')
            chunk['PRIZM_CODE'] = pd.to_numeric(chunk['PRIZM_CODE'], errors='coerce').astype('Int64')
            chunk['Visitor'] = pd.to_numeric(chunk['Visitor'], errors='coerce').fillna(0)

            # Filter for required years
            years_to_keep = [year, comp_year]
            chunk_filtered = chunk[chunk['Year'].isin(years_to_keep)]

            # Only keep rows with valid PRIZM_CODE and non-zero visitors
            chunk_filtered = chunk_filtered[
                (chunk_filtered['PRIZM_CODE'].notna()) &
                (chunk_filtered['Visitor'] > 0) &
                (chunk_filtered['DMO'].notna())
            ]

            if not chunk_filtered.empty:
                cols_to_keep = [col for col in required_columns + ['Quarter'] if col in chunk_filtered.columns]
                vtn_chunks.append(chunk_filtered[cols_to_keep])

            del chunk
            del chunk_filtered

        # Combine all chunks
        if not vtn_chunks:
            print("[DEBUG] No valid data found after filtering")
            return

        print(f"[DEBUG] Combining {len(vtn_chunks)} chunks...")
        vtn_df = pd.concat(vtn_chunks, ignore_index=True)
        del vtn_chunks

        print(f"[DEBUG] Final filtered VTN data: {len(vtn_df)} rows")

        # Process each DMO destination separately
        for _, dmo_row in dmo_df.iterrows():
            dmo_code = dmo_row['DEST_CODE']
            dmo_display = " ".join(str(dmo_row['DMO']).split()) if 'DMO' in dmo_row else str(dmo_row['DEST_NAME'])

            print(f"[DEBUG] Processing DMO: {dmo_display} (Code: {dmo_code})")

            # Filter VTN data for this specific DMO
            dmo_vtn_data = vtn_df[vtn_df['DMO'] == dmo_code]

            if dmo_vtn_data.empty:
                print(f"[DEBUG] No data found for DMO {dmo_display}")
                continue

            # Create safe filename for this DMO
            safe_dmo_name = dmo_display.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')

            # Create the combined quarterly chart with 8 horizontal barplots
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            fig.suptitle(f"{dmo_display} - Quarterly {comp_year} & {year} Visitors by PRIZM Segments",
                        fontsize=20, fontweight='bold', y=0.95)

            # Process each year and quarter combination
            for year_idx, current_year in enumerate([comp_year, year]):
                for quarter_idx, quarter in enumerate(QUARTER_ORDER):
                    ax = axes[year_idx, quarter_idx]

                    # Filter data for current year, DMO, and quarter
                    quarter_data = dmo_vtn_data[
                        (dmo_vtn_data['Year'] == current_year) &
                        (dmo_vtn_data['Quarter'] == quarter)
                    ]

                    if quarter_data.empty:
                        # If no data, create empty chart with message
                        ax.text(0.5, 0.5, f"No data\nfor {quarter} {current_year}",
                               ha='center', va='center', transform=ax.transAxes,
                               fontsize=12, style='italic')
                        ax.set_title(f"{quarter} {current_year}", fontsize=14, fontweight='bold')
                        ax.set_xlim(0, 1)
                        ax.set_ylim(0, 1)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        continue

                    # Aggregate visitors by PRIZM_CODE for this quarter
                    prizm_totals = quarter_data.groupby('PRIZM_CODE', as_index=False)['Visitor'].sum()

                    # Map PRIZM_CODE to PRIZM segments
                    prizm_totals['Segment'] = prizm_totals['PRIZM_CODE'].map(prizm_map)
                    prizm_totals = prizm_totals[prizm_totals['Segment'].notna()]

                    if prizm_totals.empty:
                        # If no valid segments, create empty chart
                        ax.text(0.5, 0.5, f"No valid segments\nfor {quarter} {current_year}",
                               ha='center', va='center', transform=ax.transAxes,
                               fontsize=12, style='italic')
                        ax.set_title(f"{quarter} {current_year}", fontsize=14, fontweight='bold')
                        ax.set_xlim(0, 1)
                        ax.set_ylim(0, 1)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        continue

                    # Group by PRIZM segment and sum visitors
                    segment_totals = prizm_totals.groupby('Segment', as_index=False)['Visitor'].sum()
                    segment_totals = segment_totals.sort_values('Visitor', ascending=True)  # Ascending for horizontal bars

                    # Keep only top 5 segments
                    segment_totals = segment_totals.tail(5)

                    # Prepare data for horizontal barplot
                    segments = segment_totals['Segment'].tolist()
                    values = segment_totals['Visitor'].tolist()

                    if not segments:
                        ax.text(0.5, 0.5, f"No segments\nfor {quarter} {current_year}",
                               ha='center', va='center', transform=ax.transAxes,
                               fontsize=12, style='italic')
                        ax.set_title(f"{quarter} {current_year}", fontsize=14, fontweight='bold')
                        ax.set_xlim(0, 1)
                        ax.set_ylim(0, 1)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        continue

                    # Get colors from color_utils module for consistency
                    colors = get_color_palette(custom_palette, len(segments))

                    # Apply slight transparency for comparison year to differentiate
                    if year_idx == 0:  # Comparison year - use slightly transparent variants
                        colors = [f"{color}CC" if len(color) == 7 else color for color in colors]

                    # Create horizontal bars
                    bars = ax.barh(range(len(segments)), values, alpha=0.8)

                    # Apply colors to bars
                    for bar, color in zip(bars, colors[:len(bars)]):
                        bar.set_facecolor(color)
                        bar.set_edgecolor('white')
                        bar.set_linewidth(0.5)

                    # Add value labels on bars
                    for i, (bar, value) in enumerate(zip(bars, values)):
                        if value > 0:
                            # Format large numbers
                            if value >= 1_000_000:
                                label = f"{value/1_000_000:.1f}M"
                            elif value >= 1_000:
                                label = f"{value/1_000:.0f}K"
                            else:
                                label = f"{value:.0f}"

                            # Position label at the end of the bar
                            ax.text(bar.get_width() + max(values)*0.01, bar.get_y() + bar.get_height()/2,
                                   label, ha='left', va='center', fontsize=9, fontweight='bold')

                    # Customize the chart
                    ax.set_title(f"{quarter} {current_year}", fontsize=14, fontweight='bold')
                    ax.set_yticks(range(len(segments)))
                    ax.set_yticklabels(segments, fontsize=10)
                    ax.set_xlabel('Visitors', fontsize=10, fontweight='bold')

                    # Format x-axis
                    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K' if x >= 1000 else f'{x:.0f}'))

                    # Add grid for better readability
                    ax.grid(True, alpha=0.3, axis='x')
                    ax.set_axisbelow(True)

                    # Set clean background
                    ax.set_facecolor('white')

            # Adjust layout and save
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)  # Make room for the main title

            # Save the combined plot
            output_filename = f"{safe_dmo_name}_quarterly_prizm_barplots_{comp_year}_{year}.png"
            output_path = os.path.join(output_dir, output_filename)
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)

            print(f"[DEBUG] Generated quarterly PRIZM barplots for {dmo_display}: {output_path}")

    except Exception as e:
        print(f"[ERROR] Error in generate_quarterly_prizm_barplots: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    print("[DEBUG] Exiting generate_quarterly_prizm_barplots")


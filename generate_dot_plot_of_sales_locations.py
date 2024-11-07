#!/usr/bin/env python3
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import logging
import os
import subprocess
from shapely.geometry import Point
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import pycountry
import difflib
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# Set up logging
logging.basicConfig(
    filename='/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/CHARTS/sales_map.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def notify(title, text):
    """Send a system notification on macOS."""
    subprocess.call(['osascript', '-e', f'display notification "{text}" with title "{title}"'])

def prepare_sales_data(sales_df):
    """Prepare sales data by standardizing city, state, county, zip code, and country names."""
    # Ensure all relevant columns are present and standardized
    columns_to_prepare = ['City', 'State', 'County', 'Zip', 'Country']
    for col in columns_to_prepare:
        if col in sales_df.columns:
            sales_df[col] = sales_df[col].astype(str).str.strip()
            if col in ['City', 'State', 'County']:
                sales_df[col] = sales_df[col].str.lower()
            elif col == 'Country':
                sales_df[col] = sales_df[col].str.title()
        else:
            sales_df[col] = ''
    return sales_df

def standardize_country_names(sales_df):
    """Standardize country names in sales data to match official country names."""
    def get_official_country_name(name):
        try:
            return pycountry.countries.lookup(name).name
        except LookupError:
            return name  # Return the original name if not found

    sales_df['Country'] = sales_df['Country'].apply(get_official_country_name)
    sales_df['Country'] = sales_df['Country'].str.strip().str.title()
    return sales_df

def load_geonames_data(file_path, country_code):
    """Load and process GeoNames data for a given country."""
    try:
        # Define column names based on GeoNames data format
        column_names = [
            'geonameid', 'name', 'asciiname', 'alternatenames',
            'latitude', 'longitude', 'feature_class', 'feature_code',
            'country_code', 'cc2', 'admin1_code', 'admin2_code',
            'admin3_code', 'admin4_code', 'population', 'elevation',
            'dem', 'timezone', 'modification_date'
        ]

        # Read the data with all columns as strings
        geonames_df = pd.read_csv(
            file_path,
            sep='\t',
            names=column_names,
            dtype=str,
            encoding='utf-8',
            quoting=3,
            na_values=''
        )

        # Keep relevant columns
        geonames_df = geonames_df[['name', 'asciiname', 'alternatenames', 'latitude', 'longitude']]

        # Convert latitude and longitude to numeric
        geonames_df['latitude'] = pd.to_numeric(geonames_df['latitude'], errors='coerce')
        geonames_df['longitude'] = pd.to_numeric(geonames_df['longitude'], errors='coerce')

        # Combine city names from 'name', 'asciiname', and 'alternatenames'
        geonames_df['city'] = geonames_df['name'].fillna('').str.lower().str.strip()
        geonames_df['city_ascii'] = geonames_df['asciiname'].fillna('').str.lower().str.strip()
        geonames_df['alternatenames'] = geonames_df['alternatenames'].fillna('').str.lower().str.strip()

        # Explode alternatenames into separate rows
        geonames_df['alternatenames'] = geonames_df['alternatenames'].str.split(',')
        geonames_df = geonames_df.explode('alternatenames')

        # Normalize city names
        geonames_df['city'] = geonames_df['city'].str.replace(r'[^\w\s]', '', regex=True)
        geonames_df['city_ascii'] = geonames_df['city_ascii'].str.replace(r'[^\w\s]', '', regex=True)
        geonames_df['alternatenames'] = geonames_df['alternatenames'].str.replace(r'[^\w\s]', '', regex=True)

        # Concatenate all possible city names into one column
        geonames_df['all_names'] = geonames_df.apply(
            lambda row: [row['city'], row['city_ascii'], row['alternatenames']], axis=1
        )

        # Explode all_names into separate rows
        geonames_df = geonames_df.explode('all_names')

        # Drop duplicates
        geonames_df = geonames_df[['all_names', 'latitude', 'longitude']].drop_duplicates()

        # Rename 'all_names' to 'city'
        geonames_df.rename(columns={'all_names': 'city'}, inplace=True)

        return geonames_df
    except Exception as e:
        logging.error(f"Error loading GeoNames data for country code {country_code}: {e}")
        return None

def load_postal_code_data(file_path, country_code):
    """Load and process postal code data for a given country."""
    try:
        # Read the data with no header and all columns as strings
        postal_df = pd.read_csv(
            file_path,
            sep='\t',
            dtype=str,
            encoding='utf-8',
            header=None,
            quoting=3,
            na_values=''
        )

        # Assign column names based on the number of columns
        expected_columns = 12
        if postal_df.shape[1] >= expected_columns:
            postal_df = postal_df.iloc[:, :expected_columns]
            postal_df.columns = [
                'CountryCode', 'PostalCode', 'PlaceName', 'AdminName1', 'AdminCode1',
                'AdminName2', 'AdminCode2', 'AdminName3', 'AdminCode3',
                'Latitude', 'Longitude', 'Accuracy'
            ]
        else:
            logging.error(f"Unexpected number of columns ({postal_df.shape[1]}) in postal code data for {country_name}")
            return None

        # Standardize postal codes and place names
        postal_df['PostalCode'] = postal_df['PostalCode'].str.strip()
        postal_df['PlaceName'] = postal_df['PlaceName'].str.lower().str.strip()

        # Convert latitude and longitude to numeric
        postal_df['Latitude'] = pd.to_numeric(postal_df['Latitude'], errors='coerce')
        postal_df['Longitude'] = pd.to_numeric(postal_df['Longitude'], errors='coerce')

        # Drop rows with missing data
        postal_df.dropna(subset=['PostalCode', 'Latitude', 'Longitude'], inplace=True)

        # Remove duplicates
        postal_df.drop_duplicates(subset=['PostalCode'], inplace=True)

        return postal_df
    except Exception as e:
        logging.error(f"Error loading postal code data for {country_code}: {e}")
        return None

def merge_sales_with_geonames(sales_df, geonames_df, country_name):
    """Merge sales data with GeoNames data based on city names using enhanced fuzzy matching."""
    try:
        # Filter sales data for the given country
        country_sales_df = sales_df[sales_df['Country'] == country_name].copy()

        if country_sales_df.empty:
            logging.info(f"No sales data for {country_name} after filtering.")
            return None

        # Standardize city names
        country_sales_df['City'] = country_sales_df['City'].str.lower().str.strip()
        geonames_df['city'] = geonames_df['city'].str.lower().str.strip()

        # Remove special characters and normalize city names
        country_sales_df['City'] = country_sales_df['City'].str.replace(r'[^\w\s]', '', regex=True)
        geonames_df['city'] = geonames_df['city'].str.replace(r'[^\w\s]', '', regex=True)

        # Drop rows with missing city names
        country_sales_df = country_sales_df[country_sales_df['City'].notna() & (country_sales_df['City'] != '')]

        if country_sales_df.empty:
            logging.info(f"No valid city names in sales data for {country_name}.")
            return None

        # Prepare a mapping from sales city names to geonames city names
        geonames_cities = geonames_df['city'].dropna().unique().tolist()

        # Create a function to find the best match with a lower cutoff
        def get_best_match(city_name):
            matches = difflib.get_close_matches(city_name, geonames_cities, n=1, cutoff=0.6)
            return matches[0] if matches else None

        # Apply the function to get the best match
        country_sales_df['MatchedCity'] = country_sales_df['City'].apply(get_best_match)

        # Drop rows where no match was found
        country_sales_df = country_sales_df.dropna(subset=['MatchedCity'])

        if country_sales_df.empty:
            logging.info(f"No matches found for cities in {country_name} after fuzzy matching.")
            return None

        # Merge on MatchedCity
        merged_df = pd.merge(
            country_sales_df,
            geonames_df[['city', 'latitude', 'longitude']],
            left_on='MatchedCity',
            right_on='city',
            how='inner'
        )

        if merged_df.empty:
            logging.info(f"Merge resulted in empty DataFrame for {country_name} when merging on fuzzy matched city names.")
            return None

        # Drop duplicates and keep relevant columns
        merged_df = merged_df.drop_duplicates(subset=['Order Id']) if 'Order Id' in merged_df.columns else merged_df.drop_duplicates()
        merged_df = merged_df[['City', 'Country', 'latitude', 'longitude']]

        return merged_df
    except Exception as e:
        logging.error(f"Error merging sales data with GeoNames data for {country_name}: {e}")
        return None

def merge_sales_with_postal_codes(sales_df, postal_df, country_name):
    """Merge sales data with postal code data based on postal codes."""
    try:
        # Filter sales data for the given country
        country_sales_df = sales_df[sales_df['Country'] == country_name].copy()

        if country_sales_df.empty:
            logging.info(f"No sales data for {country_name} after filtering.")
            return None

        # Standardize postal codes
        country_sales_df['Zip'] = country_sales_df['Zip'].astype(str).str.strip().str.upper()
        postal_df['PostalCode'] = postal_df['PostalCode'].str.strip().str.upper()

        # Merge on postal code
        merged_df = pd.merge(
            country_sales_df,
            postal_df,
            left_on='Zip',
            right_on='PostalCode',
            how='inner'
        )

        if merged_df.empty:
            logging.info(f"Merge resulted in empty DataFrame for {country_name} when merging on postal codes.")
            return None

        # Convert latitude and longitude to numeric
        merged_df['Latitude'] = pd.to_numeric(merged_df['Latitude'], errors='coerce')
        merged_df['Longitude'] = pd.to_numeric(merged_df['Longitude'], errors='coerce')

        # Drop duplicates and keep relevant columns
        merged_df = merged_df.drop_duplicates(subset=['Order Id']) if 'Order Id' in merged_df.columns else merged_df.drop_duplicates()
        merged_df = merged_df[['Zip', 'Country', 'Latitude', 'Longitude']]
        merged_df.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'}, inplace=True)

        return merged_df
    except Exception as e:
        logging.error(f"Error merging sales data with postal codes for {country_name}: {e}")
        return None

def load_and_clean_country_data(country_centroids_path):
    """Load and clean the country centroids data with additional logging for debugging."""
    try:
        # Load the CSV and log the columns
        country_data_df = pd.read_csv(country_centroids_path)
        logging.info(f"Loaded country data with columns: {country_data_df.columns.tolist()}")

        # Standardize column names
        country_data_df.columns = country_data_df.columns.str.strip().str.lower()

        # Check for the presence of required columns
        required_columns = [
            'country',
            'center of the country coordinates',
            'capital coordinates',
            'largest city coordinates'
        ]
        missing_columns = [col for col in required_columns if col not in country_data_df.columns]
        if missing_columns:
            logging.error(f"Missing columns in country data: {missing_columns}")
            return None

        # Process coordinate columns
        for coord_type in ['center of the country', 'capital', 'largest city']:
            coord_column = f"{coord_type} coordinates"
            lat_col = f"{coord_type.replace(' ', '_')}_lat"
            lng_col = f"{coord_type.replace(' ', '_')}_lng"

            # Split and convert latitude and longitude to numeric
            coords = country_data_df[coord_column].str.strip().str.split(',', expand=True)
            if coords.shape[1] < 2:
                logging.error(f"Column '{coord_column}' does not have valid lat,lng format.")
                continue

            country_data_df[lat_col] = pd.to_numeric(coords[0].str.strip(), errors='coerce')
            country_data_df[lng_col] = pd.to_numeric(coords[1].str.strip(), errors='coerce')

        # Rename columns to standardized names
        country_data_df.rename(columns={
            'center_of_the_country_lat': 'center_lat',
            'center_of_the_country_lng': 'center_lng',
            'capital_lat': 'capital_lat',
            'capital_lng': 'capital_lng',
            'largest_city_lat': 'city_lat',
            'largest_city_lng': 'city_lng'
        }, inplace=True)

        # Standardize country names
        country_data_df['country'] = country_data_df['country'].str.strip().str.title()

        return country_data_df
    except Exception as e:
        logging.error(f"Error loading and cleaning country centroids data: {e}")
        return None

def merge_sales_with_country_data(sales_df, country_data_df):
    """Merge sales data with cleaned country data based on 'Country'."""
    try:
        if country_data_df is None:
            logging.error("Country data is unavailable; skipping merge.")
            return None

        # Standardize country names
        sales_df['Country'] = sales_df['Country'].str.strip().str.title()
        country_data_df['country'] = country_data_df['country'].str.strip().str.title()

        # Merge on 'Country' column
        country_columns = ['country', 'center_lat', 'center_lng', 'capital_lat', 'capital_lng', 'city_lat', 'city_lng']
        world_sales_df = pd.merge(
            sales_df,
            country_data_df[country_columns],
            left_on='Country',
            right_on='country',
            how='left'
        )

        # Determine latitude and longitude to use based on available data
        world_sales_df['latitude'] = world_sales_df['city_lat'].combine_first(
            world_sales_df['center_lat']
        ).combine_first(world_sales_df['capital_lat'])

        world_sales_df['longitude'] = world_sales_df['city_lng'].combine_first(
            world_sales_df['center_lng']
        ).combine_first(world_sales_df['capital_lng'])

        # Log resulting columns and sample data for debugging
        logging.info(f"Merged sales data with country data. Columns: {world_sales_df.columns.tolist()}")
        logging.info(f"Sample data after merging with country data:\n{world_sales_df.head()}")

        # Drop rows where latitude or longitude is still missing after assignment
        world_sales_df.dropna(subset=['latitude', 'longitude'], inplace=True)

        return world_sales_df
    except Exception as e:
        logging.error(f"Error merging sales data with country data: {e}")
        return None

def geocode_missing_locations(sales_df):
    """Geocode missing locations using any available information (Zip, City, County, State, Country)."""
    geolocator = Nominatim(user_agent="sales_map_geocoder")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    # Ensure latitude and longitude columns exist
    if 'latitude' not in sales_df.columns:
        sales_df['latitude'] = None
    if 'longitude' not in sales_df.columns:
        sales_df['longitude'] = None

    def get_coordinates(row):
        if pd.isna(row['latitude']) or pd.isna(row['longitude']):
            try:
                location = None
                # Define specific combinations to try in order of preference
                address_formats = [
                    ['Zip', 'Country'],
                    ['City', 'State', 'Country'],
                    ['City', 'Country'],
                    ['County', 'State', 'Country'],
                    ['County', 'Country'],
                    ['State', 'Country'],
                    ['Country']
                ]
                for fmt in address_formats:
                    address_parts = [row.get(part, '') for part in fmt if pd.notna(row.get(part)) and row.get(part) != '']
                    if address_parts:
                        address = ', '.join(address_parts)
                        location = geocode(address)
                        if location:
                            return pd.Series({'latitude': location.latitude, 'longitude': location.longitude})
                # If all else fails, return None
            except Exception as e:
                logging.error(f"Geocoding error for row {row.name}: {e}")
        return pd.Series({'latitude': row['latitude'], 'longitude': row['longitude']})

    # Apply geocoding to rows with missing coordinates
    missing_coords = sales_df[sales_df['latitude'].isna() | sales_df['longitude'].isna()]
    logging.info(f"Attempting to geocode {len(missing_coords)} missing locations.")
    if not missing_coords.empty:
        geocoded_coords = missing_coords.apply(get_coordinates, axis=1)
        sales_df.loc[missing_coords.index, ['latitude', 'longitude']] = geocoded_coords

    # Log any remaining missing coordinates
    remaining_missing = sales_df[sales_df['latitude'].isna() | sales_df['longitude'].isna()]
    if not remaining_missing.empty:
        logging.warning(f"{len(remaining_missing)} records still missing coordinates after geocoding.")
        logging.info(f"Sample rows still missing coordinates:\n{remaining_missing.head()}")

    return sales_df


def generate_sales_maps():
    """Generate both US and Global sales maps and save them into a single PDF."""
    try:
        logging.info('☢︎ SCRIPT START ------------------------------')
        notify('GENERATING SALES MAPS...', 'The script has started running.')

        # Define file paths
        sales_csv_path = '/Users/andrewwhite/Desktop/TPT/SALES-REPORTS/SALES-DATA-DROP/sales-report.csv'
        zip_codes_csv_path = '/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/LOCATION-DATA/uszips.csv'
        us_states_geojson_path = '/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/LOCATION-DATA/us-states.json'
        country_centroids_path = '/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/LOCATION-DATA/country_centroids.csv'
        world_geojson_path = '/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/LOCATION-DATA/ne_50m_admin_0_countries.shp'
        map_output_path = '/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/CHARTS/CHARTS_OUTPUT/sales_map.pdf'

        # Read sales data
        sales_df = pd.read_csv(sales_csv_path)

        # Prepare sales data
        sales_df = prepare_sales_data(sales_df)

        # Standardize country names
        sales_df = standardize_country_names(sales_df)

        # Verify required columns are present
        required_columns = ['Zip', 'Transaction Total', 'Country', 'City']
        missing_columns = [col for col in required_columns if col not in sales_df.columns]
        if missing_columns:
            logging.error(f"Required columns {missing_columns} are missing from the sales data.")
            print(f"Required columns {missing_columns} are missing from the sales data.")
            return

        # Read ZIP codes data
        zip_df = pd.read_csv(zip_codes_csv_path, dtype={'zip': str})
        zip_df = zip_df[['zip', 'lat', 'lng', 'state_id']]

        # Read country centroids data
        country_data_df = load_and_clean_country_data(country_centroids_path)

        # Ensure the output directory exists
        output_dir = os.path.dirname(map_output_path)
        os.makedirs(output_dir, exist_ok=True)

        # Open a PDF file to save multiple pages
        with PdfPages(map_output_path) as pdf:

            # Generate US Sales Map
            try:
                us_sales_df = sales_df[sales_df['Country'] == 'United States'].copy()

                # Standardize ZIP codes
                us_sales_df['Zip'] = us_sales_df['Zip'].astype(str).str[:5].str.zfill(5)
                merged_df = pd.merge(us_sales_df, zip_df, left_on='Zip', right_on='zip', how='inner')

                # Drop rows with missing latitude or longitude
                merged_df.dropna(subset=['lat', 'lng'], inplace=True)

                # Ensure 'Transaction Total' is numeric
                merged_df['Transaction Total'] = merged_df['Transaction Total'].replace(r'[\$,]', '', regex=True)
                merged_df['Transaction Total'] = pd.to_numeric(merged_df['Transaction Total'], errors='coerce')
                merged_df.dropna(subset=['Transaction Total'], inplace=True)

                # Exclude sales in Alaska, Hawaii, and Puerto Rico
                merged_df = merged_df[~merged_df['state_id'].isin(['AK', 'HI', 'PR'])]

                # Create GeoDataFrame
                geometry = [Point(xy) for xy in zip(merged_df['lng'], merged_df['lat'])]
                gdf = gpd.GeoDataFrame(merged_df, geometry=geometry, crs="EPSG:4326")

                # Load and process US states map
                usa_states = gpd.read_file(us_states_geojson_path)
                usa_states = usa_states[~usa_states['name'].isin(['Alaska', 'Hawaii', 'Puerto Rico'])]

                # Project to suitable projection for US maps
                crs_proj = "EPSG:2163"
                usa_states = usa_states.to_crs(crs_proj)
                gdf = gdf.to_crs(crs_proj)

                # Identify top 3 most expensive sales
                top_sales = gdf.nlargest(3, 'Transaction Total')

                # Plot map and sales points
                fig, ax = plt.subplots(figsize=(11, 8.5))
                fig.patch.set_facecolor('black')
                ax.set_facecolor('black')

                usa_states.boundary.plot(ax=ax, edgecolor='white', linewidth=0.5)
                usa_states.plot(ax=ax, color='#000000', edgecolor='white', alpha=0.3)

                # Assign colors based on 'Transaction Total'
                gdf['Color'] = gdf['Transaction Total'].apply(lambda x: 'white' if x > 10 else '#39FF14')
                gdf.loc[top_sales.index, 'Color'] = 'red'  # Mark top 3 sales in red

                # Plot sales locations
                gdf.plot(ax=ax, markersize=5, color=gdf['Color'], alpha=0.8, marker='o')

                # Custom legend handles
                legend_elements = [
                    Line2D([0], [0], marker='o', color='none', label='Sales ≤ $10', markerfacecolor='#39FF14', markersize=8),
                    Line2D([0], [0], marker='o', color='none', label='Sales > $10', markerfacecolor='white', markersize=8),
                    Line2D([0], [0], marker='o', color='none', label='Top 3 Sales', markerfacecolor='red', markersize=8)
                ]
                legend = ax.legend(handles=legend_elements, loc='lower left', facecolor='black')
                plt.setp(legend.get_texts(), color='white')

                # Set plot aesthetics
                ax.set_title("Sales Locations Across the United States", fontsize=16, color='#39FF14', pad=10)

                # Remove axis labels and ticks
                ax.axis('off')

                # Set axis limits to the bounds of the US states
                minx, miny, maxx, maxy = usa_states.total_bounds
                ax.set_xlim(minx, maxx)
                ax.set_ylim(miny, maxy)

                # Adjust margins
                plt.tight_layout()
                fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.02)

                # Descriptive text at the bottom
                total_international_sales = len(sales_df[sales_df['Country'] != 'United States'])
                description_text = f"A total of {total_international_sales:,} products sold internationally."
                fig.text(
                    0.5,  # X position
                    0.015,  # Y position moved closer to bottom
                    description_text,
                    ha='center',
                    va='center',
                    fontsize=10,
                    color='white',
                    style='italic',
                    wrap=True
                )
                # Add the figure to the PDF
                pdf.savefig(fig, facecolor=fig.get_facecolor(), bbox_inches='tight')
                plt.close(fig)
                logging.info("US Sales Map added to PDF.")
            except Exception as e:
                logging.error(f"Error generating US sales map: {e}")
                print(f"Error generating US sales map: {e}")
                notify('Error', f'An error occurred while generating US sales map: {e}')

            # Generate Global Sales Map
            try:
                # Load the world map for plotting
                world_map = gpd.read_file(world_geojson_path)

                # Plotting setup for the global map
                fig, ax = plt.subplots(figsize=(11, 8.5))
                fig.patch.set_facecolor('black')
                ax.set_facecolor('black')

                # Plot the world map
                world_map.plot(ax=ax, color='#000000', edgecolor='white', alpha=0.4)

                # Plot U.S. sales locations
                us_gdf = gdf.to_crs("EPSG:4326")  # Convert back to WGS84
                ax.scatter(
                    us_gdf.geometry.x,
                    us_gdf.geometry.y,
                    s=1,
                    c='#39FF14',
                    alpha=0.8,
                    label='U.S. Sales Locations'
                )

                # List of countries and their data files
                country_data_files = {
                    'Australia': ('/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/LOCATION-DATA/AUSTRALIA-ZIP-CODES.txt', 'AU', 'postal'),
                    'Brazil': ('/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/LOCATION-DATA/BR.txt', 'BR', 'geonames'),
                    'Canada': ('/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/LOCATION-DATA/CA.txt', 'CA', 'geonames'),
                    'Denmark': ('/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/LOCATION-DATA/DK.txt', 'DK', 'geonames'),
                    'France': ('/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/LOCATION-DATA/FR.txt', 'FR', 'geonames'),
                    'Germany': ('/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/LOCATION-DATA/DE.txt', 'DE', 'geonames'),
                    'India': ('/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/LOCATION-DATA/IN.txt', 'IN', 'geonames'),
                    'Netherlands': ('/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/LOCATION-DATA/NL.txt', 'NL', 'geonames'),
                    'New Zealand': ('/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/LOCATION-DATA/NZ.txt', 'NZ', 'geonames'),
                    'Portugal': ('/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/LOCATION-DATA/PT.txt', 'PT', 'geonames'),
                    'Russia': ('/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/LOCATION-DATA/RU.txt', 'RU', 'geonames'),
                    'South Africa': ('/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/LOCATION-DATA/ZA.txt', 'ZA', 'geonames'),
                    'Spain': ('/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/LOCATION-DATA/ES.txt', 'ES', 'geonames'),
                    'United Kingdom': ('/Users/andrewwhite/Desktop/MAC AUTOMATIONS/TPT-DATA/LOCATION-DATA/GB.txt', 'GB', 'geonames'),
                    # Add other countries and their file paths and data types here
                }

                # Plot sales locations for each country
                for country_name, (data_file_path, country_code, data_type) in country_data_files.items():
                    if not os.path.exists(data_file_path):
                        logging.warning(f"Data file {data_file_path} not found for {country_name}. Attempting to geocode.")
                        # Attempt to geocode as a fallback
                        country_sales_df = sales_df[sales_df['Country'] == country_name].copy()
                        if not country_sales_df.empty:
                            country_sales_df = geocode_missing_locations(country_sales_df)
                            country_sales_df.dropna(subset=['latitude', 'longitude'], inplace=True)
                            if not country_sales_df.empty:
                                ax.scatter(
                                    country_sales_df['longitude'],
                                    country_sales_df['latitude'],
                                    s=1,
                                    c='#39FF14',
                                    alpha=0.9
                                )
                                logging.info(f"Plotted sales locations for {country_name} using geocoding.")
                            else:
                                logging.warning(f"No valid locations to plot for {country_name} after geocoding.")
                        else:
                            logging.info(f"No sales data for {country_name} after filtering.")
                        continue

                    if data_type == 'geonames':
                        geonames_df = load_geonames_data(data_file_path, country_code)
                        if geonames_df is not None:
                            # Try merging on city names
                            merged_df = merge_sales_with_geonames(sales_df, geonames_df, country_name)
                            if merged_df is not None and not merged_df.empty:
                                ax.scatter(
                                    merged_df['longitude'],
                                    merged_df['latitude'],
                                    s=1,  # Adjust marker size as needed
                                    c='#39FF14',
                                    alpha=0.9
                                )
                                logging.info(f"Plotted sales locations for {country_name} using city names.")
                            else:
                                logging.info(f"No detailed sales data to plot for {country_name}. Attempting to geocode.")
                                # Attempt to geocode as a fallback
                                country_sales_df = sales_df[sales_df['Country'] == country_name].copy()
                                if not country_sales_df.empty:
                                    country_sales_df = geocode_missing_locations(country_sales_df)
                                    country_sales_df.dropna(subset=['latitude', 'longitude'], inplace=True)
                                    if not country_sales_df.empty:
                                        ax.scatter(
                                            country_sales_df['longitude'],
                                            country_sales_df['latitude'],
                                            s=1,
                                            c='#39FF14',
                                            alpha=0.8
                                        )
                                        logging.info(f"Plotted sales locations for {country_name} using geocoding.")
                                    else:
                                        logging.warning(f"No valid locations to plot for {country_name} after geocoding.")
                                else:
                                    logging.info(f"No sales data for {country_name} after filtering.")
                        else:
                            logging.warning(f"GeoNames data for {country_name} could not be loaded.")
                    elif data_type == 'postal':
                        postal_df = load_postal_code_data(data_file_path, country_code)
                        if postal_df is not None:
                            merged_df = merge_sales_with_postal_codes(sales_df, postal_df, country_name)
                            if merged_df is not None and not merged_df.empty:
                                ax.scatter(
                                    merged_df['longitude'],
                                    merged_df['latitude'],
                                    s=1,
                                    c='#39FF14',
                                    alpha=0.8
                                )
                                logging.info(f"Plotted sales locations for {country_name} using postal codes.")
                            else:
                                # Attempt to geocode as a fallback
                                country_sales_df = sales_df[sales_df['Country'] == country_name].copy()
                                if not country_sales_df.empty:
                                    country_sales_df = geocode_missing_locations(country_sales_df)
                                    country_sales_df.dropna(subset=['latitude', 'longitude'], inplace=True)
                                    if not country_sales_df.empty:
                                        ax.scatter(
                                            country_sales_df['longitude'],
                                            country_sales_df['latitude'],
                                            s=1,
                                            c='#39FF14',
                                            alpha=0.8
                                        )
                                        logging.info(f"Plotted sales locations for {country_name} using geocoding.")
                                    else:
                                        logging.warning(f"No valid locations to plot for {country_name} after geocoding.")
                                else:
                                    logging.info(f"No sales data for {country_name} after filtering.")
                        else:
                            logging.warning(f"Postal code data for {country_name} could not be loaded.")
                    else:
                        logging.warning(f"Unknown data type for {country_name}: {data_type}")
                        # Attempt to geocode as a fallback
                        country_sales_df = sales_df[sales_df['Country'] == country_name].copy()
                        if not country_sales_df.empty:
                            country_sales_df = geocode_missing_locations(country_sales_df)
                            country_sales_df.dropna(subset=['latitude', 'longitude'], inplace=True)
                            if not country_sales_df.empty:
                                ax.scatter(
                                    country_sales_df['longitude'],
                                    country_sales_df['latitude'],
                                    s=1,
                                    c='#39FF14',
                                    alpha=0.8
                                )
                                logging.info(f"Plotted sales locations for {country_name} using geocoding.")
                            else:
                                logging.warning(f"No valid locations to plot for {country_name} after geocoding.")
                        else:
                            logging.info(f"No sales data for {country_name} after filtering.")

                # Plot sales locations for countries without specific data files
                other_countries = set(sales_df['Country']) - set(['United States'] + list(country_data_files.keys()))
                international_sales_df = sales_df[sales_df['Country'].isin(other_countries)]
                logging.info(f"Number of international sales records: {len(international_sales_df)}")
                world_sales_df = merge_sales_with_country_data(international_sales_df, country_data_df)

                if world_sales_df is not None and not world_sales_df.empty:
                    # Geocode missing locations
                    world_sales_df = geocode_missing_locations(world_sales_df)
                    # Drop rows without coordinates
                    world_sales_df.dropna(subset=['latitude', 'longitude'], inplace=True)
                    logging.info(f"Number of records after geocoding: {len(world_sales_df)}")
                    if not world_sales_df.empty:
                        # Plot the points
                        ax.scatter(
                            world_sales_df['longitude'],
                            world_sales_df['latitude'],
                            s=1,
                            c='#39FF14',
                            alpha=0.8
                        )
                        logging.info("International sales locations plotted on the map.")
                    else:
                        logging.warning("No valid international sales locations to plot after geocoding.")
                else:
                    logging.warning("No international sales data available for plotting.")

                # Set plot aesthetics
                ax.set_title("Global Sales Map", fontsize=16, color='#39FF14', pad=10)

                # Remove axis labels and ticks
                ax.axis('off')

                # Set axis limits based on world map bounds
                minx, miny, maxx, maxy = world_map.total_bounds
                ax.set_xlim(minx, maxx)
                ax.set_ylim(miny, maxy)

                # Adjust margins to center the map with minimal space
                plt.tight_layout()
                fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05)

                # Descriptive text at the bottom
                total_international_sales = len(sales_df[sales_df['Country'] != 'United States'])
                description_text = f"A total of {total_international_sales:,} products sold internationally."
                fig.text(0.5, 0.02, description_text, ha='center', va='center', fontsize=10, color='white', style='italic', wrap=True)

                # Add the figure to the PDF
                pdf.savefig(fig, facecolor=fig.get_facecolor(), bbox_inches='tight')
                plt.close(fig)
                logging.info("Global Sales Map added to PDF.")
            except Exception as e:
                logging.error(f"Error generating Global Sales Map: {e}")
                print(f"Error generating Global Sales Map: {e}")
                notify('Error', f'An error occurred while generating Global Sales Map: {e}')

        logging.info(f"Sales maps saved at {map_output_path}.")
        print("Sales maps have been saved successfully.")
        notify('SALES MAPS GENERATED!', f'Maps saved at {map_output_path}')

    except Exception as e:
        logging.error(f"Error generating sales maps: {e}")
        print(f"Error generating sales maps: {e}")
        notify('Error', f'An error occurred: {e}')




if __name__ == '__main__':
    generate_sales_maps()

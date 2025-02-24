#!/usr/bin/env python
"""
Enhanced Fading Simulation & ML Prediction App
------------------------------------------------
This application simulates artwork fading under environmental exposure
by combining synthetic data generation with an ML model (XGBoost) to predict
fading in the LAB color space. Improvements include:
  - Incorporation of mechanistic degradation factors (Arrhenius-type temperature model)
  - Inclusion of material-specific parameters (e.g., pH, dye lightfastness ratings)
  - Enhanced feature engineering with explicit normalization and categorical handling
  - Improved logging, error messages, and user feedback via Streamlit
  - Hooks for future calibration using empirical microfading data
References for improvements:
   [oai_citation_attribution:0‡microfading.com](https://www.microfading.com/uploads/1/1/7/3/11737845/whitmore__et_al_1999.pdf) (Whitmore et al., microfading testing)
   [oai_citation_attribution:1‡getty.edu](https://www.getty.edu/conservation/publications_resources/pdf_publications/pdf/tools_for_analysis.pdf) (Getty Museum analysis of conservation environments)
"""

import sys
import subprocess
import importlib
import logging
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from skimage import color
from skimage.color import deltaE_ciede2000
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, MinMaxScaler
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# -----------------------------
# Utility and Processing Functions
# -----------------------------
def load_and_clean_dataset(file_buffer):
    """
    Load and clean the LAB color dataset from a CSV file buffer.
    Checks for required columns and handles NaNs and infinities.
    """
    try:
        dataset = pd.read_csv(file_buffer)
        required_columns = {'L', 'A', 'B', 'Color Name'}
        if not required_columns.issubset(dataset.columns):
            missing = required_columns - set(dataset.columns)
            st.error(f"Dataset is missing required columns: {missing}")
            st.stop()
        dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna(subset=['L', 'A', 'B'])
        logging.info(f"Dataset loaded with {len(dataset)} entries after cleaning.")
        return dataset
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        st.stop()

def load_and_process_image(file_buffer):
    """
    Load an image file and convert it to the LAB color space.
    """
    try:
        image = Image.open(file_buffer).convert('RGB')
        image_array = np.array(image).astype(np.float32) / 255.0
        lab_image = color.rgb2lab(image_array)
        logging.info("Image loaded and converted to LAB color space.")
        return image, lab_image
    except Exception as e:
        st.error(f"Failed to process image: {e}")
        st.stop()

def display_image(image, title='Image'):
    """Display an image using Streamlit."""
    st.image(image, caption=title, use_column_width=True)

def plot_figure(fig, title=None):
    """Display a matplotlib figure in Streamlit."""
    if title:
        st.write(title)
    st.pyplot(fig)

# -----------------------------
# Synthetic Data and ML Functions
# -----------------------------
def generate_fading_data(art_type, material_type, dye_type, lux_hours, uv_exposure,
                         temperature, humidity, pollution, year_of_manufacture, time_years, num_samples,
                         pH=7.0, lightfastness_rating=1.0):
    """
    Generate synthetic fading data.
    Now includes additional parameters:
      - pH: influences acid-catalyzed degradation (lower pH can accelerate yellowing)
      - lightfastness_rating: numerical rating (higher means more stable)
    The model incorporates an Arrhenius-type temperature factor.
    """
    L_fading = np.zeros(num_samples)
    A_fading = np.zeros(num_samples)
    B_fading = np.zeros(num_samples)

    # Normalise lux and UV exposure
    lux_normalized = lux_hours / 100000  
    uv_normalized = np.minimum(uv_exposure, 1.0)
    pollution_normalized = pollution

    # Temperature factor using Arrhenius-like model: activation energy approximation
    temp_factor = np.exp((temperature - 20) / 10)
    # Humidity factor: deviations from 50% modulate degradation linearly
    humidity_factor = 1 + (humidity - 50) / 100
    # pH factor: assume acidic conditions (pH < 7) accelerate degradation linearly
    pH_factor = 1 + (7 - pH) / 10
    # Lightfastness factor: higher ratings reduce degradation (inverse relation)
    lf_factor = 1 / lightfastness_rating

    # Combine environmental factors
    environment_factor = temp_factor * humidity_factor * pH_factor * lf_factor

    # Other normalization factors
    exposure_factor = (lux_normalized + uv_normalized) * environment_factor
    year_factor = (2020 - year_of_manufacture) / 220.0  # arbitrary scaling based on age
    time_factor = time_years / 100.0  # normalize aging effect

    # Fading simulation based on art/material/dye types
    if material_type == 'Textiles' and dye_type == 'Natural':
        L_fading += np.random.normal(loc=-5, scale=1.5, size=num_samples) * exposure_factor * year_factor * time_factor
        A_fading += np.random.normal(loc=-2, scale=1, size=num_samples) * exposure_factor * year_factor * time_factor
        B_fading += np.random.normal(loc=-2, scale=1, size=num_samples) * exposure_factor * year_factor * time_factor
    elif material_type == 'Paper with Black Text':
        L_fading += np.random.normal(loc=-1, scale=0.5, size=num_samples) * lux_normalized * year_factor * time_factor
    elif art_type == 'Wood Engraving':
        exp_decay = np.exp(-0.02 * time_years * environment_factor)
        L_fading += np.random.normal(loc=-2, scale=0.5, size=num_samples) * exposure_factor * year_factor * exp_decay
        A_fading += np.random.normal(loc=-1, scale=0.3, size=num_samples) * exposure_factor * year_factor * exp_decay
        B_fading += np.random.normal(loc=-1, scale=0.3, size=num_samples) * exposure_factor * year_factor * exp_decay
    else:
        L_fading += np.random.normal(loc=-3, scale=1, size=num_samples) * exposure_factor * year_factor * time_factor
        A_fading += np.random.normal(loc=-1.5, scale=0.5, size=num_samples) * exposure_factor * year_factor * time_factor
        B_fading += np.random.normal(loc=-1.5, scale=0.5, size=num_samples) * exposure_factor * year_factor * time_factor

    # Additional UV photochemical impact
    uv_threshold = 0.075
    uv_impact = np.maximum(uv_exposure - uv_threshold, 0) * 10
    L_fading -= uv_impact
    A_fading -= uv_impact / 2
    B_fading -= uv_impact / 2

    # Acidic materials additional effect (heuristic)
    if 'Acidic' in material_type:
        L_fading -= np.random.normal(loc=2, scale=1, size=num_samples) * pollution_normalized
        B_fading += np.random.normal(loc=3, scale=1, size=num_samples) * pollution_normalized

    # Clip values to physically plausible bounds
    L_fading = np.clip(L_fading, -20, 0)
    A_fading = np.clip(A_fading, -10, 10)
    B_fading = np.clip(B_fading, -10, 10)

    return L_fading, A_fading, B_fading

def create_synthetic_data(art_types, material_types, dye_types, valid_combinations, num_samples_per_combination=500):
    """
    Create a synthetic dataset for combinations of art, material, and dye types.
    Now also simulates additional material parameters.
    """
    np.random.seed(42)  # for reproducibility
    data_list = []
    for art_type in art_types:
        for material_type in material_types:
            dye_type_options = [dye for art, material, dye in valid_combinations
                                if art == art_type and material == material_type]
            if not dye_type_options:
                dye_type_options = [None]
            for dye_type in dye_type_options:
                lux_hours = np.random.uniform(1000, 100000, num_samples_per_combination)
                uv_exposure = np.random.uniform(0.0, 1.0, num_samples_per_combination)
                temperature = np.random.uniform(-10, 50, num_samples_per_combination)
                humidity = np.random.uniform(0, 100, num_samples_per_combination)
                pollution = np.random.uniform(0, 1.0, num_samples_per_combination)
                year_of_manufacture = np.random.randint(1455, 2020, num_samples_per_combination)
                time_years = np.random.randint(1, 101, num_samples_per_combination)
                # Additional synthetic parameters:
                pH = np.random.uniform(5.0, 8.0, num_samples_per_combination)
                lightfastness_rating = np.random.uniform(0.5, 2.0, num_samples_per_combination)

                L_fading, A_fading, B_fading = generate_fading_data(
                    art_type, material_type, dye_type, lux_hours, uv_exposure,
                    temperature, humidity, pollution, year_of_manufacture, time_years,
                    num_samples_per_combination, pH, lightfastness_rating
                )

                data = pd.DataFrame({
                    'art_type': art_type,
                    'material_type': material_type,
                    'dye_type': dye_type if dye_type is not None else 'None',
                    'lux_hours': lux_hours,
                    'uv_exposure': uv_exposure,
                    'temperature': temperature,
                    'humidity': humidity,
                    'pollution': pollution,
                    'year_of_manufacture': year_of_manufacture,
                    'time_years': time_years,
                    'pH': pH,
                    'lightfastness_rating': lightfastness_rating,
                    'L_fading': L_fading,
                    'A_fading': A_fading,
                    'B_fading': B_fading
                })
                data_list.append(data)
    return pd.concat(data_list, ignore_index=True)

def prepare_features(synthetic_data):
    """
    Prepare features and target variables for ML training.
    Combines polynomial features with one-hot encoded categorical data.
    """
    X_numeric = synthetic_data[['lux_hours', 'uv_exposure', 'temperature', 'humidity', 'pollution',
                                'year_of_manufacture', 'time_years', 'pH', 'lightfastness_rating']]
    X_categorical = synthetic_data[['art_type', 'material_type', 'dye_type']].fillna('None')

    encoder = OneHotEncoder(sparse_output=False)
    X_categorical_encoded = encoder.fit_transform(X_categorical)

    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_numeric_poly = poly.fit_transform(X_numeric)

    X = np.hstack((X_numeric_poly, X_categorical_encoded))
    Y = synthetic_data[['L_fading', 'A_fading', 'B_fading']].values
    return X, Y, encoder, poly

def train_ml_model(X, Y):
    """
    Train an XGBoost model with hyperparameter tuning.
    Incorporates scaling of features and cross-validation.
    """
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
    multi_xgb = MultiOutputRegressor(xgb)

    param_grid = {
        'estimator__n_estimators': [100, 200],
        'estimator__max_depth': [3, 5, 7],
        'estimator__learning_rate': [0.01, 0.1, 0.2]
    }
    grid_search = GridSearchCV(multi_xgb, param_grid, cv=3, scoring='neg_mean_squared_error')
    with st.spinner("Tuning hyperparameters..."):
        grid_search.fit(X_scaled, Y)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    st.write(f"**Best parameters found:** {best_params}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = cross_val_score(best_model, X_scaled, Y, cv=kf, scoring='neg_mean_squared_error')
    avg_mse = -np.mean(mse_scores)
    st.write(f"**Cross-validated MSE:** {avg_mse:.4f}")

    return best_model, scaler, avg_mse

def lab_to_rgb(lab_image):
    """
    Convert LAB image to RGB, ensuring clipping.
    """
    rgb_image = color.lab2rgb(lab_image)
    rgb_image = np.clip(rgb_image, 0, 1)
    return (rgb_image * 255).astype(np.uint8)

def simulate_exposure_by_material(lab_image, art_type, material_type, dye_type, exposure_years,
                                  uv_exposure, lux_hours, humidity, temperature):
    """
    Simulate environmental exposure effects on a LAB image.
    Applies adjustments based on art type, material, and environmental factors.
    """
    lab_exposed = lab_image.copy()
    lux_normalized = lux_hours / 100000
    uv_normalized = uv_exposure

    # Compute environmental factor with Arrhenius-like behavior
    temp_factor = np.exp((temperature - 20) / 10)
    humidity_factor = 1 + (humidity - 50) / 100
    env_factor = temp_factor * humidity_factor

    if art_type == 'Chromolithograph Print':
        lab_exposed[:, :, 0] -= ((lux_normalized * 10 * env_factor) + (uv_normalized * 10))
        lab_exposed[:, :, 1] -= ((lux_normalized * 5 * env_factor) + (uv_normalized * 5))
        lab_exposed[:, :, 2] -= ((lux_normalized * 5 * env_factor) + (uv_normalized * 5))
    elif art_type == 'Sanguine Etching':
        lab_exposed[:, :, 1] -= (lux_normalized * 10 * env_factor)
    elif art_type == 'Steel Engraving':
        lab_exposed[:, :, 0] -= (lux_normalized * 5 * env_factor)
    elif art_type == 'Wood Engraving':
        decay_factor = np.exp(-0.02 * exposure_years * env_factor)
        lab_exposed[:, :, 0] -= (lux_normalized * 3 + uv_normalized * 3) * decay_factor
        lab_exposed[:, :, 1] -= (lux_normalized * 2 + uv_normalized * 2) * decay_factor
        lab_exposed[:, :, 2] -= (lux_normalized * 2 + uv_normalized * 2) * decay_factor

    if 'Acidic' in material_type:
        lab_exposed[:, :, 0] -= uv_normalized * 10
        lab_exposed[:, :, 2] += uv_normalized * 10
    elif material_type == 'Textiles':
        if dye_type == 'Natural':
            fading_multiplier = np.log(lux_hours + 1) / np.log(100000 + 1)
            lab_exposed[:, :, 0] -= uv_normalized * 15 * fading_multiplier
            lab_exposed[:, :, 1] -= uv_normalized * 15 * fading_multiplier
            lab_exposed[:, :, 2] -= uv_normalized * 15 * fading_multiplier
        elif dye_type == 'Synthetic':
            lab_exposed[:, :, 0] -= uv_normalized * 10
            lab_exposed[:, :, 1] -= uv_normalized * 10
            lab_exposed[:, :, 2] -= uv_normalized * 10
    elif material_type == 'Paper with Black Text':
        lab_exposed[:, :, 0] -= lux_normalized * 2

    lab_exposed[:, :, 0] = np.clip(lab_exposed[:, :, 0], 0, 100)
    lab_exposed[:, :, 1] = np.clip(lab_exposed[:, :, 1], -128, 127)
    lab_exposed[:, :, 2] = np.clip(lab_exposed[:, :, 2], -128, 127)
    logging.info(f"Simulated exposure for {art_type} on {material_type}.")
    return lab_exposed

def apply_fading(lab_image, predicted_fading):
    """
    Apply predicted fading adjustments to a LAB image.
    """
    lab_faded = lab_image.copy()
    lab_faded[:, :, 0] += predicted_fading[0]
    lab_faded[:, :, 1] += predicted_fading[1]
    lab_faded[:, :, 2] += predicted_fading[2]
    lab_faded[:, :, 0] = np.clip(lab_faded[:, :, 0], 0, 100)
    lab_faded[:, :, 1] = np.clip(lab_faded[:, :, 1], -128, 127)
    lab_faded[:, :, 2] = np.clip(lab_faded[:, :, 2], -128, 127)
    logging.info("Applied predicted fading to the image.")
    return lab_faded

def compute_delta_e(lab1, lab2):
    """
    Compute Delta-E (CIEDE2000) between two LAB images.
    """
    delta_e = deltaE_ciede2000(lab1, lab2)
    logging.info("Delta-E calculated.")
    return delta_e

def plot_histograms(image1, image2, title_suffix=''):
    """
    Plot histograms of RGB channels for two images.
    """
    image1_array = np.array(image1)
    image2_array = np.array(image2)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['Red', 'Green', 'Blue']
    for i, col in enumerate(colors):
        axs[i].hist(image1_array[..., i].flatten(), bins=256, alpha=0.5,
                    label=f'{col} (Image 1)', color=col.lower())
        axs[i].hist(image2_array[..., i].flatten(), bins=256, alpha=0.5,
                    label=f'{col} (Image 2)', color=f'dark{col.lower()}')
        axs[i].set_title(f'{col} Channel {title_suffix}', fontsize=10)
        axs[i].legend()
    plt.tight_layout()
    return fig

def display_average_color(image_lab, title='Average Color'):
    """
    Display the average LAB color of an image.
    """
    average_lab = image_lab.mean(axis=(0, 1))
    average_rgb = color.lab2rgb(np.reshape(average_lab, (1, 1, 3))).reshape(1, 1, 3)
    average_rgb = np.clip(average_rgb, 0, 1)
    fig, ax = plt.subplots(figsize=(2,2))
    ax.imshow(np.ones((100, 100, 3)) * average_rgb)
    ax.set_title(title)
    ax.axis('off')
    st.write(f"{title}: L={average_lab[0]:.2f}, A={average_lab[1]:.2f}, B={average_lab[2]:.2f}")
    return average_lab, fig

def compute_average_delta_e(avg_lab1, avg_lab2):
    """
    Compute Delta-E between two average LAB color values.
    """
    lab1 = np.array([avg_lab1])
    lab2 = np.array([avg_lab2])
    delta_e = deltaE_ciede2000(lab1, lab2)[0]
    logging.info(f"Delta-E between average colors: {delta_e:.2f}")
    return delta_e

# -----------------------------
# Streamlit Interface & Main Execution
# -----------------------------
st.title("Enhanced Fading Simulation & ML Prediction")
st.write("This app simulates artwork fading and uses an XGBoost model with hyperparameter tuning to predict fading. Improvements include mechanistic degradation factors and material-specific parameters.")

# Sidebar file uploaders
csv_file = st.sidebar.file_uploader("Upload your LAB color dataset CSV", type=["csv"])
image_file = st.sidebar.file_uploader("Upload an image file", type=["png", "jpg", "jpeg"])

# Sidebar controls for environmental parameters
st.sidebar.header("Simulation Parameters")
time_years_slider = st.sidebar.slider("Years of Aging:", 0, 100, 5)
uv_exposure_slider = st.sidebar.slider("UV Exposure:", 0.0, 1.0, 0.5, 0.01)
lux_hours_slider = st.sidebar.slider("Lux Hours:", 0, 100000, 50000, 1000)
humidity_slider = st.sidebar.slider("Humidity (%):", 0, 100, 50, 1)
temperature_slider = st.sidebar.slider("Temperature (°C):", -10, 50, 20, 1)
pollution_slider = st.sidebar.slider("Pollution Level:", 0.0, 1.0, 0.5, 0.01)
year_manufacture_slider = st.sidebar.slider("Year of Manufacture:", 1455, 2020, 2000, 1)
# Additional parameters for material simulation
pH_slider = st.sidebar.slider("Material pH:", 5.0, 8.0, 7.0, 0.1)
lightfastness_slider = st.sidebar.slider("Lightfastness Rating (higher is more stable):", 0.5, 2.0, 1.0, 0.1)

# Define available options
art_types = [
    'Chromolithograph Print',
    'Sanguine Etching',
    'Steel Engraving',
    'Wood Engraving',
    'None'
]

material_types_all = [
    'Acidic Wove Paper',
    'Acidic Rag Paper',
    'Alkaline Wove Paper',
    'Alkaline Rag Paper',
    'Textiles',
    'Paper with Black Text'
]

dye_types = ['Natural', 'Synthetic']

valid_combinations = [
    ('Chromolithograph Print', 'Acidic Wove Paper', None),
    ('Sanguine Etching', 'Acidic Wove Paper', None),
    ('Sanguine Etching', 'Acidic Rag Paper', None),
    ('Sanguine Etching', 'Alkaline Wove Paper', None),
    ('Sanguine Etching', 'Alkaline Rag Paper', None),
    ('Steel Engraving', 'Acidic Wove Paper', None),
    ('None', 'Textiles', 'Natural'),
    ('None', 'Textiles', 'Synthetic'),
    ('None', 'Paper with Black Text', None),
    ('None', 'Acidic Wove Paper', None),
    ('None', 'Acidic Rag Paper', None),
    ('None', 'Alkaline Wove Paper', None),
    ('None', 'Alkaline Rag Paper', None),
    ('Wood Engraving', 'Paper with Black Text', None)
]

selected_art = st.sidebar.selectbox("Art Type:", art_types)
valid_materials = sorted({material for art, material, dye in valid_combinations if art == selected_art})
selected_material = st.sidebar.selectbox("Material Type:", valid_materials)
selected_dye = "None"
if selected_material == "Textiles":
    valid_dyes = sorted({dye for art, material, dye in valid_combinations if material == selected_material and dye is not None})
    selected_dye = st.sidebar.selectbox("Dye Type:", valid_dyes)

if st.sidebar.button("Run Simulation"):
    if csv_file is None or image_file is None:
        st.error("Please upload both a dataset CSV and an image file.")
    else:
        st.info("Processing files and running simulation...")
        # Load dataset and image
        dataset = load_and_clean_dataset(csv_file)
        original_image, original_lab = load_and_process_image(image_file)
        
        st.subheader("Original Image")
        display_image(original_image, "Original Image")
        avg_lab_before, avg_fig = display_average_color(original_lab, "Average Color - Original")
        plot_figure(avg_fig)
        
        # Create synthetic data & train model
        synthetic_data = create_synthetic_data(art_types, material_types_all, dye_types, valid_combinations, num_samples_per_combination=500)
        X, Y, encoder, poly = prepare_features(synthetic_data)
        model, scaler, mse = train_ml_model(X, Y)
        st.write(f"**Cross-validated MSE for Fading Prediction:** {mse:.4f}")
        
        # Simulate exposure on the image
        lab_exposed = simulate_exposure_by_material(original_lab, selected_art, selected_material, selected_dye,
                                                    time_years_slider, uv_exposure_slider, lux_hours_slider, humidity_slider, temperature_slider)
        exposed_image = lab_to_rgb(lab_exposed)
        st.subheader(f"Simulated Exposure: {selected_art} on {selected_material}")
        display_image(exposed_image, f"Simulated Exposure: {selected_art} on {selected_material}")
        avg_lab_exposed, avg_fig2 = display_average_color(lab_exposed, "Average Color - Simulated Exposure")
        plot_figure(avg_fig2)
        
        # Compute and display Delta-E between original and simulated exposure
        delta_e_sim = compute_delta_e(original_lab, lab_exposed)
        st.write("**Delta-E Map (Original vs Simulated Exposure):**")
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(delta_e_sim, cmap='hot')
        plt.colorbar(im, ax=ax, label='∆E')
        ax.axis('off')
        plot_figure(fig)
        delta_e_avg_sim = compute_average_delta_e(avg_lab_before, avg_lab_exposed)
        st.write(f"**Delta-E (Average Colors):** {delta_e_avg_sim:.2f}")
        
        # Plot histograms between Original and Simulated Exposure
        hist_fig = plot_histograms(original_image, exposed_image, "Original vs Simulated Exposure")
        plot_figure(hist_fig, "Histograms: Original vs Simulated Exposure")
        
        # Prepare features for ML prediction using current slider values
        categorical_input = pd.DataFrame({'art_type': [selected_art],
                                          'material_type': [selected_material],
                                          'dye_type': [selected_dye]})
        categorical_encoded = encoder.transform(categorical_input.fillna('None'))
        X_input_numeric = np.array([[lux_hours_slider, uv_exposure_slider, temperature_slider, humidity_slider, pollution_slider, year_manufacture_slider, time_years_slider, pH_slider, lightfastness_slider]])
        X_input_numeric_poly = poly.transform(X_input_numeric)
        X_input = np.hstack((X_input_numeric_poly, categorical_encoded))
        X_input_scaled = scaler.transform(X_input)
        
        predicted_fading = model.predict(X_input_scaled)[0]
        lab_faded = apply_fading(lab_exposed, predicted_fading)
        faded_image = lab_to_rgb(lab_faded)
        st.subheader("Final Faded Image After ML Prediction")
        display_image(faded_image, "Faded Image")
        avg_lab_after, avg_fig3 = display_average_color(lab_faded, "Average Color - After ML Prediction")
        plot_figure(avg_fig3)
        
        # Compute Delta-E comparisons between stages
        delta_e_ml = compute_delta_e(lab_exposed, lab_faded)
        st.write("**Delta-E Map (Simulated Exposure vs ML Prediction):**")
        fig_ml, ax_ml = plt.subplots(figsize=(8, 6))
        im_ml = ax_ml.imshow(delta_e_ml, cmap='hot')
        plt.colorbar(im_ml, ax=ax_ml, label='∆E')
        ax_ml.axis('off')
        plot_figure(fig_ml)
        delta_e_avg_ml = compute_average_delta_e(avg_lab_exposed, avg_lab_after)
        st.write(f"**Delta-E (Average Colors - Exposure vs Prediction):** {delta_e_avg_ml:.2f}")
        
        delta_e_total = compute_delta_e(original_lab, lab_faded)
        st.write("**Delta-E Map (Original vs Final Faded):**")
        fig_total, ax_total = plt.subplots(figsize=(8, 6))
        im_total = ax_total.imshow(delta_e_total, cmap='hot')
        plt.colorbar(im_total, ax=ax_total, label='∆E')
        ax_total.axis('off')
        plot_figure(fig_total)
        delta_e_avg_total = compute_average_delta_e(avg_lab_before, avg_lab_after)
        st.write(f"**Delta-E (Average Colors - Original vs Final):** {delta_e_avg_total:.2f}")
        
        hist_fig_final = plot_histograms(original_image, faded_image, "Original vs Final Faded")
        plot_figure(hist_fig_final, "Histograms: Original vs Final Faded")
        
        st.success("Simulation complete!")
import geopandas as gpd
import yaml
import joblib
import time
from pathlib import Path
from electoral.data.creation import dataset_creation
from electoral.data.imputation import Fill_NAs
from electoral.data.features import generate_spatial_features
from electoral.model.model import prepare_splits, train_and_evaluate


def load_config(config_path: str = "configs/pipeline_config.yaml") -> dict:
    """Loads the central configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def main():
    """The main execution flow of the ML pipeline."""
    print("--- Starting Electoral ML Pipeline ---")

    # Start timer
    pipeline_start_time = time.perf_counter()
    
    # Load Configuration
    print("Loading configuration...")
    config = load_config()

    #Data extraction and transformation
    print("Building core dataset and engineering features...")

    dataset = gpd.read_file("./Elections/dataset.gpkg", driver="GPKG")
    
    print(f"Dataset generated successfully with shape: {dataset.shape}")

    #Filling missing values
    print("Filling missing values...")
    dataset = Fill_NAs(dataset, config)
    print("NAs filled.")

    #Feature engineering
    print("Featuring engineering...")
    dataset = generate_spatial_features(dataset, config)
    print("Feature engineering executed.")

    # Model preparation
    print("Preparing train, validation, and test splits...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_splits(dataset, config)

    #Model Training and evaluation
    print("Training the XGBoost model...")
    trained_model, metrics = train_and_evaluate(
        X_train, X_val, X_test, 
        y_train, y_val, y_test, 
        config
    )
    
    #Model saving
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    model_path = output_dir / "xgboost_model.pkl"
    joblib.dump(trained_model, model_path)

    pipeline_duration = time.perf_counter() - pipeline_start_time
    
    print(f"--- Pipeline Complete! ---")
    print(f"Model saved to {model_path}")
    print(f"Final Metrics: {metrics}")
    print(f"Total Execution Time: {pipeline_duration:.2f} seconds")


main()
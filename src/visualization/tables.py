import pandas as pd

def generate_comparison_table(results_json_paths: list) -> pd.DataFrame:
    """
    Compiles multiple JSON evaluator outputs into a master comparison table.
    """
    import json
    
    rows = []
    for path in results_json_paths:
        with open(path, 'r') as f:
            data = json.load(f)
            
        row = {
            "Run ID": data.get("run_id"),
            "Model": data.get("metadata", {}).get("model_name"),
            "Recipe": data.get("metadata", {}).get("recipe"),
            "Clean Top-1 Acc": data.get("metrics", {}).get("Clean Top-1 Acc"),
            "Mean Corrupted Error": data.get("metrics", {}).get("Mean Corrupted Error"),
            "AEI": data.get("metrics", {}).get("Absolute Error Increase (AEI)"),
            "Raw ECE": data.get("metrics", {}).get("Raw ECE"),
            "Scaled ECE": data.get("metrics", {}).get("Temp-Scaled ECE")
        }
        rows.append(row)
        
    return pd.DataFrame(rows)

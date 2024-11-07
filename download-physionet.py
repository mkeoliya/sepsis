import kagglehub

# Download latest version to "data"
path = kagglehub.dataset_download("salikhussaini49/prediction-of-sepsis", "data")

print("Path to dataset files:", path)
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
# Initialize API client

repo_id = "Anusha3/ab_predictive_maintenance"
repo_type = "space"

# Step 1: Check if the space exists, and create if it doesn't
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False, space_sdk='docker')
    print(f"Space '{repo_id}' created.")

print(f"Attempting to upload to repo: {repo_id}, type: {repo_type}")
print(f"HF_TOKEN being used: {os.getenv('HF_TOKEN')[:5]}...{os.getenv('HF_TOKEN')[-5:] if os.getenv('HF_TOKEN') else 'None'} (first/last 5 chars)")

api.upload_folder(
    folder_path="predictive_maintenance/deployment",     # the local folder containing your files
    repo_id=repo_id,          # the target repo
    repo_type=repo_type,                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)

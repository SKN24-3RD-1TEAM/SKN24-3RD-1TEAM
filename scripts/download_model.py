import os
from huggingface_hub import snapshot_download

def download_model(repo_id, local_dir):
    os.makedirs(local_dir, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,             # 허깅페이스 모델 이름
        local_dir=local_dir, # 저장할 내 컴퓨터 경로
        local_dir_use_symlinks=False
    )

if __name__ == '__main__':
    download_model('Azure99/Blossom-V6.3-36B', './model/local_models/Azure99/Blossom-V6.3-36B')
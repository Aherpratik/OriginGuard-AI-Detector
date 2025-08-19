# split_train_val.py
import os, random, shutil

def make_val_split(src_root, dst_root, split=0.2):
    """
    Copies split fraction of each class from src_root/train to dst_root/val,
    leaving the rest in src_root/train.
    """
    for cls in ['real', 'ai_generated']:
        train_dir = os.path.join(src_root, 'train', cls)
        val_dir   = os.path.join(dst_root, 'val',   cls)
        os.makedirs(val_dir, exist_ok=True)

        files = [f for f in os.listdir(train_dir)
                 if f.lower().endswith(('.jpg','.jpeg','.png'))]
        random.shuffle(files)
        n_val = max(1, int(len(files)*split))

        for fname in files[:n_val]:
            src = os.path.join(train_dir, fname)
            dst = os.path.join(val_dir, fname)
            shutil.copy(src, dst)
        print(f"Copied {n_val} files into {val_dir}")

if __name__=="__main__":
    make_val_split('data', 'data', split=0.2)

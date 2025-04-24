
import os
import shutil
rename_dict = {
    0: '玄関', 1: 'ホール', 2: '洋室', 3: '廊下', 4: 'クローゼット',
    5: '和室', 6: '浴室', 7: '収納', 8: 'バルコニー', 9: 'トイレ'
}
def copy_and_rename_files(src_folder, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    for filename in os.listdir(src_folder):
        if filename.endswith(('.jpg', '.png')) and '_' in filename:
            prefix, rest = filename.split('_', 1)
            if prefix.isdigit():
                num = int(prefix)
                if num in rename_dict:
                    new_filename = f"{rename_dict[num]}_{rest}"
                    src_path = os.path.join(src_folder, filename)
                    dst_path = os.path.join(dst_folder, new_filename)
                    if not os.path.exists(dst_path):
                        shutil.copy2(src_path, dst_path)
                        print(f"Copied: {filename} -> {new_filename}")
copy_and_rename_files("data/train/low_mapped", "data/train_new/low")
copy_and_rename_files("data/train/high_mapped", "data/train_new/high")
import os
import re
import shutil

src_dir = "pick_pen/first_run_0417w/demo_buffer/"  # 替换为源目录
dst_dir = "pick_pen/first_run/demo_buffer/"  # 替换为目标目录

os.makedirs(dst_dir, exist_ok=True)

for root, _, files in os.walk(src_dir):
    for file in files:
        # 查找文件名中的数字
        match = re.search(r'(\d+)', file)
        if match:
            num = int(match.group(1))
            if 0 <= num <= 10000:
                src_path = os.path.join(root, file)
                dst_path = os.path.join(dst_dir, file)
                #print(f"file {file}")
                shutil.copy2(src_path, dst_path)

print("文件拷贝完成")

import os
import csv
import shutil

def flatten_dataset(root_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "groundtruth.csv")

    data = []
    counter = 1  # to rename images uniquely

    # Traverse the nested directory
    for gender in os.listdir(root_dir):
        gender_path = os.path.join(root_dir, gender)
        if not os.path.isdir(gender_path):
            continue

        for age_group in os.listdir(gender_path):
            age_path = os.path.join(gender_path, age_group)
            if not os.path.isdir(age_path):
                continue

            clean_age = age_group.replace("age", "").strip()

            for img_name in os.listdir(age_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    src_path = os.path.join(age_path, img_name)

                    # Rename image to avoid conflicts (e.g., male_20-29_001.jpg)
                    new_name = f"img_{counter:05d}" + os.path.splitext(img_name)[1]
                    dst_path = os.path.join(output_dir, new_name)

                    # Copy image to new location
                    shutil.copy2(src_path, dst_path)

                    # Add entry to CSV data
                    data.append({
                        "name": new_name,
                        "age": clean_age,
                        "gender": gender,
                        "original_path": src_path
                    })

                    counter += 1

    # Write CSV
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "age", "gender", "original_path"])
        writer.writeheader()
        writer.writerows(data)

    print(f"✅ Dataset flattened successfully into: {output_dir}")
    print(f"📄 CSV saved as: {csv_path}")
    print(f"🖼️ Total images copied: {len(data)}")


# ====== USAGE ======
root_directory = "/home/noor/Downloads/batox"         # original structured dataset
output_directory = "/home/noor/Downloads/batbox_dataset"     # new flattened folder
flatten_dataset(root_directory, output_directory)

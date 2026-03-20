from pathlib import Path

IMG_DIR = Path("/Volumes/msh_uiowa/Research Data/Han_AIR_Dec_2025/data_volumes/data_all_volumes2")
FOLDER_DIR = Path("/Volumes/msh_uiowa/Research Data/Han_AIR_Dec_2025/layers_rsync_dir/layers_2026_02_28")
EXPECTED_N = 1025


# -----------------------------------
# 1) result dirs with wrong file count
# -----------------------------------
bad_count_dirs = []

for folder in sorted(FOLDER_DIR.iterdir()):
    if not folder.is_dir():
        continue

    n_files = sum(1 for x in folder.iterdir() if x.is_file())
    if n_files != EXPECTED_N:
        # print(f"{folder} only had {n_files}, and not the expected {EXPECTED_N}")
        bad_count_dirs.append((folder.name, n_files))

print("\n=== folders with file count != 1025 ===")
for name, n in bad_count_dirs:
    print(f"{name}    ({n})")

print("\n=== copy/paste block: bad count dir names ===")
for name, _ in bad_count_dirs:
    print(name)


# -----------------------------------
# 2) source imgs missing matching dir
# -----------------------------------
missing_dirs = []

for img_path in sorted(IMG_DIR.glob("*.img")):
    stem = img_path.stem
    expected_dir = FOLDER_DIR / stem
    if not expected_dir.is_dir():
        missing_dirs.append(img_path)

print("\n=== .img files with no matching result dir ===")
for img_path in missing_dirs:
    print(str(img_path))

print("\n=== copy/paste block: missing stems ===")
for img_path in missing_dirs:
    print(img_path.stem)
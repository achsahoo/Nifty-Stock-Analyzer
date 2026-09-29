"""
Manage Git Data Tracking & Safe Push Utility
============================================
Controls whether the 'data/' folder is locked (ignored by git so local older data
never overrides remote/Streamlit Cloud) or unlocked (ready to commit fresh data).

Usage:
    python manage_git_data.py --lock      # Ignore local data changes (SAFE MODE)
    python manage_git_data.py --unlock    # Allow pushing fresh data to GitHub
    python manage_git_data.py --status    # Check current data tracking mode
"""

import sys
import os
import subprocess

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GITIGNORE_PATH = os.path.join(BASE_DIR, ".gitignore")


def get_tracked_data_files():
    try:
        res = subprocess.check_output(["git", "ls-files", "data/"], cwd=BASE_DIR, text=True)
        files = [f.strip() for f in res.splitlines() if f.strip()]
        return files
    except Exception as e:
        print(f"Error fetching tracked files: {e}")
        return []


def lock_data():
    """Marks all tracked data files as assume-unchanged and ensures data/ is in .gitignore."""
    print("🔒 Locking 'data/' folder so local data will NEVER be committed or pushed to Git...")
    files = get_tracked_data_files()
    if not files:
        print("No tracked files found in data/.")
        return

    # Update index in chunks of 100
    chunk_size = 100
    for i in range(0, len(files), chunk_size):
        chunk = files[i:i+chunk_size]
        subprocess.run(["git", "update-index", "--assume-unchanged"] + chunk, cwd=BASE_DIR)

    # Verify lock count
    res_v = subprocess.check_output(["git", "ls-files", "-v", "data/"], cwd=BASE_DIR, text=True)
    locked = sum(1 for l in res_v.splitlines() if l and l[0].islower())
    print(f"✅ Successfully locked {locked} / {len(files)} parquet files (assume-unchanged).")

    # Ensure data/ is in .gitignore
    if os.path.exists(GITIGNORE_PATH):
        with open(GITIGNORE_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        lines = content.splitlines()
        has_data = any(line.strip() == "data/" or line.strip() == "data" for line in lines)
        if not has_data:
            with open(GITIGNORE_PATH, "a", encoding="utf-8") as f:
                f.write("\ndata/\n")
            print("✅ Added 'data/' to .gitignore.")
    print("\n🎉 SAFE MODE ACTIVE: You can now freely run 'git add .' and 'git push' without worrying about overwriting newer data on GitHub/Streamlit Cloud!\n")


def unlock_data():
    """Clears assume-unchanged and removes data/ from .gitignore so you can push fresh data."""
    print("🔓 Unlocking 'data/' folder to prepare for pushing updated data to GitHub...")
    files = get_tracked_data_files()
    if not files:
        print("No tracked files found in data/.")
        return

    chunk_size = 100
    for i in range(0, len(files), chunk_size):
        chunk = files[i:i+chunk_size]
        subprocess.run(["git", "update-index", "--no-assume-unchanged"] + chunk, cwd=BASE_DIR)

    res_v = subprocess.check_output(["git", "ls-files", "-v", "data/"], cwd=BASE_DIR, text=True)
    unlocked = sum(1 for l in res_v.splitlines() if l and l[0].isupper())
    print(f"✅ Successfully unlocked {unlocked} / {len(files)} parquet files.")

    # Comment out data/ in .gitignore
    if os.path.exists(GITIGNORE_PATH):
        with open(GITIGNORE_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        new_lines = []
        for line in content.splitlines():
            if line.strip() == "data/" or line.strip() == "data":
                new_lines.append("# data/  # Temporarily un-ignored to push data")
            else:
                new_lines.append(line)
        with open(GITIGNORE_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")
        print("✅ Commented out 'data/' in .gitignore.")

    print("\n🚀 UNLOCKED: You can now stage updated data files using:")
    print("   git add data/")
    print("   git commit -m \"Update market data to latest date\"")
    print("   git push")
    print("\n💡 After pushing, remember to re-lock data with: python manage_git_data.py --lock\n")


def check_status():
    """Checks the current lock state of data files."""
    try:
        res = subprocess.check_output(["git", "ls-files", "-v", "data/"], cwd=BASE_DIR, text=True)
        lines = res.splitlines()
        # In git ls-files -v, lower-case letter (e.g. 'h') indicates assume-unchanged
        locked_count = sum(1 for line in lines if line and line[0].islower())
        total_count = len(lines)

        print("📊 Data Tracking Status:")
        print(f"   • Total tracked files in data/: {total_count}")
        print(f"   • Locked (assume-unchanged): {locked_count}")
        print(f"   • Unlocked (tracked for commits): {total_count - locked_count}")

        if locked_count > 0:
            print("\n   🟢 State: LOCKED (SAFE MODE) - Local data changes will NOT be pushed to GitHub.")
        else:
            print("\n   🟡 State: UNLOCKED - Local data changes WILL be included in git commits.")
    except Exception as e:
        print(f"Error checking status: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ["--lock", "lock"]:
        lock_data()
    elif sys.argv[1] in ["--unlock", "unlock"]:
        unlock_data()
    elif sys.argv[1] in ["--status", "status"]:
        check_status()
    else:
        print("Usage:")
        print("  python manage_git_data.py --lock     # Protect data (safe mode)")
        print("  python manage_git_data.py --unlock   # Allow pushing data")
        print("  python manage_git_data.py --status   # Check status")

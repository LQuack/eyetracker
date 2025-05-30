import os

import subprocess
from pathlib import Path

from eyetracker.tracker import EyeTracker


def get_git_root() -> Path:
    return Path(
        subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip()
    )


# Example: reference a file inside the Git repo
output_dir = get_git_root() / 'examples'
raw_video_file = os.path.join(output_dir, 'example_zebrafish_eye_video.avi')

eyetrace_data_file = os.path.join(output_dir, "example_eyetrack_data.npy")
eyetrace_metadata_file = os.path.join(output_dir, "example_eyetrack_metadata.npy")

print("\nStarting the Eye Tracking Analysis...")

try:
    # Step 3: Run the eyetracker function
    with EyeTracker(raw_video_file) as tracker:
        tracker.run_pipeline(
            data_output_path=eyetrace_data_file,
            metadata_output_path=eyetrace_metadata_file
        )

    print("\nEye Tracking Analysis Completed Successfully!")
    print(f"Data saved to: {eyetrace_data_file}")
    print(f"Metadata saved to: {eyetrace_metadata_file}")

except Exception as e:
    print(f"An error occurred during the analysis: {e}")

import os
import time
import cv2
import numpy as np
from scipy import signal
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from mpl_point_clicker import clicker
import concurrent.futures

from setuptools.sandbox import save_path

from eyetracker.utils import binarise_with_morphosnakes, calc_theta_centroid


class EyeTracker:
    DOWNSAMPLE_FACTOR = 10
    RANDOM_FRAME_INTERVAL = 200
    DEFAULT_SEMI_AXIS = (50, 25)
    DEFAULT_MORPH_ITER = 35
    BRIGHTNESS_MODIFIER = 1

    def __init__(self, video_path, save_plot=False):
        """
        Initialize the EyeTracker with a video file path.

        Args:
            video_path (str): Path to the video file to analyze
        """
        self.video_path = video_path
        self.cap = None
        self.eyetrack_data = {}
        self.eyetrack_metadata = {}
        self.width_slices = None
        self.height_slices = None
        self.slope_angle = None
        self.morph_iteration_number = self.DEFAULT_MORPH_ITER
        self.semi_axis = self.DEFAULT_SEMI_AXIS
        self.brightness_modifier = self.BRIGHTNESS_MODIFIER
        self.save_plot = save_plot

    def __enter__(self):
        """Context manager entry - open video capture"""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video file: {self.video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - release resources"""
        if self.cap is not None:
            self.cap.release()

    def process_frame(self, video_frame):
        """
        Process a single frame to extract eye angles.

        Args:
            video_frame (np.array): Grayscale video frame

        Returns:
            np.array: Array of angles for left and right eyes
        """
        angles = np.zeros(2)
        for eye in range(2):
            width_slice_eye = self.width_slices[eye]
            height_slice_eye = self.height_slices[eye]
            video_frame_eye = video_frame[
                slice(height_slice_eye[0], height_slice_eye[1]),
                slice(width_slice_eye[0], width_slice_eye[1])]
            video_frame_eye = video_frame_eye[::self.DOWNSAMPLE_FACTOR, ::self.DOWNSAMPLE_FACTOR]

            PIL_image = Image.fromarray(video_frame_eye)
            brightener = ImageEnhance.Brightness(PIL_image)
            video_frame_eye = brightener.enhance(float(self.brightness_modifier))
            video_frame_eye = np.array(video_frame_eye)

            binarised = binarise_with_morphosnakes(video_frame_eye, self.semi_axis, self.morph_iteration_number)
            angle, _ = calc_theta_centroid(binarised)

            angles[eye] = angle
        return angles

    def select_eye_and_midline_regions(self, video_frame):
        """
        Interactive selection of eye regions and fish midline.

        Args:
            video_frame (np.array): Sample frame for selection

        Returns:
            tuple: (width_slices, height_slices, slope_angle)
        """
        while True:
            print('starting image sub-selection')
            fig, ax = plt.subplots(constrained_layout=True)
            ax.imshow(video_frame, cmap='Greys_r', vmin=0, vmax=100)
            klicky = clicker(ax, ['midline', 'left_eye', 'right_eye'], markers=['o', 'x', '*'])
            plt.show(block=True)

            print('click to draw two points on the midline of the fish')
            print('for each eye, click to draw 4 points around the eye')

            coordinates = klicky.get_positions()
            fishmidline_coords = coordinates['midline']
            eye_coords = [coordinates['left_eye'].astype(int), coordinates['right_eye'].astype(int)]

            # plotting midline of the fish
            fig, ax = plt.subplots(constrained_layout=True)
            ax.set_title('midline plotting')
            ax.imshow(video_frame, cmap='Greys_r', vmin=0, vmax=100)

            # Calculate the slope of the fish midline
            midline_start_x, midline_start_y = fishmidline_coords[0, 0], fishmidline_coords[0, 1]
            midline_end_x, midline_end_y = fishmidline_coords[1, 0], fishmidline_coords[1, 1]

            slope = (midline_start_y - midline_end_y) / (midline_start_x - midline_end_x)
            intercept = (midline_start_x * midline_end_y - midline_end_x * midline_start_y) / (midline_start_x -
                                                                                               midline_end_x)

            ax.axline([0, intercept], slope=slope, c='r')
            ax.set_xlim([len(video_frame), 0])
            ax.set_ylim([len(video_frame), 0])
            plt.show(block=True)

            self.slope_angle = np.rad2deg(
                np.arctan2((midline_start_y - midline_end_y), (midline_start_x - midline_end_x)))
            print(f'the fishmidline is {self.slope_angle} degrees')

            # plotting the eye selection
            width_slices = np.zeros([2, 2])
            height_slices = np.zeros([2, 2])
            for eye in range(0, 2):
                coords = eye_coords[eye]
                fig, ax = plt.subplots(constrained_layout=True)
                ax.set_title('eye selection')
                xs = coords[:, 0]
                ys = coords[:, 1]
                width_slice = (np.min(xs), np.max(xs))
                height_slice = (np.min(ys), np.max(ys))
                ax.imshow(video_frame, cmap='Greys_r', vmin=0, vmax=100)
                ax.set_xlim(width_slice)
                ax.set_ylim(height_slice)
                plt.show(block=True)
                width_slices[eye, :] = width_slice
                height_slices[eye, :] = height_slice
            width_slices = width_slices.astype(int)
            height_slices = height_slices.astype(int)

            check = input('happy? if not, press n')
            if check == 'n':
                continue
            else:
                break

        return width_slices, height_slices

    def interactive_parameter_tuning(self, random_frame_n):
        """
        Interactive tuning of processing parameters.

        Args:
            random_frame_n (int): Frame interval for testing

        Returns:
            tuple: (morph_iteration_number, semi_axis, brightness_modifier)
        """
        while True:
            print('starting eye tracking analysis trial')

            self.brightness_modifier = input(f'increase brightness by? If not say 1.0')

            for i in range(random_frame_n, random_frame_n + random_frame_n * 5, random_frame_n):
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, dpi=100)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)  # choose random frame
                _, video_frame_color = self.cap.read()  # read frame
                video_frame = cv2.cvtColor(video_frame_color, cv2.COLOR_BGR2GRAY)  # convert from RGB to grey

                angles = np.zeros(2)
                centroids_x = np.zeros(2)
                centroids_y = np.zeros(2)

                for eye in range(0, 2):
                    width_slice = self.width_slices[eye]
                    height_slice = self.height_slices[eye]
                    video_frame_eye = video_frame[
                        slice(height_slice[0], height_slice[1]), slice(width_slice[0], width_slice[1])]
                    video_frame_eye = video_frame_eye[::self.DOWNSAMPLE_FACTOR, ::self.DOWNSAMPLE_FACTOR]

                    PIL_image = Image.fromarray(video_frame_eye)
                    brightener = ImageEnhance.Brightness(PIL_image)
                    video_frame_eye = brightener.enhance(float(self.brightness_modifier))
                    video_frame_eye = np.array(video_frame_eye)

                    binarised = binarise_with_morphosnakes(video_frame_eye, self.semi_axis, self.morph_iteration_number)
                    angle, centroid = calc_theta_centroid(binarised)

                    angles[eye] = angle
                    centroids_x[eye] = centroid[0]
                    centroids_y[eye] = centroid[1]

                    if eye == 0:
                        ax1.imshow(binarised)
                        ax3.imshow(video_frame_eye, cmap='Greys_r')  # plot the image
                        ax3.axline([centroids_x[eye], centroids_y[eye]], slope=np.tan(angles[eye]), c='r')
                        ax3.set_title('Left eye')
                    else:
                        ax2.imshow(binarised)
                        ax4.imshow(video_frame_eye, cmap='Greys_r')  # plot the image
                        ax4.axline([centroids_x[eye], centroids_y[eye]], slope=np.tan(angles[eye]), c='r')
                        ax4.set_title('Right eye')
            plt.show(block=True)
            plt.close('all')

            check = input('happy? if not, press n')
            if check == 'n':
                self.morph_iteration_number = input(
                    f'change morph iteration number from {self.morph_iteration_number} into:')
                semi_axis_h = input(f'change semi axis from {self.semi_axis[0]} into:')
                semi_axis_w = input(f'change semi axis from {self.semi_axis[1]} into:')

                self.morph_iteration_number = int(self.morph_iteration_number)
                semi_axis_h = int(semi_axis_h)
                semi_axis_w = int(semi_axis_w)
                self.semi_axis = (semi_axis_h, semi_axis_w)
                continue
            else:
                break

    def analyze_video(self, data_output_path=None, metadata_output_path=None):
        """
        Main analysis pipeline for the video.

        Args:
            data_output_path (str): Path to save eye tracking data
            metadata_output_path (str): Path to save metadata

        Returns:
            tuple: (eyetrack_metadata, eyetrack_data)
        """
        start_time = time.time()
        print('Starting full video analysis...')

        self.cap = cv2.VideoCapture(self.video_path)

        framecount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'Total frame count is {framecount}')
        current_angles = np.zeros((framecount, 2))
        failed_frames = []

        batch_size = 1000
        max_workers = 6
        checkpoints = {25: False, 50: False, 75: False, 90: False, 100: False}

        frame_index = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            while frame_index < framecount:
                print('Processing a new batch...')
                frames_batch = []
                batch_indices = []
                batch_count = 0

                while batch_count < batch_size and frame_index < framecount:
                    ret, frame = self.cap.read()
                    if not ret or frame is None:
                        print(f"Warning: Empty or unreadable frame at position {frame_index}")
                        failed_frames.append(frame_index)
                        frame_index += 1
                        continue

                    try:
                        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frames_batch.append(gray_frame)
                        batch_indices.append(frame_index)
                        batch_count += 1
                    except Exception as e:
                        print(f"Error converting frame {frame_index}: {str(e)}")
                        failed_frames.append(frame_index)
                    finally:
                        frame_index += 1

                # Only process if we got any frames in this batch
                if frames_batch:
                    # Submit frame processing
                    futures = {
                        executor.submit(
                            self.process_frame,
                            frame
                        ): idx for frame, idx in zip(frames_batch, batch_indices)
                    }

                    for future in concurrent.futures.as_completed(futures):
                        idx = futures[future]
                        try:
                            current_angles[idx, :] = future.result()
                        except Exception as e:
                            print(f"Processing failed for frame {idx}: {str(e)}")
                            failed_frames.append(idx)
                            current_angles[idx, :] = np.nan

                # Progress Reporting
                current_progress = (frame_index / framecount) * 100
                for threshold in sorted(checkpoints.keys()):
                    if current_progress >= threshold and not checkpoints[threshold]:
                        print(f"Analysis {threshold}% complete")
                        checkpoints[threshold] = True

        # Fill in missing frames using linear interpolation
        if failed_frames:
            print(f"Warning: {len(failed_frames)} frames failed processing")
            for eye in range(2):
                valid_mask = ~np.isnan(current_angles[:, eye])
                if np.any(valid_mask):
                    current_angles[~valid_mask, eye] = np.interp(
                        np.where(~valid_mask)[0],
                        np.where(valid_mask)[0],
                        current_angles[valid_mask, eye]
                    )

        print("Analysis 100% complete")

        all_angles_degrees = np.degrees(current_angles)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total execution time: {elapsed_time:.2f} seconds")
        print(f"Processed {framecount - len(failed_frames)}/{framecount} frames successfully")

        self.eyetrack_metadata.update({
            'videofile': self.video_path,
            'width_slice': self.width_slices,
            'height_slice': self.height_slices,
            'slope_angle': self.slope_angle,
            'morph_iteration_number': self.morph_iteration_number,
            'semi_axis': self.semi_axis,
            'brightness_modifier': self.brightness_modifier,
            'framecount': framecount,
            'failed_frames': failed_frames,
            'framerate': self.cap.get(cv2.CAP_PROP_FPS)
        })

        self.eyetrack_data = {
            'angleindegrees_right': all_angles_degrees[:, 0],
            'angleindegrees_left': all_angles_degrees[:, 1],
            'angleindegrees_right_norm': all_angles_degrees[:, 0] + self.slope_angle,
            'angleindegrees_left_norm': all_angles_degrees[:, 1] + self.slope_angle
        }

        self.eyetrack_data['right_eye_zero_position'] = np.median(self.eyetrack_data['angleindegrees_right_norm'])
        self.eyetrack_data['left_eye_zero_position'] = np.median(self.eyetrack_data['angleindegrees_left_norm'])

        self.eyetrack_data['right_angles_eye_centric'] = \
            self.eyetrack_data['angleindegrees_right_norm'] - self.eyetrack_data['right_eye_zero_position']
        self.eyetrack_data['left_angles_eye_centric'] = \
            self.eyetrack_data['angleindegrees_left_norm'] - self.eyetrack_data['left_eye_zero_position']

        # Making left eye angles so that nasal is positive
        self.eyetrack_data['left_angles_eye_centric'] = -self.eyetrack_data['left_angles_eye_centric']

        # Plot the final results
        self.plot_results()

        # Save results if output paths are provided
        if data_output_path and metadata_output_path:
            dir_path = os.path.dirname(data_output_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

            np.save(data_output_path, self.eyetrack_data, allow_pickle=True)
            np.save(metadata_output_path, self.eyetrack_metadata, allow_pickle=True)

        return self.eyetrack_metadata, self.eyetrack_data

    def plot_results(self):
        """Plot the final tracking results."""
        time_seconds = np.arange(len(self.eyetrack_data['angleindegrees_right_norm'])) / self.fps
        plt.figure()
        plt.plot(time_seconds, self.eyetrack_data['angleindegrees_right_norm'], 'r', alpha=0.3, label='Right Eye')
        plt.plot(time_seconds, self.eyetrack_data['angleindegrees_left_norm'], 'b', alpha=0.3, label='Left Eye')
        plt.legend()
        plt.xlabel('Time (seconds)')
        plt.ylabel('angle (degrees)')
        plt.title(f'Eye angles for {os.path.basename(self.video_path)}')
        plt.tight_layout()

        if self.save_plot:
            save_path = os.path.splitext(self.video_path)[0] + '_plot.png'
            plt.savefig(save_path, dpi=300)

        plt.show()

    def run_pipeline(self, data_output_path, metadata_output_path):
        """
        Complete pipeline from parameter selection to full analysis.

        Args:
            data_output_path (str): Path to save eye tracking data
            metadata_output_path (str): Path to save metadata

        Returns:
            tuple: (eyetrack_metadata, eyetrack_data)
        """
        # Get a sample frame for parameter selection
        random_frame_n = self.RANDOM_FRAME_INTERVAL
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_n)
        _, video_frame = self.cap.read()
        video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)

        # Step 1: Select eye regions and midline
        self.width_slices, self.height_slices = self.select_eye_and_midline_regions(video_frame)

        # Step 2: Tune processing parameters
        self.interactive_parameter_tuning(random_frame_n)

        # Step 3: Run full analysis
        return self.analyze_video(data_output_path, metadata_output_path)


def main():
    print("Welcome to the Eye Tracking Analysis Pipeline.")

    # Step 1: Input raw video file path
    raw_video_file = input("Enter the path to the raw video file: ").strip()
    if not os.path.exists(raw_video_file):
        print(f"Error: The file {raw_video_file} does not exist.")
        return

    # Step 2: Input file paths for saving output
    output_dir = input("Enter the directory to save eye tracking data and metadata: ").strip()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    eyetrace_data_file = os.path.join(output_dir, "eyetrack_data.npy")
    eyetrace_metadata_file = os.path.join(output_dir, "eyetrack_metadata.npy")

    print("\nStarting the Eye Tracking Analysis...")

    try:
        # Step 3: Run the eye tracking pipeline
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


if __name__ == "__main__":
    main()

from absl import app, flags
import time
import numpy as np
import os
import pickle
import imageio
import cv2

FLAGS = flags.FLAGS


def _get_folder():
    print()
    print("Please provide a folder that's either a /buffer/, /demo_buffer/, /preference_buffer/, or /interventions/.")
    folder = input(" ")
    print(folder)
    return folder


def _save_video(frames: np.ndarray, default_path: str):
    print()
    print("Output path: ")
    output_path = input(f"[{default_path}] ")
    if output_path == "":
        output_path = f"{default_path}"
    output_path = os.path.abspath(output_path)
    assert os.path.exists(os.path.dirname(output_path))
    if os.path.exists(output_path):
        print()
        yn = input("Will override previous video. Proceed? [y/n] [n] ")
        if yn != 'y':
            return
    print(output_path)
    print()
    imageio.mimsave(
        uri=output_path,
        ims=frames,
        fps=10,
        macro_block_size=None  # This can help with sizes not multiples of 16
    )


def inspect_buffer(path: str):
    if os.path.isfile(path):
        files = [path]
    else:
        files = [os.path.join(path, fp) for fp in os.listdir(path) if os.path.isfile(os.path.join(path, fp))]
    transitions = []
    for fp in files:
        assert os.path.basename(fp).endswith(".pkl")
        if not (os.path.basename(fp).endswith(".pkl")):
            continue
        with open(fp, "rb") as f:
            obj = pickle.load(f)
        assert isinstance(obj, list)
        assert len(obj) > 0
        assert isinstance(obj[0], dict)
        assert set(obj[0].keys()).issuperset(["observations", "actions", "next_observations"])
        transitions += obj
    
    done_indices = [i for i, o in enumerate(transitions) if o['dones'] == 1]
    print()
    print("Pick a trajectory from the following end timesteps:")
    print(done_indices + ["all"])
    done_index = input(f"[{done_indices[0]}] ")
    if done_index == "":
        done_index = done_indices[0]
    if done_index == "all":
        done_index = len(transitions) - 1
        start_index = 0
    else:
        done_index = int(done_index)
        assert done_index in done_indices
        start_index = done_indices[done_indices.index(done_index) - 1] + 1 if done_indices.index(done_index) - 1 >= 0 else 0
    print(done_index)

    camera_keys = list(transitions[start_index]["observations"].keys() - {"state"})
    assert len(camera_keys) > 0
    print()
    print("Pick a camera from the following cameras:")
    print(camera_keys)
    camera_key = input(f"['{camera_keys[0]}'] ")
    if camera_key == "":
        camera_key = camera_keys[0]
    assert camera_key in camera_keys
    print(camera_key)

    frames = []
    for i, tr in enumerate(transitions[start_index:done_index+1]):
        frame = tr["observations"][camera_key]
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        assert frame.shape == (1, 128, 128, 3)
        frame = frame[0]
        new_frame = np.zeros((450, 800, 3), dtype=np.uint8)
        new_frame[50:,:400,:] = cv2.resize(frame, (400, 400))
        new_frame[50:,400:,:] = cv2.resize(tr["next_observations"][camera_key][0], (400, 400))

        new_frame = cv2.putText(
            img = new_frame,
            text = f"t={i}",
            org = (10, 40),
            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 1,
            color = (255, 255, 255),
            thickness = 2,
            lineType = cv2.LINE_AA,
        )

        new_frame = cv2.putText(
            img = new_frame,
            text = f"r={tr['rewards']}",
            org = (120, 40),
            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 1,
            color = (255, 255, 255),
            thickness = 2,
            lineType = cv2.LINE_AA,
        )

        if "info" in tr and "intervene_action" in tr["info"]:
            new_frame = cv2.putText(
                img = new_frame,
                text = f"intervene",
                org = (220, 40),
                fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 1,
                color = (255, 255, 255),
                thickness = 2,
                lineType = cv2.LINE_AA,
            )

        frames.append(new_frame)
    frames = np.stack(frames, axis=0)

    _save_video(frames, "./video.mp4")


def inspect_classifier_data(path: str):
    assert os.path.exists(path), path
    files = [os.path.join(path, fp) for fp in os.listdir(path) if os.path.isfile(os.path.join(path, fp))]
    successes = []
    failures = []
    for fp in files:
        assert os.path.basename(fp).endswith(".pkl")
        if not os.path.basename(fp).endswith(".pkl"):
            continue
        with open(fp, "rb") as f:
            obj = pickle.load(f)
        assert isinstance(obj, list)
        if len(obj) == 0:
            continue
        assert isinstance(obj[0], dict)
        assert set(obj[0].keys()).issuperset(['observations', 'actions', 'next_observations', 'rewards', 'masks', 'dones'])
        if "success" in os.path.basename(fp):
            successes += obj
        elif "failure" in os.path.basename(fp):
            failures += obj
        else:
            raise Exception()
    
    camera_keys = list(successes[0]["observations"].keys() - {"state"})
    assert len(camera_keys) > 0
    print()
    print("Pick a camera from the following cameras:")
    print(camera_keys)
    camera_key = input(f"['{camera_keys[0]}'] ")
    if camera_key == "":
        camera_key = camera_keys[0]
    assert camera_key in camera_keys
    print(camera_key)

    success_frames = []
    for i, tr in enumerate(successes):
        frame = tr["observations"][camera_key]
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        assert frame.shape == (1, 128, 128, 3)
        frame = frame[0]
        new_frame = np.zeros((450, 400, 3), dtype=np.uint8)
        new_frame[50:,:,:] = cv2.resize(frame, (400, 400))
        success_frames.append(new_frame)
    success_frames = np.stack(success_frames, axis=0)

    failure_frames = []
    for i, tr in enumerate(failures):
        frame = tr["observations"][camera_key]
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        assert frame.shape == (1, 128, 128, 3)
        frame = frame[0]
        new_frame = np.zeros((450, 400, 3), dtype=np.uint8)
        new_frame[50:,:,:] = cv2.resize(frame, (400, 400))
        failure_frames.append(new_frame)
    failure_frames = np.stack(failure_frames, axis=0)

    _save_video(success_frames, "./successes.mp4")
    _save_video(failure_frames, "./failures.mp4")


def main(_):
    folder = _get_folder()
    path = os.path.normpath(folder)
    folder_name = os.path.basename(path) if os.path.isdir(path) else os.path.basename(os.path.dirname(path))
    if folder_name == "buffer" or folder_name == "bc_buffer":
        inspect_buffer(path)
    elif folder_name == "classifier_data":
        inspect_classifier_data(path)
    elif folder_name == "demo_data":
        inspect_buffer(path)
    else:
        raise Exception()

if __name__ == "__main__":
    app.run(main)

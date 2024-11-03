# Training on Franka Arm Walkthrough

We demonstrate how to use HIL-SERL with real robot manipulators with 4 tasks featured in the paper: RAM insertion, USB pick up and insertion, object handover, and egg flip. These representative tasks were chosen to highlight the spectrum of use cases supported by our codebase, such as dual arm support (object handover), multi-stage reward tasks (USB pick up and insertion), and dynamic tasks (egg flip). We provide detailed instructions and tips for the entire training and evaluation pipeline for the RAM insertion task, so we suggest carefully reading through this as a starting point. 

## 1. RAM Insertion

### Procedure

#### Robot Setup
If you haven't already, read through the instructions for setting up the Python environment and installing our Franka controllers in [README.md](../README.md). The following steps will assume all installation steps there have been completed and the Python environment activated. To setup the workspace, you can refer to the image of our workspace setup in our paper.

1. Adjust for the weight of the wrist cameras by editing `Desk > Settings > End-effector > Mechanical Data > Mass`.

2. Unlock the robot and activate FCI in the Franka Desk. The `franka_server` launch file is found at [serl_robot_infra/robot_servers/launch_right_server.sh](../serl_robot_infra/robot_servers/launch_right_server.sh). You will need to edit the `setup.bash` path as well as the flags for the `python franka_server.py` command. You can refer to the [README.md](../serl_robot_infra/README.md) for `serl_robot_infra` for instructions on setting these flags. To launch the server, run:
   
```bash
bash serl_robot_infra/robot_servers/launch_right_server.sh
```

#### Editing Training Configuration
For each task, we create a folder in the experiments folder to store data (i.e. task demonstrations, reward classifier data, training run checkpoints), launch scripts, and training configurations (see [experiments/ram_insertion](../examples/experiments/ram_insertion/)). Next, we will walkthrough all of the changes you need to make the training configuration in [experiments/ram_insertion/config.py](../examples/experiments/ram_insertion/config.py)) to begin training:

3. First, in the `EnvConfig` class, change `SERVER_URL` to the URL of the running Franka server.

4. Next, we need to configure the cameras. For this task, we used two wrist cameras. All cameras used for a task (both for the reward classifier and policy training) are listed in `REALSENSE_CAMERAS` and their corresponding image crops are set in `IMAGE_CROP` in the `EnvConfig` class. The camera keys used for policy training and for the reward classifier are listed in `TrainConfig` class in `image_keys` and `classifier_keys` respectively. Change the serial numbers in `REALSENSE_CAMERAS` to the serial numbers of the cameras in your setup (this can be found in the RealSense Viewer application). To adjust the image crops (and potentially the exposure), you can run the reward classifier data collection script (see step 6) or the demonstration data collection script (see step 8) to visualize the camera inputs.

5. Finally, we need to collect some poses for the training process. For this task, `TARGET_POSE` refers to the arm pose when grasping the RAM stick fully inserted into the motherboard, `GRASP_POSE` refers to the arm pose when grasping the RAM stick sitting on the holder, and `RESET_POSE` refers to the arm pose to reset to. `ABS_POSE_LIMIT_HIGH` and `ABS_POSE_LIMIT_LOW` determine the bounding box for the policy. We have `RANDOM_RESET` enabled, meaning there is randomization around the `RESET_POSE` for every reset (`RANDOM_XY_RANGE` and `RANDOM_RZ_RANGE` control the amount of randomization). You should recollect `TARGET_POSE`, `GRASP_POSE`, and ensure the bounding box is set for safe exploration. To collect the current pose of the Franka arm, you can run:
    ```bash
    curl -X POST http://<FRANKA_SERVER_URL>:5000/getpos_euler
    ```

#### Training Reward Classifier
The reward for this task is given via a reward classifier trained on camera images. For this task, we use the same two wrist images used for training the policy to train the reward classifier. The following steps goes through collecting classifier data and training the reward classifier.

> **TIP**: Depending on the difficulty of discerning whether task reward should be given, it may be beneficial sometimes to have a separate camera for the classifier or multiple zoomed in crops of the same camera image. 

6. First, we need to collect training data for the classifier. Navigate into the examples folder and run:
    ```bash
    cd examples
    python record_success_fail.py --exp_name ram_insertion --successes_needed 200
    ```
   While the script is running, all transitions recorded are marked as negative (or no reward) by default. If the space bar is held during a transition, that transition will be marked as positive. The script will terminate when enough positive transitions have been collected (defaults to 200, but can be set via the successes_needed flag). For this task, you should collect negative transitions of the RAM stick held in various locations in the workspace and during the insertion process, and pressing the space bar when the RAM is fully inserted. The classifier data will be saved to the folder `experiments/ram_insertion/classifier_data`.

   > **TIP**: To train a classifier robust against false positives (this is important for training a successful policy), we've found it helpful to collect 2-3x more negative transitions as positive transitions to cover all failure modes. For instance, for RAM insertion, this may include attempting an insertion on a wrong location on the motherboard, only halfway-in insertions, or holding the RAM stick right next to the slot.

7. To train the reward classifier, navigate to this task's experiment folder and run:
    ```bash
    cd experiments/ram_insertion
    python ../../train_reward_classifier.py --exp_name ram_insertion
    ```
    The reward classifier will be trained on the camera images specified by the classifier keys in the training config. The trained classifier will be saved to the folder `experiments/ram_insertion/classifier_ckpt`.

#### Recording Demonstrations
A small number of human demonstrations is crucial to accelerating the reinforcement learning process, and for this task, we use 20 demonstrations. 

8. To record the 20 demonstrations with the spacemouse, run:
    ```bash
    python ../../record_demos.py --exp_name ram_insertion --successes_needed 20
    ```
    Once the episode is deemed successful by the reward classifier or the episode times out, the robot will reset. The script will terminate once 20 successful demonstrations have been collected, which will be saved to the folder `experiments/ram_insertion/demo_data`.

     > **TIP**: During the demo data collection progress, you may notice the reward classifier outputting false positives (episode terminating with reward given without a successful insertion) or false negatives (no reward given despite successful insertion). In that case, you should collect additional classifier data to target the classifier failure modes observed (i.e., if the classifier is giving false positives for holding RAM stick in the air, you should collect more negative data of that occurring). Alternatively, you can also adjust the reward classifier threshold, although we strongly recommend collecting additional classifier data (or even adding more classifier cameras/images if needed) before doing this.

#### Policy Training
Policy training is done asynchronously via an actor node, responsible for rolling out the policy in the environment and sending the collected transitions to the learner, and a learner node, responsible for training the policy and sending the updated policy back to the actor. Both the actor and the learner should be running during policy training.

9. Inside the folder corresponding to the RAM insertion experiment ([experiments/ram_insertion](../examples/experiments/ram_insertion/)), you will find `run_actor.sh` and `run_learner.sh`. In both scripts, edit `checkpoint_path` to point to the folder where checkpoints and other data generated in the training process will be saved to and in `run_learner.sh`, edit `demo_path` to point to the path of the recorded demonstrations (if there are multiple demonstration files, you can provide multiple `demo_path` flags). To begin training, launch both nodes:
    ```bash
    bash run_actor.sh
    bash run_learner.sh
    ```

    > **TIP**: To resume a previous training run, simply edit `checkpoint_path` to point to the folder corresponding to the previous run and the code will automatically load the latest checkpoint and training buffer data and resume training.

10. During training, you should give some interventions as necessary with the spacemouse to speed up the training run, particularly closer to the beginning of the run or when the policy is exploring an incorrect behavior repeatedly (i.e. moving the RAM stick away from the motherboard). For reference, with the randomizations on and giving occasional interventions, the policy took around 1.5 hours to converge to 100% success rate.

    > **TIP**: For this task, we added a bit of further randomization in the grasping pose. Hence, you should periodically regrasp the RAM stick and pressing F1 and placing the RAM back in the holder.

11. To evaluate the trained policy, add the flags `--eval_checkpoint_step=CHECKPOINT_NUMBER_TO_EVAL` and `--eval_n_trajs=N_TIMES_TO_EVAL` to `run_actor.sh`. Then, launch the actor:
    ```bash
    bash run_actor.sh
    ```

## 2. USB Pick Up and Insertion

### Procedure

#### Robot Setup
These steps assume you have completed all installation procedures have been completed and the Python environment is activated. To setup the workspace, you can refer to the image of our workspace setup in our paper.

1. Adjust for the weight of the wrist cameras by editing `Desk > Settings > End-effector > Mechanical Data > Mass`.

2. Unlock the robot and activate FCI in the Franka Desk. If you haven't done so already, you will need to edit the `franka_server` launch file (refer to step 2 in RAM Insertion instructions). To launch the server, run:
   
```bash
bash serl_robot_infra/robot_servers/launch_right_server.sh
```

#### Editing Training Configuration
Next, we need to edit the training configuration in [experiments/usb_pickup_insertion/config.py](../examples/experiments/usb_pickup_insertion/config.py)) to begin training:

3. First, in the `EnvConfig` class, change `SERVER_URL` to the URL of the running Franka server.

4. Next, we need to configure the cameras. For this task, we used 2 wrist cameras and 1 side cameras (with different crops for the policy and the classifier). Change the serial numbers in `REALSENSE_CAMERAS` to the serial numbers of the cameras in your setup (this can be found in the RealSense Viewer application) and adjust the image crops in `IMAGE_CROP`. To visualize the camera views, you can run the reward classifier data collection script or the demonstration data collection script.

> **NOTE**: This config is an example of how to use multiple cropped views of the same camera. For each cropped view of the same camera, you will need to add an entry to `REALSENSE_CAMERAS` and `IMAGE_CROP`. Note that we avoid initializing the same camera more than once in the `init_cameras` function in `USBEnv` in [experiments/usb_pickup_insertion/wrapper.py](../examples/experiments/usb_pickup_insertion/wrapper.py))

5. Finally, you will need to collect the `TARGET_POSE` which for this task, refers to the arm pose when grasping the USB while it is fully inserted in the port. Also, ensure that the bounding box is set for safe exploration (see `ABS_POSE_LIMIT_HIGH` and `ABS_POSE_LIMIT_LOW`). Note that `RESET_POSE` (pose the arm where move before dropping the USB during episode reset) is already defined and that `RANDOM_RESET` is enabled. To collect the current pose of the Franka arm, you can run:
    ```bash
    curl -X POST http://<FRANKA_SERVER_URL>:5000/getpos_euler
    ```

#### Training Reward Classifier
For this task, we use a cropped view of the side camera to train the reward classifier. The following steps goes through collecting classifier data and training the reward classifier.

6. First, we need to collect training data for the classifier. Navigate into the examples folder and run:
    ```bash
    cd examples
    python record_success_fail.py --exp_name usb_pickup_insertion
    ```
    Refer to step 6 in the RAM Insertion instructions for more details on using this script. For this task, we found it helpful to collect negative transitions of the RAM stick held in various locations in the workplace, only partially-successful insertions, and successful insertions into the wrong USB port. The classifier data will be saved to the folder `experiments/usb_pickup_insertion/classifier_data`.

7. To train the reward classifier, navigate to this task's experiment folder and run:
    ```bash
    cd experiments/usb_pickup_insertion
    python ../../train_reward_classifier.py --exp_name usb_pickup_insertion
    ```
    The reward classifier will be trained on the camera images specified by the classifier keys in the training config. The trained classifier will be saved to the folder `experiments/usb_pickup_insertion/classifier_ckpt`.

#### Recording Demonstrations
A small number of human demonstrations is crucial to accelerating the reinforcement learning process, and for this task, we use 20 demonstrations. 

8. To record the 20 demonstrations with the spacemouse, run:
    ```bash
    python ../../record_demos.py --exp_name usb_pickup_insertion --successes_needed 20
    ```
    Once the episode is deemed successful by the reward classifier or the episode times out, the robot will reset. This is also a good chance to verify the reward classifier and reset is working as intended. The script will terminate once 20 successful demonstrations have been collected, which will be saved to the folder `experiments/usb_pickup_insertion/demo_data`.

#### Policy Training

9. Inside the folder corresponding to the USB pickup and insertion experiment ([experiments/usb_pickup_insertion](../examples/experiments/usb_pickup_insertion/)), you will find `run_actor.sh` and `run_learner.sh`. In both scripts, edit `checkpoint_path` to point to the folder where checkpoints and other data generated in the training process will be saved to and in `run_learner.sh`, edit `demo_path` to point to the path of the recorded demonstrations (if there are multiple demonstration files, you can provide multiple `demo_path` flags). To begin training, launch both nodes:
    ```bash
    bash run_actor.sh
    bash run_learner.sh
    ```

10. During training, you should give some interventions as necessary with the spacemouse to speed up the training run, particularly closer to the beginning of the run or when the policy is exploring an incorrect behavior repeatedly (i.e. moving the USB away from the motherboard). For reference, our policy took around 2.5 hours to converge to 100% success rate.

11. To evaluate the trained policy, add the flags `--eval_checkpoint_step=CHECKPOINT_NUMBER_TO_EVAL` and `--eval_n_trajs=N_TIMES_TO_EVAL` to `run_actor.sh`. Then, launch the actor:
    ```bash
    bash run_actor.sh
    ```

## 3. Object Handover
Instructions to follow soon!

## 4. Egg Flip
Instructions to follow soon!

## Additional Tips for Success
- Tips on giving human interventions
    - First and foremost, use a spacemouse for interventions (something like a keyboard is much less precise)!
    - Give interventions frequently at the beginning parts of training (this can be as frequent as intervening for some timesteps every episode
    or every other episode). It's important to strike a balance between letting the policy explore (essential for RL) while also guiding it to explore efficiently through interventions. For instance, for an insertion task (i.e. RAM insertion), in the beginning of training, the policy will have a lot of random motion. We would typically let it explore these random motions for 20-30 timesteps then intervene to guide the object close to the insertion port, where we would let the policy practice the insertion. Alternating between exploration and interventions will give the policy opportunities to explore each portion of the task while also not wasting time exploring completely incorrect behaviors (i.e. exploring random motions at the edge of the bounding box far away from the insertion port). We also found it beneficial to intervene to help the policy finish the task and get the reward semi-frequently even in the beginning (i.e. letting 1/3 or more of the episodes get reward) - the frequent rewards will help the value backup propagate faster and speed up training
    - As the policy starts to behave more reasonably (policy can successfully finish the task by itself with minimal/none interventions every once in a while), we can taper off the rate of interventions significantly. At this stage, we would mostly hold off on intervening unless the policy is repeatedly making the same mistake. 
    - Sometimes, we may want the trained policy to have a more robust retry behavior (meaning the policy can successfully complete the task even if it makes a mistake earlier on) or to be more robust to external disturbances. In that case, we can also use interventions to help it practice these edge cases. For instance, if we want our USB pickup and insert policy to be robust against an edge case (i.e. still being able to successfully insert the USB after first dropping the USB close to the motherboard after an initial failed attempt), we can intervene to cause the policy to make this mistake and let it practice recovering from the mistake. Using interventions to bring the policy to a place to practice these recovery behaviors is effective for reaching 100% success rate as these mistakes may occur too infrequently otherwise yet the policy still needs to learn how to recover to improve from a 97% success rate policy to a 100% success rate policy.

More to follow soon!


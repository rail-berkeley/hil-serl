import tqdm
import time

for step in tqdm.tqdm(
      range(0, 1000), dynamic_ncols=True, desc="learner"):
    print("for loop")
    time.sleep(3)

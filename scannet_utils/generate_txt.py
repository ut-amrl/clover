import random

all_scenes = [f"scene{i:04d}_00" for i in range(210)]

random.shuffle(all_scenes)

num_train = 150
num_val = 10
num_test = 50

train_scenes = all_scenes[:num_train]
val_scenes = all_scenes[num_train : num_train + num_val]
test_scenes = all_scenes[num_train + num_val :]

with open("scannet_train.txt", "w") as f:
    f.write("\n".join(train_scenes))

with open("scannet_val.txt", "w") as f:
    f.write("\n".join(val_scenes))

with open("scannet_test.txt", "w") as f:
    f.write("\n".join(test_scenes))

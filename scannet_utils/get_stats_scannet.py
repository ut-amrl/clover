import csv
import json
from collections import defaultdict

view_counts = defaultdict(int)
instance_counts = defaultdict(int)

for i in range(240):
    scene_id = f"scene{i:04d}_00"
    anno_file = "data/ScanNet/2d_instance/" + scene_id + ".json"

    with open(anno_file, "r") as f:
        data = json.load(f)

        instances = defaultdict(set)
        for frame in data["frames"]:
            for instance in frame["instances"]:
                instance_id = instance["instanceId"]
                class_name = instance["label"]

                instances[class_name].add(instance_id)
                view_counts[class_name] += 1

        for class_name, instance_ids in instances.items():
            instance_counts[class_name] += len(instance_ids)

print("View counts:")
for class_name, count in view_counts.items():
    print(f" - {class_name}: {count}")

print("\nInstance counts:")
for class_name, count in instance_counts.items():
    print(f" - {class_name}: {count}")

# write csv file
columns = ["class_name", "view_count", "instance_count"]
with open("scannet_stats.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(columns)
    for class_name in sorted(view_counts.keys()):
        writer.writerow(
            [class_name, view_counts[class_name], instance_counts[class_name]]
        )

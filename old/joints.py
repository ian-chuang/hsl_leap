import math
pi = math.pi
joints = [
  ["if_rot", "joint_0", -1.047, 1.047, 0],
  ["if_mcp", "joint_1", -0.314, 2.23, pi/2],
  ["if_pip", "joint_2", -0.506, 1.885, 0],
  ["if_dip", "joint_3", -0.366, 2.042, 0],
  ["mf_rot", "joint_4", -1.047, 1.047, 0],
  ["mf_mcp", "joint_5", -0.314, 2.23, pi/2],
  ["mf_pip", "joint_6", -0.506, 1.885, 0],
  ["mf_dip", "joint_7", -0.366, 2.042, 0],
  ["rf_rot", "joint_8", -1.047, 1.047, 0],
  ["rf_mcp", "joint_9", -0.314, 2.23, pi/2],
  ["rf_pip", "joint_10", -0.506, 1.885, 0],
  ["rf_dip", "joint_11", -0.366, 2.042, 0],
  ["th_cmc", "joint_12", -0.349, 2.094, pi/2],
  ["th_axl", "joint_13", -0.349, 2.094, pi],
  ["th_mcp", "joint_14", -0.47, 2.443, 0],
  ["th_ipl", "joint_15", -1.34, 1.88, 0],
]


import json
import os


def rad_to_encoder(angle_rad, resolution=1024 * 4):
  return int((math.pi + angle_rad) / (2 * math.pi) * resolution)


def homing_offset_encoder(angle_rad, resolution=1024 * 4):
  return int((angle_rad) / (2 * math.pi) * resolution)


def build_dicts(joints_list, resolution=1024 * 4):
  mj = {}
  default = {}
  for i, j in enumerate(joints_list):
    name_mj = j[0]
    name_def = j[1]
    min_rad = j[2]
    max_rad = j[3]
    homing_rad = j[4]

    entry = {
      "id": i,
      "drive_mode": 0,
      "homing_offset": homing_offset_encoder(homing_rad, resolution),
      "range_min": rad_to_encoder(min_rad, resolution),
      "range_max": rad_to_encoder(max_rad, resolution),
    }

    mj[name_mj] = entry
    default[name_def] = entry

  return mj, default


def write_json(data, path):
  with open(path, "w") as f:
    json.dump(data, f, indent=4, sort_keys=True)


if __name__ == "__main__":
  res = 1024 * 4
  mj_dict, def_dict = build_dicts(joints, resolution=res)
  base = os.path.dirname(__file__)
  write_json(mj_dict, os.path.join(base, "leap_hand_mj.json"))
  write_json(def_dict, os.path.join(base, "leap_hand.json"))

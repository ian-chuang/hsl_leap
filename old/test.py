import re
import json
import math

def extract_joint_limits(xml_text):
    # Regex to match a joint block (non-greedy)
    joint_block_re = re.compile(
        r'<joint\s+name="(?P<name>[^"]+)"[^>]*>(?P<body>.*?)</joint>',
        re.DOTALL
    )

    # Regex to match the limit tag inside a joint
    limit_re = re.compile(
        r'<limit[^>]*lower="(?P<lower>[-\d\.eE]+)"[^>]*upper="(?P<upper>[-\d\.eE]+)"'
    )

    results = []

    for match in joint_block_re.finditer(xml_text):
        joint_name = match.group("name")
        joint_body = match.group("body")

        limit_match = limit_re.search(joint_body)
        if limit_match:
            lower = float(limit_match.group("lower"))
            upper = float(limit_match.group("upper"))
            results.append((joint_name, lower, upper))

    return results


def rad_to_encoder(angle_rad, resolution=1024 * 4):
    return int((math.pi + angle_rad) / (2 * math.pi) * resolution)


if __name__ == "__main__":
    # Read URDF/XML from file
    with open("test.urdf", "r") as f:
        xml_text = f.read()

    joint_limits = extract_joint_limits(xml_text)

    output = {}
    output2 = {}

    for idx, (name, lower, upper) in enumerate(joint_limits):
        
        range_min = rad_to_encoder(lower)
        range_max = rad_to_encoder(upper)

        offset = 0
        if idx in [1, 5, 9, 12]: 
            offset = 1024

        if idx in [13]:
            offset = 2048

        output[name] = {
            "id": idx,
            "drive_mode": 0,
            "homing_offset": offset,
            "range_min": range_min,
            "range_max": range_max,
        }
        

    print(json.dumps(output, indent=4))

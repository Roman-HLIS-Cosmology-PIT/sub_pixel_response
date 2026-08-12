from pathlib import Path

import yaml
from sub_pixel_response.imagesim import run_simulation

# Note: I want to try writing this again later to see if it can all be done in imagesim.py
# without taking too much time

image_dir = Path("all_scas")
config_dir = Path("all_scas_configs")

image_dir.mkdir(exist_ok=True)
config_dir.mkdir(exist_ok=True)

new_config = "example_test.yaml"

with open(new_config) as f:
    base_config = yaml.safe_load(f)

for sca in range(1, 19):
    print(f"Running SCA {sca}")

    outfile = image_dir / f"roman_sca_{sca:02d}.fits"

    # Skip SCAs that are already finished
    if outfile.exists() and outfile.stat().st_size > 0:
        print(f"{outfile} already exists, skipping SCA {sca}")
        continue

    config = base_config.copy()

    config["SCA"] = sca
    config["outFile"] = str(image_dir / f"roman_sca_{sca:02d}.fits")

    # write a temporary yaml
    temp_yaml = config_dir / f"config_sca_{sca:02d}.yaml"
    with open(temp_yaml, "w") as f:
        yaml.safe_dump(config, f)

    # now we run simulation from imagesim.py, this is all a trial run to see if code works
    run_simulation(str(temp_yaml))

from pathlib import Path

import numpy as np
import yaml
from sub_pixel_response.imagesim import run_simulation
from sub_pixel_response.multi_sca_utils import euler_angle_conversion_w, r_sca, r_wfi
from sub_pixel_response.refdistort import distortion_headers

# run_all_scas.py may need fixed, trying to figure out which alpha, delta, and phi values need to be equal to
# WFI function arguments


def run_all_scas(WFI_RACEN, WFI_DECCEN, WFI_LONPOLE):
    """
    Run simulations for all SCAs (1 to 18) using the provided configuration.

    Parameters:
    ----------
    WFI_RACEN : float
        Right Ascension of the WFI center in degrees.
    WFI_DECCEN : float
        Declination of the WFI center in degrees.
    WFI_LONPOLE : float
        Longitude of the celestial pole in degrees.
    """

    # Alpha, delta, and phi values for rotation matrix from inertial to SCA frame (in degrees)
    alpha_i = WFI_RACEN
    delta_i = WFI_DECCEN
    phi_i = WFI_LONPOLE

    image_dir = Path("all_scas")
    config_dir = Path("all_scas_configs")

    image_dir.mkdir(exist_ok=True)
    config_dir.mkdir(exist_ok=True)

    new_config = "example_test.yaml"

    with open(new_config) as f:
        base_config = yaml.safe_load(f)

    for sca in range(1, 19):
        print(f"Running SCA {sca}")

        # Getting the rotation matrix R for each SCA
        R = r_sca(sca)

        if R is None:
            print(f"No rotation matrix found for SCA {sca}, skipping.")
            continue

        # Converting rotation matrix R to Euler angles (alpha, delta, phi)
        matrix_conversion = euler_angle_conversion_w(R @ r_wfi(alpha_i, delta_i, phi_i))

        alpha, delta, phi = matrix_conversion

        if alpha < 0:
            alpha = alpha + 2 * np.pi

        if delta < 0:
            delta = delta + 2 * np.pi

        if phi < 0:
            phi = phi + 2 * np.pi

        outfile = image_dir / f"roman_sca_{sca:02d}.fits"

        # Skip SCAs that are already finished
        if outfile.exists() and outfile.stat().st_size > 0:
            print(f"{outfile} already exists, skipping SCA {sca}")
            continue

        config = base_config.copy()

        config["SCA"] = sca
        config["outFile"] = str(image_dir / f"roman_sca_{sca:02d}.fits")

        # Added WFI_RACEN, WFI_DECCEN, and WFI_LONPOLE to the config
        # Not sure if the arguments should be in the config, will ask everyone later if this is okay
        config["WFI_RACEN"] = WFI_RACEN
        config["WFI_DECCEN"] = WFI_DECCEN
        config["WFI_LONPOLE"] = WFI_LONPOLE

        # Add SCA-specific distortion/WCS keywords
        sca_header = distortion_headers[sca - 1]

        if not config.get("OLDWCS", False):
            for kw in sca_header:
                config[kw] = sca_header[kw]

        # Adding SCA specific rotation angles to the config
        config["CRVAL1"] = float(np.degrees(alpha))
        config["CRVAL2"] = float(np.degrees(delta))
        config["LONPOLE"] = float(np.degrees(phi))

        print("RACEN:", np.degrees(alpha), "DEC_CEN:", np.degrees(delta), "LONPOLE:", np.degrees(phi))

        print("normalized lonpole:", np.degrees(phi) % 360.0)

        # write a temporary yaml
        temp_yaml = config_dir / f"config_sca_{sca:02d}.yaml"
        with open(temp_yaml, "w") as f:
            yaml.safe_dump(config, f)

        print("\n===== FINAL CONFIG WCS VALUES =====")
        print("SCA =", config["SCA"])
        print("CRVAL1 =", config["CRVAL1"])
        print("CRVAL2 =", config["CRVAL2"])
        print("LONPOLE =", config["LONPOLE"])

        print("\nCD:")
        print("CD1_1 =", config.get("CD1_1"))
        print("CD1_2 =", config.get("CD1_2"))
        print("CD2_1 =", config.get("CD2_1"))
        print("CD2_2 =", config.get("CD2_2"))

        print("\nSIP:")
        for key in sorted(config):
            if key.startswith(("A_", "B_")) or key in ("A_ORDER", "B_ORDER"):
                print(key, "=", config[key])

        print("\n===== SCA ROTATION TEST =====")
        print("R =")
        print(R)

        print("\nEuler angles:")
        print("alpha =", np.degrees(alpha))
        print("delta =", np.degrees(delta))
        print("phi   =", np.degrees(phi))

        # now we run simulation from imagesim.py, this is all a trial run to see if code works
        run_simulation(str(temp_yaml))


# This section needs changed to take in arguments from run_all_scas function above
if __name__ == "__main__":
    run_all_scas(WFI_RACEN=80.5, WFI_DECCEN=-69.5, WFI_LONPOLE=225)

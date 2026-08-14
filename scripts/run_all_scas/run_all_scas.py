from pathlib import Path

import numpy as np
import yaml
from sub_pixel_response.imagesim import run_simulation

# Note: I want to try writing this again later to see if it can all be done in imagesim.py
# without taking too much time


def r_sca(sca_number):
    """
    Get the rotation matrix for a given SCA number.

    Parameters:
    ----------
    sca_number : int
        SCA number (1 to 18).

    Returns
    -------
    np.ndarray
        3x3 rotation matrix.
    """
    R_matrices = {
        1: np.array(
            [
                [0.999999315634242, 0.000000756773992, -0.001169927550709],
                [0.000000000000000, 0.999999790789090, 0.000646855297397],
                [0.001169927795471, -0.000646854854711, 0.999999106423476],
            ]
        ),
        2: np.array(
            [
                [0.999999317472245, -0.000002226814009, -0.001168353578692],
                [0.000000000000000, 0.999998183697735, -0.001905938412203],
                [0.001168355700780, 0.001905937111347, 0.999997501171220],
            ]
        ),
        3: np.array(
            [
                [0.999999320647781, -0.000004882749786, -0.001165624354528],
                [0.000000000000000, 0.999991226436935, -0.004188919807704],
                [0.001165634581296, 0.004188916961952, 0.999990547090676],
            ]
        ),
        4: np.array(
            [
                [0.999993746083975, 0.000003956266759, -0.003536633609215],
                [0.000000000000000, 0.999999374308016, 0.001118652572210],
                [0.003536635822060, -0.001118645576250, 0.999993120395904],
            ]
        ),
        5: np.array(
            [
                [0.999993743280690, -0.000005092637341, -0.003537424704405],
                [0.000000000000000, 0.999998963712116, -0.001439643947090],
                [0.003537428370199, 0.001439634939641, 0.999992706999289],
            ]
        ),
        6: np.array(
            [
                [0.999993753496664, -0.000013138707319, -0.003534514821982],
                [0.000000000000000, 0.999993091064063, -0.003717233398632],
                [0.003534539241888, 0.003717210178921, 0.999986844603884],
            ]
        ),
        7: np.array(
            [
                [0.999982604926478, 0.000013348076824, -0.005898276552062],
                [0.000000000000000, 0.999997439319063, 0.002263041165692],
                [0.005898291655705, -0.002263001799924, 0.999980044290084],
            ]
        ),
        8: np.array(
            [
                [0.999982580295055, -0.000001820634454, -0.005902465851664],
                [0.000000000000000, 0.999999952428316, -0.000308453180996],
                [0.005902466132454, 0.000308447807832, 0.999982532724200],
            ]
        ),
        9: np.array(
            [
                [0.999982474526266, -0.000015200040292, -0.005920338612335],
                [0.000000000000000, 0.999996704174254, -0.002567419060041],
                [0.005920358124804, 0.002567374064805, 0.999979178758281],
            ]
        ),
        10: np.array(
            [
                [0.999999281495090, -0.000000776459662, 0.001198752977111],
                [0.000000000000000, 0.999999790227638, 0.000647722686919],
                [-0.001198753228576, -0.000647722221527, 0.999999071722879],
            ]
        ),
        11: np.array(
            [
                [0.999999281469882, 0.000002283730509, 0.001198772081556],
                [0.000000000000000, 0.999998185381686, -0.001905054680476],
                [-0.001198774256874, 0.001905053311637, 0.999997466852872],
            ]
        ),
        12: np.array(
            [
                [0.999999283066333, 0.000005014914020, 0.001197431279904],
                [0.000000000000000, 0.999991230192174, -0.004188023250054],
                [-0.001197441781239, 0.004188020247519, 0.999990513264794],
            ]
        ),
        13: np.array(
            [
                [0.999993645162039, -0.000003997295552, 0.003565055337604],
                [0.000000000000000, 0.999999371406965, 0.001121242915054],
                [-0.003565057578574, -0.001121235789737, 0.999993016572999],
            ]
        ),
        14: np.array(
            [
                [0.999993636610722, 0.000005126438086, 0.003567451721251],
                [0.000000000000000, 0.999998967513086, -0.001437001309005],
                [-0.003567455404602, 0.001436992164806, 0.999992604130378],
            ]
        ),
        15: np.array(
            [
                [0.999993641945108, 0.000013245932714, 0.003565935207391],
                [0.000000000000000, 0.999993101038782, -0.003714549076386],
                [-0.003565959808809, 0.003714525459079, 0.999986743027754],
            ]
        ),
        16: np.array(
            [
                [0.999982442445639, -0.000013435585461, 0.005925759018026],
                [0.000000000000000, 0.999997429642397, 0.002267313079374],
                [-0.005925774249385, -0.002267273270902, 0.999979872133165],
            ]
        ),
        17: np.array(
            [
                [0.999982408009136, 0.000001803740908, 0.005931582335042],
                [0.000000000000000, 0.999999953764330, -0.000304091003477],
                [-0.005931582609293, 0.000304085653911, 0.999982361774279],
            ]
        ),
        18: np.array(
            [
                [0.999982293396816, 0.000015251910148, 0.005950853739057],
                [0.000000000000000, 0.999996715586803, -0.002562970075082],
                [-0.005950873284184, 0.002562924693588, 0.999979009041775],
            ]
        ),
    }

    return R_matrices.get(sca_number)


def euler_angle_conversion(R):
    """
    Convert a rotation matrix to Euler angles (alpha, delta, phi).

    Parameters:
    ----------
    R : np.ndarray
        3x3 rotation matrix.

    Returns
    -------
    tuple
        Euler angles (alpha, delta, phi) in radians.
    """
    delta = np.arctan2(R[2, 2], np.hypot(R[2, 0], R[2, 1]))
    alpha = np.arctan2(R[2, 1], R[2, 0])
    s = R @ np.array([-np.sin(delta) * np.cos(alpha), -np.sin(delta) * np.sin(alpha), np.cos(delta)])
    phi = np.arctan2(-s[0], s[1]) + 2 * np.pi
    return alpha, delta, phi


image_dir = Path("all_scas")
config_dir = Path("all_scas_configs")

image_dir.mkdir(exist_ok=True)
config_dir.mkdir(exist_ok=True)

new_config = "example_test.yaml"

with open(new_config) as f:
    base_config = yaml.safe_load(f)

for sca in [14]:
    print(f"Running SCA {sca}")

    # Getting the rotation matrix R for each SCA
    R = r_sca(sca)

    if R is None:
        print(f"No rotation matrix found for SCA {sca}, skipping.")
        continue

    # Converting rotation matrix R to Euler angles (alpha, delta, phi)
    alpha, delta, phi = euler_angle_conversion(R)

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

    # Adding SCA specific rotation angles to the config
    config["raCen"] = float(alpha)
    config["decCen"] = float(delta)
    config["LONPOLE"] = float(phi)

    print("RACEN:", np.degrees(alpha), "DEC_CEN:", np.degrees(delta), "LONPOLE:", np.degrees(phi))

    print("normalized lonpole:", np.degrees(phi) % 360.0)

    # write a temporary yaml
    temp_yaml = config_dir / f"config_sca_{sca:02d}.yaml"
    with open(temp_yaml, "w") as f:
        yaml.safe_dump(config, f)

    print(f"\n===== Running SCA {sca} =====")
    print(f"Config: {temp_yaml}")

    print(
        f"SCA {sca}",
        f"alpha / raCen={np.degrees(alpha):.6f} deg",
        f"delta / decCen={np.degrees(delta):.6f} deg",
        f"phi / LONPOLE={np.degrees(phi):.6f} deg",
    )

    print("\n--- WCS TEST ---")

    print("\nCD:")
    print("CD1_1 =", config.get("CD1_1"))
    print("CD1_2 =", config.get("CD1_2"))
    print("CD2_1 =", config.get("CD2_1"))
    print("CD2_2 =", config.get("CD2_2"))

    # now we run simulation from imagesim.py, this is all a trial run to see if code works
    run_simulation(str(temp_yaml))

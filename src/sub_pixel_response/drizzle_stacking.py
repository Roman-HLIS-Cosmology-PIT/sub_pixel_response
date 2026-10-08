from drizzlepac import astrodrizzle

if __name__ == "__main__":
    images = [
        "myTestPoly_10_2_1.fits",
        "myTestPoly_10_5_2.fits",
        "myTestPoly_10_5_3.fits",
        "myTestPoly_10_6_4.fits",
    ]
    astrodrizzle.AstroDrizzle(images, output="test_stack")

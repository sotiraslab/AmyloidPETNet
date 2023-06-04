# formulas for converting SUVRs to Centiloids

def av45_suvr_to_centiloid(suvr):

    return 188.22 * suvr - 189.16

av45_centiloid_cutoff = 20

fbb_centiloid_cutoff = 18

def fbb_suvr_to_centiloid(suvr):

    return 157.15 * suvr - 151.87
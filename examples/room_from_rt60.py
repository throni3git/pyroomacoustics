"""
This example creates a room with reverberation time specified by inverting Sabine's formula.
This results in a reverberation time slightly longer than desired.
The simulation is pure image source method.
The audio sample with the reverb added is saved back to `examples/samples/guitar_16k_reverb.wav`.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

import pyroomacoustics as pra

methods = ["ism", "hybrid"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulates and adds reverberation to a dry sound sample. Saves it into `./examples/samples`."
    )
    parser.add_argument(
        "--method",
        "-m",
        choices=methods,
        default=methods[1],
        help="Simulation method to use",
    )
    args = parser.parse_args()

    # The desired reverberation time and dimensions of the room
    rt60_tgt = 0.3  # seconds
    room_dim = [10, 7.5, 3.5]  # meters

    # import a mono wavfile as the source signal
    # the sampling frequency should match that of the room
    fs, audio = wavfile.read("examples/samples/guitar_16k.wav")

    # We invert Sabine's formula to obtain the parameters for the ISM simulator
    e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

    pra.constants.set("octave_bands_keep_dc", True)
    pra.constants.set("octave_bands_n_fft", 1024)

    # Create the room
    if args.method == "ism":
        room = pra.ShoeBox(
            room_dim,
            fs=fs,
            materials=pra.Material(e_absorption),
            max_order=max_order,
            use_rand_ism=True,
            air_absorption=True,
            use_third_octave_filterbank=True,
        )
    elif args.method == "hybrid":
        test_material = pra.Material(
            energy_absorption={
                "coeffs": [0.20, 0.20, 0.24, 0.25, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9],
                "center_freqs": [
                    125,
                    250,
                    500,
                    1000,
                    2000,
                    4000,
                    5039,
                    6349,
                    8000,
                    16000,
                ],
            },
            scattering={
                "coeffs": [0.3, 0.5, 0.6, 0.6, 0.7, 0.7, 0.7],
                "center_freqs": [125, 250, 500, 1000, 2000, 4000, 8000],
            },
        )

        room = pra.ShoeBox(
            room_dim,
            fs=fs,
            materials=test_material,
            max_order=3,
            ray_tracing=True,
            air_absorption=True,
            use_third_octave_filterbank=True,
        )

    dtf_sofa = pra.MeasuredDirectivityFile(
        "LSPs_HATS_GuitarCabinets_Akustikmessplatz", fs=fs
    )
    rotation_data = pra.Rotation3D([0, 0], "yz", degrees=True)
    directivity_loudspeaker = dtf_sofa.get_source_directivity(
        "Palmer_1x12inch",
        # "Genelec_8020",
        orientation=rotation_data,
        use_third_octave_bands=True,
    )

    # place the source in the room
    room.add_source(
        [2.5, 3.73, 1.76],
        signal=audio,
        delay=0.5,
        directivity=directivity_loudspeaker,
    )
    room.add_source(
        [2.5, 4.73, 1.76],
        signal=audio,
        delay=0.5,
        directivity=directivity_loudspeaker,
    )

    # define the locations of the microphones
    mic_locs = np.c_[
        [6.3, 4.87, 1.2],
        [6.3, 4.93, 1.2],  # mic 1  # mic 2
        [6.3, 4.98, 1.2],  # mic 1  # mic 2
    ]

    # finally place the array in the room
    room.add_microphone_array(mic_locs)

    # Run the simulation (this will also build the RIR automatically)
    room.simulate()

    # export RIR (normalized)
    the_rir = room.rir[0][0].copy()
    the_rir /= np.max(np.abs(the_rir))
    wavfile.write(f"examples/samples/ir_{args.method}.wav", fs, the_rir)

    room.mic_array.to_wav(
        f"examples/samples/guitar_16k_reverb_{args.method}.wav",
        norm=True,
        bitdepth=np.int16,
    )

    # measure the reverberation time
    rt60 = room.measure_rt60()
    print("The desired RT60 was {}".format(rt60_tgt))
    print("The measured RT60 is {}".format(rt60[1, 0]))

    # plot the RIRs
    select = None  # plot all RIR
    # select = (2, 0)  # uncomment to only plot the RIR from mic 2 -> src 0
    # select = [(0, 0), (2, 0)]  # only mic 0 -> src 0, mic 2 -> src 0
    fig, axes = room.plot_rir(select=select, kind="ir")  # impulse responses
    fig, axes = room.plot_rir(select=select, kind="tf")  # transfer function
    fig, axes = room.plot_rir(select=select, kind="spec")  # spectrograms

    plt.tight_layout()
    plt.show()

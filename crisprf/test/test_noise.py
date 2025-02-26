from scipy.io import savemat

from crisprf.util import RFDataShape, SRTDataset, gen_noise, plot_sample


def test_gen_noise():
    sample = SRTDataset()[0]
    shapes = RFDataShape.from_sample(**sample)
    noise = gen_noise(y=sample["y"], dT=shapes.dT, snr=2.0)

    y_noise = (sample["y"] + noise).cpu()
    plot_sample(
        prefix_scope=("y",), save_path="tmp/noise_real.png", y_noise=y_noise, **sample
    )

    savemat("tmp/y_noise.mat", {"y": sample["y"].cpu(), "y_noise": y_noise})

from crisprf.util.bridging import RFDataShape, plot_sample
from crisprf.util.dataloading import SRTDataset
from crisprf.util.noise import gen_noise


def test_gen_noise():
    sample = SRTDataset()[0]
    shapes = RFDataShape.from_sample(**sample)
    noise = gen_noise(y=sample["y"], dt=shapes.dt, snr=2.0)

    plot_sample(
        prefix_scope=("y",),
        save_path="tmp/noise_real.png",
        y_noise=sample["y"] + noise,
        **sample
    )

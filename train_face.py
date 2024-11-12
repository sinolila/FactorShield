
import sys
import os
from steganogan import SteganoGAN
from steganogan.encoders import BasicEncoder, DenseEncoder
from steganogan.decoders import BasicDecoder, DenseDecoder
from steganogan.critics import BasicCritic
def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, 'steganogan')
    sys.path.insert(0, project_root)
    steganogan = SteganoGAN(1, DenseEncoder, DenseDecoder, BasicCritic, hidden_size=32, cuda=True, verbose=True,log_dir='models/samplesCelebA')
    steganogan.load_metrics()
    steganogan.save_plot()

if __name__ == '__main__':
    main()
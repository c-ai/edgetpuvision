"""A demo which runs object classification and streams video to the browser."""

# export TEST_DATA=/usr/lib/python3/dist-packages/edgetpu/test_data
#
# python3 -m edgetpuvision.classify_server \
#   --model ${TEST_DATA}/mobilenet_v2_1.0_224_inat_bird_quant.tflite \
#   --labels ${TEST_DATA}/inat_bird_labels.txt

from .apps import run_server
from .classify import add_render_gen_args, render_gen

def main():
    run_server(add_render_gen_args, render_gen)

if __name__ == '__main__':
    main()

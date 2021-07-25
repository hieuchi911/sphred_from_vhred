from mains import VHREDTrainer, VHREDTester
from configs import args
import tensorflow as tf

if __name__ == '__main__':
    train = True
    if train:
        trainer = VHREDTrainer()
        trainer.main(is_nucleus=args['sphred-nucleus'])
    else:
        tester = VHREDTester()
        tester.main(is_nucleus=args['sphred-nucleus'])

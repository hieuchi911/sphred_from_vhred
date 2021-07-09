from mains import VHREDTrainer, VHREDTester
import tensorflow as tf

if __name__ == '__main__':
    train = True
    if train:
        trainer = VHREDTrainer()
        trainer.main()
    else:
        tester = VHREDTester()
        tester.main()

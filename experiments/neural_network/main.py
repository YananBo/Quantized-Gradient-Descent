import argparse
import os

import numpy as np

from train import *

agents = 5

w = np.array([[0.6, 0.2, 0, 0.2, 0],[0.2, 0.8, 0, 0, 0], [0, 0, 0.6, 0.1, 0.3], [0.2, 0, 0.1, 0.3, 0.4],[0, 0, 0.3, 0.4, 0.3]])
dataset = "cifar10"
epochs = 4000
bs = 32

def parse_args():
    ''' Function parses command line arguments '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test_num", default=0, type=int)
    parser.add_argument("-r","--run_num", default=0, type=int)
    parser.add_argument("-s", "--stratified", action='store_true')
    return parser.parse_args()

args = parse_args()
cwd = os.getcwd()
results_path = os.path.join(cwd, "results")
if not os.path.isdir(results_path):
    os.mkdir(results_path)

stratified = args.stratified
fname = os.path.join(results_path,f"{dataset}_e{epochs}_hom{stratified}_{args.test_num}_{args.run_num}.csv")


print(f"Test Num {args.test_num}, run num: {args.run_num}, {fname}")
if args.test_num == 0:
    QGDTrainer(dataset=dataset, batch_size=bs, epochs=epochs, num=0.001, w=w, fname=fname, stratified=stratified)
elif args.test_num == 1:
    CDSGDTrainer(dataset=dataset, batch_size=bs, epochs=epochs, num=0.001, w=w, fname=fname, stratified=stratified)
elif args.test_num == 2:
    CDSGDPTrainer(dataset=dataset, batch_size=bs, epochs=epochs, num=0.001, w=w, fname=fname, stratified=stratified) 
elif args.test_num == 3:
    CDSGDNTrainer(dataset=dataset, batch_size=bs, epochs=epochs, num=0.001, w=w, fname=fname, stratified=stratified)
elif args.test_num == 4:
    DLASTrainer(dataset=dataset, batch_size=bs, epochs=epochs, w=w, kappa=0.37, fname=fname, stratified=stratified)
elif args.test_num == 5:
    DAMSGradTrainer(dataset=dataset, batch_size=bs, epochs=epochs, w=w, fname=fname, stratified=stratified)
elif args.test_num == 6:
    DAdaGradTrainer(dataset=dataset, batch_size=bs, epochs=epochs, w=w, fname=fname, stratified=stratified)
elif args.test_num == 7:
    DAdSGDTrainer(dataset=dataset, batch_size=bs, epochs=epochs,w=w, fname=fname, stratified=stratified)
    
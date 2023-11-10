#!/bin/bash

deepspeed --bind_cores_to_rank CNN_Training_Pytorch_Cifar/Training.py --deepspeed $@
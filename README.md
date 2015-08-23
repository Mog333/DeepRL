# DeepRL
Code for running Deep Reinforcement Learning Experiments

DQN is the Deep Q Network outlined in:

Human-level control through deep reinforcement learning.
Nature, 518(7540):529-533, February 2015


Example command to run:

THEANO_FLAGS='device=cpu,floatX=float32' python run_dqn.py --seed 666 --steps-per-epoch 2500 --test-length 2500 --max-history 10000 --replay-start-size 1000

OR specify cudnn / atlas blas library, and use CUDNN v3 timing for selecting conv implementation:

THEANO_FLAGS='blas.ldflags=-lf77blas -latlas -lgfortran,device=gpu,floatX=float32,optimizer_including=cudnn,dnn.conv.algo_fwd=time_once,dnn.conv.algo_bwd=time_once' python run_dqn.py --rom breakout --base-rom-path ../roms --network-type dnn



THEANO_FLAGS='blas.ldflags=-lf77blas -latlas -lgfortran,device=gpu,floatX=float32,optimizer_including=cudnn,dnn.conv.algo_fwd=time_once,dnn.conv.algo_bwd=time_once,profile=True,profile_memory=True' nohup python run_dqtn.py --rom freeway --base-rom-path ../ALE/roms --network-type dnn > freewayOutput &
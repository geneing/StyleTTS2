import os
from multiprocessing import Process
import subprocess
import signal 

# nodelist=[(0,1),(1,2),(2,3),(3,4),(4,0),(5,5),(6,6)]
# nodelist=[4,1,2,3,5,6]
nodelist=[4]
num_machines = len(nodelist)
all_processes = []

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    for p in all_processes:
        p.kill()
    
    exit(0)
    

def run_node(node,rank):
    cmd=f'ssh rnd2{node}.itw "mamba activate tts; cd /rhome/eingerman/Projects/DeepLearning/TTS/StyleTTS2; accelerate launch --machine_rank {rank} --num_machines {num_machines} train_first.py --config_path ./Configs/config_libritts.yml"'
    print(cmd)
    subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)    
    
    for  rank, node in enumerate(nodelist):
        p = Process(target=run_node, args=(node,rank))
        p.start()
        all_processes.append(p)

    for p in all_processes:
        p.join()
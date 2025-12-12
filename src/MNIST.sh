#!/bin/bash
#SBATCH --job-name=MNIST100
#SBATCH --ntasks=1
#SBATCH --mem=512
#SBATCH --time=15:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=francoferrari123@outlook.com

#SBATCH --gres=gpu:1 # se solicita una gpu cualquiera( va a tomar la primera que quede disponible indistintamente si es una p100 o una a100)

#SBATCH --partition=normal
#SBATCH --qos=gpu

source /etc/profile.d/modules.sh

cd ~/tesisFING/MNIST100
python main.py --numEpochs=100 --labeledSize=100
python main.py --numEpochs=100 --labeledSize=100
python main.py --numEpochs=100 --labeledSize=100
python main.py --numEpochs=100 --labeledSize=100
python main.py --numEpochs=100 --labeledSize=100
python main.py --numEpochs=100 --labeledSize=100
python main.py --numEpochs=100 --labeledSize=100
python main.py --numEpochs=100 --labeledSize=100
python main.py --numEpochs=100 --labeledSize=100
python main.py --numEpochs=100 --labeledSize=100
python main.py --numEpochs=100 --labeledSize=100
python main.py --numEpochs=100 --labeledSize=100
python main.py --numEpochs=100 --labeledSize=100
python main.py --numEpochs=100 --labeledSize=100
python main.py --numEpochs=100 --labeledSize=100

Conda environment can be created using:

  $ conda create -n deep_ADNI python=3.6
  
  $ git clone https://github.com/the-yanqi/MML-ADNI.git
  
  $ pip install -r deep_learning_ADNI/recquirements.txt

Training a network
------------------

You can train a network by typing::

  $ python main/DDP_network.py results_path --classifier vgg --n_classes 3 --batch_size 8 --lr 1

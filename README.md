# csc2529_project
Course Project for CSC2529

We use the library [cleverhans 4.0.0](https://github.com/cleverhans-lab/cleverhans/releases/tag/v4.0.0) for our attack implementations. 

Code Organization:
- ```attack_results/```: storage for all experimental results in pkl files.
- ```cleverhans_fixed/```: modified PGD implementation to fix a memory leak bug.
- ```comparison_defenses/```: copies of implementations of [Ensemble Adversarial Training](https://github.com/JZ-LIANG/Ensemble-Adversarial-Training) and [Fast Adversarial Training](https://github.com/locuslab/fast_adversarial).
- attack.py: loads test data and evaluates given models against FGSM, PGD and CW attacks.
-  




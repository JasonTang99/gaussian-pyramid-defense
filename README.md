# csc2529_project
Course Project for CSC2529

We use the library [cleverhans 4.0.0](https://github.com/cleverhans-lab/cleverhans/releases/tag/v4.0.0) for our attack implementations. 


TODO List / Ideas:
- Compare how much the scaling factor matters compared to number of ensemble elements
- Do small scaling factors (e.g. 1.1) work as well as larger ones?
    - Would be more efficient and easier to train ensemble models from the base one

- voting
x Figure out and run ens_adv_train

- Figure out CW L2/Linf norm 
    maybe rerun with lower binary_search if exceeds?
- read surveys and find out epsilons
    usually use {2, 5, 10, 16}/256 = {0.78%, 1.95%, 3.91%, 6.25%}
    


- Run experiments
- Repeat for CIFAR10

- Be more precise in norm specification and problem statement




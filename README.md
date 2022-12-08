# csc2529_project
Course Project for CSC2529

We use the library [cleverhans 4.0.0](https://github.com/cleverhans-lab/cleverhans/releases/tag/v4.0.0) for our attack implementations. 

Code Organization:
- 
- attack.py: loads test data and evaluates given models against FGSM, PGD and CW attacks.

Denoiser:
- adversarial_dataset.py: loads and generates custom MNIST and CIFAR10 adversarial dataset for training
- train_denoiser.py: trains the denoiser models with user-specified parameters. 
- Sample Usage:
```python
python train_denoiser.py --dataset=cifar10 --arch=dncnn --lr=1e-3 --batch_size=64 --epochs=5
```
- test_denoiser.py: test denoiser performance by evaluating model accuracy, PSNR, and SSIM of reconstructed images. Also generate FGSM, PGD and CW attacks
- plot_denoiser_exp.ipynb: plot visualization and results on denoisers

Pretrained denoiser models can be found in /trained_denoisers
Denoiser architecutures are defined in /models/denoisers.py




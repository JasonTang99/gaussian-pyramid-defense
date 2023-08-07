# Gaussian Pyramid Ensemble Defense
Course Project for CSC2529

We use the library [cleverhans 4.0.0](https://github.com/cleverhans-lab/cleverhans/releases/tag/v4.0.0) for our attack implementations. 

## Code Organization:

### Ensemble + Attacks
- ```attack_results/```: storage for all experimental results in pkl files.
- ```cleverhans_fixed/```: modified PGD implementation to fix a memory leak bug.
- ```comparison_defenses/```: copies of implementations of [Ensemble Adversarial Training](https://github.com/JZ-LIANG/Ensemble-Adversarial-Training) and [Fast Adversarial Training](https://github.com/locuslab/fast_adversarial).
- ```graphs/```: holds graphs for use in the report.
- ```models/```: defines our custom denoisers, GPEnsemble, and Resnet18 for use in other functions.
- ```trained_models/```: holds trained ```.pth``` models as well as pickle files with validation set accuracies for use in the weighted voting functions in GPEnsemble. Also holds trained models for EnsAdv and FastAdv for us to compare with.
- ```attack.py```: loads test data and evaluates given models against FGSM, PGD and CW attacks.
- ```datasets.py```: provides dataloaders to other functions.
- ```parse_args.py```: provides command line argument parsing capability to other functions.
- ```plot_exp.ipynb```: Jupyter Notebook for processing all the experimental data and producing ensemble and ensemble+denoiser graphs.
- ```run_comparison_experiments.py```: runs all attacks on compared methods (EnsAdv and FastAdv).
- ```run_ensemble_experiments.py```: runs all attacks on a variety of hyperparameter settings for our GPEnsemble method (also includes denoiser+ensemble evaluation)
- ```run_model_training.py```: trains all the prerequisite ensemble members for later use in GPEnsemble.
- ```requirements.txt```: pip packages needed to run all code.

Sample Usage:
```
python run_model_training.py
python run_ensemble_experiments.py
python run_comparison_experiments.py
```


### Denoiser
- ```trained_denoisers/```: holds pretrained denoiser models.
- ```models/denoisers.py```: defines denoiser architectures.
- ```adversarial_dataset.py```: loads and generates custom MNIST and CIFAR10 adversarial dataset for training.
- ```test_denoiser.py```: test denoiser performance by evaluating model accuracy, PSNR, and SSIM of reconstructed images. Also generate FGSM, PGD and CW attacks.
- ```plot_denoiser_exp.ipynb```: plot visualization and results on denoisers
- ```train_denoiser.py```: trains the denoiser models with user-specified parameters. 

Sample Usage:
```
python train_denoiser.py --dataset=cifar10 --arch=dncnn --lr=1e-3 --batch_size=64 --epochs=5
```

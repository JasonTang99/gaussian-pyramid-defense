# Comparison Methods

We compare against implemented versions of [Ensemble Adversarial Training](https://github.com/JZ-LIANG/Ensemble-Adversarial-Training) and [Fast Adversarial Training](https://github.com/locuslab/fast_adversarial).

We ran Ensemble Adversarial Training default settings using 4 of our own baseline resnet18s using this command:

```
python main_ens_adv_train.py \
    --eps 0.0625 \
    --attacker 'fgsm' \
    --loss_schema 'averaged' \
    --dataset 'cifar10'
```
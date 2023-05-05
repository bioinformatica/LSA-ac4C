<div align="center">
<br>
<a href="#"><img src="https://user-images.githubusercontent.com/24775272/235886713-0f5ad0f3-43b6-4d18-b739-d6f80e5667fe.png"/> </a>

  <h4 align="center">A hybrid neutral network incorporating double-layer LSTM and self-attention for the ac4C sites prediction</h4>
</div>

<p align="center">
  <a href="#">
    <img src="https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/torch-2.0.0-blue">
  </a>
  <a href="./LICENSE">
      <img src="https://img.shields.io/badge/license-Apache%202.0-green">
  </a>
</p>

## Intruduction

N4-acetylcytidine (ac4C) is an essential component of the epitranscriptome, which plays a crucial role in regulating mRNA expression by enhancing stability and 
translation efficiency. Numerous studies have shown a correlation be-tween ac4C and the occurrence, progression and prognosis of various types of cancer and diseases.
Therefore, pre-cise prediction of ac4C sites is an important step towards understanding the biological functions of this modification and developing effective
therapeutic interventions. While wet experiments are the primary method for studying ac4C, computational methods have emerged as a promising supplement due to
cost-effectiveness and shorter experi-mental cycles. However, current prediction tools heavily rely on manually selected features and hyperparameters to train models,
which would limit their predictive power and generalization ability. In this study, we proposed a novel procedure, LSA-ac4C, which combines double-layer LSTM and self-attention mechanism to accurately predict ac4C sites. Benchmarking comparisons show that LSA-ac4C has demonstrated superior performance compared to the existing state-of-the-art method, with ACC, MCC and AUROC improving by 2.89%, 5.96% and 1.53%, respectively, on an independent test set. In general, LSA-ac4C represents a potent tool for predicting ac4C sites in human mRNA and will be beneficial for RNA modification research.

## Requirements

LSA-ac4C requires python version 3.7, 3.8, or 3.9. We recommend using the pre-compiled binary wheels available on PyPI. For more details on the specific version numbers of packages we used, see <a href='./Requirements.txt'>Requirements.txt</a>. 


## Getting Started

After installing the required dependencies, you can quickly start with the following command
```
git clone https://github.com/bioinformatica/LSA-ac4C.git
```

## Running the test

Running the following command will predict the example sequence.

```
python predict_LSA-ac4C.py -input_path ./example.fasta -result_path ./
```
<br>

Running the following command will train models.

```
python train_LSA-ac4C.py
```



## Citation
If you use LSA-ac4C in your work, please cite the following paper:

Fei-liao Lai, Feng Gao* (2023) LSA-ac4C: A hybrid neural network incorporating double-layer LSTM and self-attention mechanism for the prediction of N4-acetylcytidine sites in human mRNA. 

BibTeX entry:

```bibtex
@article{LSA-ac4C,
  title={LSA-ac4C: A hybrid neural network incorporating double-layer LSTM and self-attention mechanism for the prediction of N4-acetylcytidine sites in human mRNA},
  author={Fei-liao Lai, Feng Gao},
  year={2023}
}
```

## License
>You can check out the full license [here](./LICENSE)
>
This project is licensed under the terms of the **Apache 2.0** license.



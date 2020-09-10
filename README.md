# Unbiased Differentiable Gradient Descent
This repository contains the code used for the experiments in "Differentiable Unbiased Online Learning to Rank" published at CIKM 2018.

Citation
--------

If you use this code to produce results for your scientific publication, or if you share a copy or fork, please refer to our CIKM 2018 paper:

```
@inproceedings{Oosterhuis2018Unbiased,
  title={Differentiable Unbiased Online Learning to Rank},
  author={Oosterhuis, Harrie and de Rijke, Maarten},
  booktitle={Proceedings of the 2018 ACM on Conference on Information and Knowledge Management},
  year={2018},
  organization={ACM}
}
```

License
-------

The contents of this repository are licensed under the [MIT license](LICENSE). If you modify its contents in any way, please link back to this repository.

Reproducing Experiments
--------
Recreating the results in the paper can be done with the following command:
```
python scripts/CIKM2018.py --data_sets cikm2018 --click_models per nav inf --log_folder testoutput/logs/ --average_folder testoutput/average --output_folder testoutput/fullruns/ --n_runs 125 --n_proc 1 --n_impr 10000
```
This runs all experiments included in the results section of our paper. 
It is up to the user to download the datasets and link to them in the [dataset collections](utils/datasetcollections.py) file.
The output folders including the folder where the data will be stored (in this case testoutput/fullruns/) has to exist before running the code, if folders are missing an error message will indicate this.
Speeding up the simulations can be done by allocating more processes using the *n_proc* flag.

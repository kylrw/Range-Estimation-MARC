from sagemaker.experiments.experiment import _Experiment

exp = _Experiment.load(experiment_name=experiment_name, sagemaker_session=sm_session)
exp._delete_all(action="--force")
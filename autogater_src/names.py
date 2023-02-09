class Names:
    index = "arbitrary_index"
    timepoint = "timepoint"
    temperature = "temperature"
    stain = "stain"
    label = "label"
    label_probs = "label_probs"
    label_preds = "label_predictions"
    cluster_preds = "cluster_predictions"
    # data_file_name = "pipeline_data.csv"
    data_file_name = "sampled_pipeline_data.csv"

    # strains
    yeast = "yeast"
    bacillus = "bacillus"
    ecoli = "ecoli"

    # treatments
    inducer_concentration = "inducer_concentration"
    ethanol = "ethanol"
    heat = "heat"
    treatments_dict = {
        ethanol: {yeast: [0.0, 5.0, 10.0, 12.5, 15.0, 20.0, 80.0],
                  bacillus: [0, 5, 10, 15, 40],
                  ecoli: [0, 5, 10, 15, 40]},
        heat: [0]
    }
    time_points = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]

    # feature columns
    morph_cols = ["FSC-A", "FSC-H", "FSC-W", "SSC-A", "SSC-H", "SSC-W"]
    sytox_cols = ["RL1-A", "RL1-H", "RL1-W"]
    bl_cols = ["BL1-A", "BL1-H", "BL1-W"]
    # morph_cols = ["log_{}".format(x) for x in morph_cols]
    # sytox_cols = ["log_{}".format(x) for x in sytox_cols]
    # bl_cols = ["log_{}".format(x) for x in bl_cols]

    # experiment data dictionary
    # might want to make the values lists of experiment_ids since different experiments could apply
    exp_dict = {
        (yeast, ethanol): "yeast_ethanol",
        (bacillus, ethanol): "bacillus_ethanol",
        (ecoli, ethanol): "ecoli_ethanol"
    }
    # each experiment should have a corresponding folder with the same name as the experiment_id
    # inside the folder you will have data files: dataset, train, test, normalized_train, normalized_test, etc.
    # then the LiveDeadPipeline can call file if it exists or otherwise create it using preprocessing methods
    exp_data_dir = "experiment_data/processed"
    harness_output_dir = "test_harness_outputs"
    pipeline_output_dir = "pipeline_outputs"

    num_live = "num_live"
    num_dead = "num_dead"
    percent_live = "percent_live"

    # labeling methods:
    thresholding_method = "thresholding_method"
    condition_method = "condition_method"
    cluster_method = "cluster_method"
